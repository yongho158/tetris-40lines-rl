"""Local, on-demand AI games with engine-verified input animation frames."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import webbrowser

from .engine import PIECE_NAMES, SHAPES, SPAWN_X, SPAWN_Y, fits, rotated
from .env import TetrisEnv
from .evaluate import file_sha256, predict_action


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def validate_seed(seed: Any) -> int | None:
    if seed is None:
        return None
    if type(seed) is not int or not 0 <= seed <= 2**53 - 1:
        raise APIError(400, "invalid_seed", "seed must be an integer from 0 to 9007199254740991, or null")
    return seed


def active_geometry(board, piece: int, rotation=0, x=SPAWN_X, y=SPAWN_Y) -> tuple[dict, list]:
    def cells(at_y):
        return [[int(x + dx), int(at_y + dy)] for dx, dy in SHAPES[piece, rotation]]

    ghost_y = y
    while fits(board, piece, rotation, x, ghost_y + 1):
        ghost_y += 1
    return ({"piece": PIECE_NAMES[piece], "rotation": int(rotation),
             "x": int(x), "y": int(y), "cells": cells(y)}, cells(ghost_y))


def live_state(env: TetrisEnv) -> dict:
    state = env.public_state()
    state.update(done=bool(env._done), reason=env._reason,
                 target_lines=env.target_lines, max_pieces=env.max_pieces)
    if env._done:
        state.update(current=None, active=None, ghost_cells=[])
    else:
        state["active"], state["ghost_cells"] = active_geometry(env.board, env.current)
    return state


def advance_with_frames(env: TetrisEnv, action: int) -> tuple[dict, list[dict], dict]:
    """Replay one reachable path, then commit exactly one env.step.

    The engine owns locking, draws, line clearing and termination. Animation
    replays the same SRS primitives and never changes a policy observation.
    Empty hold reveals one queue item during this action; reconstruct that
    intermediate public queue from the committed environment's public queue.
    """
    candidate = env.candidate(action)
    path = candidate["path"]
    before = live_state(env)
    board = env.board.copy()
    piece, held = env.current, env.hold
    rotation, x, y = 0, SPAWN_X, SPAWN_Y
    empty_hold = candidate["hold"] and held < 0
    hold_available = True
    frames = []
    for index, token in enumerate(path, 1):
        if token == "hold":
            if not hold_available or index != 1:
                raise AssertionError("Reachable path contains an invalid hold")
            piece, held = (held if held >= 0 else env.next_queue[0]), piece
            rotation, x, y = 0, SPAWN_X, SPAWN_Y
            hold_available = False
        elif token in ("left", "right", "down"):
            x += int(token == "right") - int(token == "left")
            y += int(token == "down")
        elif token in ("cw", "ccw"):
            ok, rotation, x, y = rotated(board, piece, rotation, x, y,
                                         1 if token == "cw" else -1)
            if not ok:
                raise AssertionError("Reachable path rotation failed")
        elif token == "hard_drop":
            if index != len(path):
                raise AssertionError("Hard drop must end the input path")
            while fits(board, piece, rotation, x, y + 1):
                y += 1
        else:
            raise AssertionError(f"Unknown engine input: {token}")
        if not fits(board, piece, rotation, x, y):
            raise AssertionError("Reachable path input collided")
        active, ghost = active_geometry(board, piece, rotation, x, y)
        ticks = before["input_ticks"] + index
        state = {**before, "current": PIECE_NAMES[piece],
                 "hold": PIECE_NAMES[held] if held >= 0 else None,
                 "hold_available": hold_available, "active": active,
                 "ghost_cells": ghost, "input_ticks": ticks,
                 "game_time_seconds": ticks / 60.0}
        frames.append({"input": token, "state": state})
    if not path or path[-1] != "hard_drop":
        raise AssertionError("Reachable placement has no final hard drop")
    if (piece, rotation, x, y) != (candidate["piece"], candidate["rotation"],
                                  candidate["x"], candidate["y"]):
        raise AssertionError("Animated path does not reach selected candidate")
    obs, _, _, _, _ = env.step(action)
    after = live_state(env)
    if after["input_ticks"] != frames[-1]["state"]["input_ticks"]:
        raise AssertionError("Animation input count differs from engine")
    if after["board"] != candidate["board_after"].tolist():
        raise AssertionError("Animation candidate differs from committed board")
    if empty_hold:
        # After hold: [new current after lock, first four next after lock].
        # env.current is retained even on terminal states; UI current is null.
        intermediate_next = [PIECE_NAMES[env.current], *after["next"][:4]]
        for frame in frames:
            frame["state"]["next"] = intermediate_next
    frames.append({"input": "lock", "state": after})
    decision = {"action": int(action), "inputs": list(path),
                "piece": candidate["piece_name"], "hold": candidate["hold"]}
    return obs, frames, decision


@dataclass
class Session:
    env: TetrisEnv
    obs: dict
    seed: int
    revision: int = 0
    total_inference_ms: float = 0.0
    total_compute_ms: float = 0.0
    touched: float = 0.0


class LiveApplication:
    """Bounded in-memory games; all state mutation and inference are serialized."""

    def __init__(self, policy, model: dict, *, default_seed=None,
                 max_sessions=32, session_ttl=3600, target_lines=40, max_pieces=500):
        if max_sessions < 1 or session_ttl <= 0:
            raise ValueError("Session bounds must be positive")
        self.policy, self.model = policy, model
        self.default_seed = validate_seed(default_seed)
        self.max_sessions, self.session_ttl = max_sessions, session_ttl
        self.target_lines, self.max_pieces = target_lines, max_pieces
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.lock = threading.RLock()

    def warmup(self):
        env = TetrisEnv(target_lines=self.target_lines, max_pieces=self.max_pieces)
        try:
            obs, _ = env.reset(seed=0)
            predict_action(self.policy, obs)
            active_geometry(env.board, env.current)
        finally:
            env.close()

    def status(self):
        return {"model": self.model, "default_seed": self.default_seed,
                "target_lines": self.target_lines, "max_pieces": self.max_pieces,
                "hidden_rows": 4, "input_hz": 60,
                "mode": "live_inference", "ready": True}

    def _expire(self):
        cutoff = time.monotonic() - self.session_ttl
        for session_id in list(self.sessions):
            if self.sessions[session_id].touched < cutoff:
                self.sessions.pop(session_id).env.close()

    def new(self, seed=None):
        seed = validate_seed(seed)
        if seed is None:
            seed = secrets.randbits(32)
        with self.lock:
            self._expire()
            while len(self.sessions) >= self.max_sessions:
                _, old = self.sessions.popitem(last=False)
                old.env.close()
            env = TetrisEnv(target_lines=self.target_lines, max_pieces=self.max_pieces)
            obs, _ = env.reset(seed=seed)
            session_id = secrets.token_urlsafe(24)
            self.sessions[session_id] = Session(env, obs, seed, touched=time.monotonic())
            return {"session_id": session_id, "revision": 0, "seed": seed,
                    "state": live_state(env), "model": self.model}

    def step(self, session_id, revision):
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 64:
            raise APIError(400, "invalid_session", "session_id must be a session token")
        if type(revision) is not int or revision < 0:
            raise APIError(400, "invalid_revision", "revision must be a nonnegative integer")
        with self.lock:
            self._expire()
            session = self.sessions.get(session_id)
            if session is None:
                raise APIError(404, "session_missing", "Game expired or was evicted; start a new game")
            if revision != session.revision:
                raise APIError(409, "stale_revision", "Game already advanced; start a new game to resynchronize")
            if session.env._done:
                raise APIError(409, "game_finished", "Game has ended; start a new game")
            started = time.perf_counter()
            action = predict_action(self.policy, session.obs)
            inference_ms = (time.perf_counter() - started) * 1000
            obs, frames, decision = advance_with_frames(session.env, action)
            compute_ms = (time.perf_counter() - started) * 1000
            session.obs = obs
            session.revision += 1
            session.total_inference_ms += inference_ms
            session.total_compute_ms += compute_ms
            session.touched = time.monotonic()
            self.sessions.move_to_end(session_id)
            decision["inference_ms"] = inference_ms
            return {"session_id": session_id, "revision": session.revision,
                    "frames": frames, "state": frames[-1]["state"], "decision": decision,
                    "stats": {"inference_ms": inference_ms,
                              "total_inference_ms": session.total_inference_ms,
                              "placements": session.env.pieces, "compute_ms": compute_ms,
                              "total_compute_ms": session.total_compute_ms}}

    def close(self):
        with self.lock:
            for session in self.sessions.values():
                session.env.close()
            self.sessions.clear()


def make_server(application: LiveApplication, port=8766, *, html_path=None):
    html_path = Path(html_path) if html_path else Path(__file__).with_name("live.html")

    class Handler(BaseHTTPRequestHandler):
        server_version = "TetrisLive/1.0"

        def _check_request(self):
            port = self.server.server_address[1]
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            host = self.headers.get("Host", "").lower()
            if host not in allowed_hosts:
                raise APIError(403, "invalid_host", "Only localhost access is supported")
            origin = self.headers.get("Origin")
            if origin is not None and origin.lower() != f"http://{host}":
                raise APIError(403, "invalid_origin", "Cross-origin access is not allowed")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise APIError(403, "cross_site", "Cross-site access is not allowed")

        def _send(self, status, value, content_type="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(
                value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _error(self, exc):
            self._send(exc.status, {"error": str(exc), "code": exc.code})

        def do_GET(self):
            try:
                self._check_request()
                path = urlsplit(self.path).path
                if path == "/":
                    self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/status":
                    self._send(200, application.status())
                elif path == "/favicon.ico":
                    self._send(204, b"")
                else:
                    raise APIError(404, "not_found", "Route not found")
            except APIError as exc:
                self._error(exc)
            except OSError:
                self._error(APIError(500, "page_missing", "Live player HTML could not be read"))

        def do_POST(self):
            try:
                self._check_request()
                if self.headers.get_content_type() != "application/json":
                    raise APIError(415, "content_type", "Use application/json")
                if self.headers.get("Transfer-Encoding"):
                    raise APIError(400, "transfer_encoding", "Chunked requests are not supported")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise APIError(400, "invalid_length", "Invalid Content-Length") from None
                if not 0 < length <= 4096:
                    raise APIError(413, "body_size", "JSON request must be between 1 and 4096 bytes")
                try:
                    body = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeError):
                    raise APIError(400, "invalid_json", "Invalid JSON body") from None
                if not isinstance(body, dict):
                    raise APIError(400, "invalid_json", "JSON body must be an object")
                path = urlsplit(self.path).path
                if path == "/api/new":
                    result = application.new(body.get("seed"))
                elif path == "/api/step":
                    result = application.step(body.get("session_id"), body.get("revision"))
                else:
                    raise APIError(404, "not_found", "Route not found")
                self._send(200, result)
            except APIError as exc:
                self._error(exc)
            except Exception as exc:
                self.log_error("Live inference failed: %s", exc)
                self._error(APIError(500, "inference_failed", "Live inference failed; see server terminal"))

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            # Avoid a console line per placement; errors retain standard logging.
            pass

        def log_error(self, format, *args):
            print(f"Live server: {format % args}", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/final_model.pt"))
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--seed", type=int, help="Initial seed shown in the player; blank selects a random game")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the player")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        validate_seed(args.seed)
    except APIError as exc:
        parser.error(str(exc))
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if not Path(__file__).with_name("live.html").is_file():
        parser.error("live.html is missing from the tetris_rl package")
    import torch
    from .policy import load_policy
    torch.set_num_threads(1)
    print(f"Loading {checkpoint.name} and warming the game engine...", flush=True)
    policy = load_policy(checkpoint, device="cpu")
    model = {"checkpoint": checkpoint.name, "sha256": file_sha256(checkpoint),
             "kind": type(policy).__name__}
    application = LiveApplication(policy, model, default_seed=args.seed)
    application.warmup()
    try:
        server = make_server(application, args.port)
    except OSError as exc:
        application.close()
        parser.error(f"Cannot bind 127.0.0.1:{args.port}: {exc}. Choose another --port.")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Live AI ready: {url}", flush=True)
    print("Each placement is inferred on demand. Ctrl+C stops the server.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nLive server stopped.", flush=True)
    finally:
        server.server_close()
        application.close()


if __name__ == "__main__":
    main()
