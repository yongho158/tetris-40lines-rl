"""Live games must use fresh inference and animate the actual engine input path."""
from __future__ import annotations

import copy
import http.client
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from tetris_rl.engine import PIECE_NAMES, SHAPES, STATES, encode_action, fits, replay_path
from tetris_rl.env import TetrisEnv
from tetris_rl.evaluate import run_episode
from tetris_rl.live import APIError, LiveApplication, advance_with_frames, live_state, make_server
from tetris_rl.policy import load_policy


class FirstValid:
    def __init__(self):
        self.calls = 0

    def predict(self, obs, deterministic=True):
        assert set(obs) == {"board", "context", "candidates", "mask"}
        assert obs["context"].shape == (12,)
        self.calls += 1
        return int(np.flatnonzero(obs["mask"])[0])


def make_app(**kwargs):
    return LiveApplication(FirstValid(), {"checkpoint": "test", "sha256": "test", "kind": "test"}, **kwargs)


@pytest.mark.parametrize("held", [-1, 4])
def test_hold_frames_consume_correct_queue_and_only_one_input_tick(held):
    env = TetrisEnv()
    env.reset(seed=3_000_003)
    env.hold = held
    env._refresh_candidates()
    old = copy.deepcopy(env)
    action = int(np.flatnonzero(env.action_masks()[STATES:])[0]) + STATES
    expected_current = old.hold if held >= 0 else old._take_next()
    expected_next = [PIECE_NAMES[p] for p in old.next_queue]
    outgoing = old.current
    _, frames, decision = advance_with_frames(env, action)
    assert decision["hold"] and frames[0]["input"] == "hold"
    for frame in frames[:-1]:
        state = frame["state"]
        assert state["current"] == PIECE_NAMES[expected_current]
        assert state["hold"] == PIECE_NAMES[outgoing]
        assert state["next"] == expected_next
        assert state["hold_available"] is False
    assert frames[0]["state"]["input_ticks"] == 1
    assert frames[-1]["state"]["hold_available"] is True
    assert frames[-1]["input"] == "lock"
    assert frames[-1]["state"]["input_ticks"] == len(decision["inputs"])
    assert frames[-2]["state"]["input_ticks"] == frames[-1]["state"]["input_ticks"]


def test_frames_replay_every_reachable_empty_board_placement_and_preserve_rules():
    template = TetrisEnv()
    template.reset(seed=37)
    for action in np.flatnonzero(template.action_masks()):
        env = copy.deepcopy(template)
        before = env.board.copy()
        expected = env.candidate(action)
        _, frames, decision = advance_with_frames(env, int(action))
        piece = expected["piece"]
        tokens = [v for v in decision["inputs"] if v != "hold"]
        for count, frame in enumerate([f for f in frames if f["input"] not in ("hold", "lock")], 1):
            prefix = tokens[:count]
            rotation, x, ghost_y = replay_path(
                before, piece, prefix if prefix[-1] == "hard_drop" else [*prefix, "hard_drop"])
            active = frame["state"]["active"]
            assert (active["rotation"], active["x"]) == (rotation, x)
            assert fits(before, piece, rotation, x, active["y"])
            assert active["cells"] == [[int(x + dx), int(active["y"] + dy)] for dx, dy in SHAPES[piece, rotation]]
            assert frame["state"]["ghost_cells"] == [[int(x + dx), int(ghost_y + dy)] for dx, dy in SHAPES[piece, rotation]]
            assert frame["state"]["board"] == before.tolist()
        assert frames[-2]["input"] == "hard_drop"
        assert frames[-2]["state"]["active"]["cells"] == frames[-2]["state"]["ghost_cells"]
        np.testing.assert_array_equal(env.board, expected["board_after"])
        assert frames[-1]["state"] == live_state(env)


@pytest.mark.parametrize("reason", ["success", "lock_out"])
def test_terminal_lock_frames_clear_active_piece(reason):
    env = TetrisEnv(target_lines=1)
    env.reset(seed=37)
    if reason == "success":
        env.current = 0
        env.board[23, :] = 2
        env.board[23, 3:7] = 0
        action = encode_action(0, 0, 3, 22)
    else:
        env.current = 3
        env.board[4:, :9] = 1
        action = encode_action(0, 0, 3, 2)
    env._refresh_candidates()
    _, frames, _ = advance_with_frames(env, action)
    assert frames[-2]["state"]["active"] is not None
    assert frames[-1]["state"]["active"] is None
    assert frames[-1]["state"]["done"]
    assert frames[-1]["state"]["reason"] == reason


def test_sessions_infer_on_demand_and_reject_stale_or_terminal_steps():
    app = make_app(max_pieces=1)
    one, two = app.new(3000000), app.new(3000001)
    assert app.policy.calls == 0
    result = app.step(one["session_id"], 0)
    assert app.policy.calls == 1
    assert result["revision"] == result["state"]["pieces"] == 1
    assert result["state"]["done"]
    assert result["state"]["reason"] == "piece_limit"
    assert result["state"]["current"] is None
    assert result["state"]["active"] is None
    assert result["state"]["ghost_cells"] == []
    assert app.sessions[two["session_id"]].revision == 0
    for revision, code in [(0, "stale_revision"), (1, "game_finished")]:
        with pytest.raises(APIError) as error:
            app.step(one["session_id"], revision)
        assert error.value.status == 409 and error.value.code == code
    assert app.policy.calls == 1
    app.close()


def test_simultaneous_same_revision_advances_only_once():
    app = make_app()
    game = app.new(3000000)
    barrier = threading.Barrier(3)
    results = []

    def step():
        barrier.wait()
        try:
            results.append(app.step(game["session_id"], 0)["revision"])
        except APIError as exc:
            results.append(exc.code)

    threads = [threading.Thread(target=step) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert set(results) == {1, "stale_revision"}
    assert app.policy.calls == 1


def test_session_bounds_expiration_and_seed_validation():
    app = make_app(max_sessions=2)
    first = app.new(1)
    second = app.new(2)
    app.new(3)
    assert len(app.sessions) == 2
    with pytest.raises(APIError, match="expired"):
        app.step(first["session_id"], 0)
    app.sessions[second["session_id"]].touched = 0
    with pytest.raises(APIError, match="expired"):
        app.step(second["session_id"], 0)
    for seed in [-1, True, 1.5, "5", 2**53]:
        with pytest.raises(APIError) as error:
            app.new(seed)
        assert error.value.code == "invalid_seed"
    assert type(app.new(None)["seed"]) is int


def test_frozen_model_fresh_live_game_equals_direct_evaluation():
    checkpoint = Path(__file__).resolve().parents[1] / "artifacts" / "final_model.pt"
    policy = load_policy(checkpoint)
    seed = 3000071
    direct, replay = run_episode(policy, seed, record=True)
    app = LiveApplication(policy, {"checkpoint": checkpoint.name})
    game = app.new(seed)
    session_id = game["session_id"]
    actions = []
    all_inputs = 0
    revision = 0
    while not game["state"]["done"]:
        game = app.step(session_id, revision)
        revision = game["revision"]
        actions.append(game["decision"]["action"])
        all_inputs += len(game["decision"]["inputs"])
        assert game["frames"][-1]["input"] == "lock"
        assert game["stats"]["placements"] == revision
    assert actions == [action["action"] for action in replay["actions"]]
    assert game["state"]["board"] == replay["frames"][-1]["board"]
    for key in ("reason", "lines", "pieces", "input_ticks", "game_time_seconds"):
        assert game["state"][key] == direct[key]
    assert all_inputs == direct["input_ticks"]
    assert game["state"]["reason"] == "success"


@pytest.fixture
def http_server(tmp_path):
    html = tmp_path / "live.html"
    html.write_text("<!doctype html><title>Live test</title>", encoding="utf-8")
    app = make_app()
    server = make_server(app, port=0, html_path=html)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield app, server.server_address[1]
    server.shutdown()
    server.server_close()
    worker.join(timeout=5)
    app.close()


def request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        headers = {"Content-Type": "application/json", **(headers or {})}
        raw = json.dumps(body) if body is not None else None
        connection.request(method, path, body=raw, headers=headers)
        response = connection.getresponse()
        encoded = response.read()
        data = json.loads(encoded) if response.getheader("Content-Type").startswith("application/json") else encoded
        return response.status, data
    finally:
        connection.close()


def test_http_new_step_status_and_error_contract(http_server):
    app, port = http_server
    status, html = request(port, "GET", "/")
    assert status == 200 and b"Live test" in html
    status, data = request(port, "GET", "/api/status")
    assert status == 200 and data["mode"] == "live_inference"
    status, game = request(port, "POST", "/api/new", {"seed": 3000000})
    assert status == 200 and game["revision"] == 0
    args = {"session_id": game["session_id"], "revision": 0}
    status, step = request(port, "POST", "/api/step", args)
    assert status == 200 and step["revision"] == 1
    status, error = request(port, "POST", "/api/step", args)
    assert status == 409 and error["code"] == "stale_revision"
    assert app.policy.calls == 1


@pytest.mark.parametrize("headers,expected", [
    ({"Origin": "https://example.com"}, "invalid_origin"),
    ({"Host": "example.com"}, "invalid_host"),
    ({"Sec-Fetch-Site": "cross-site"}, "cross_site"),
    ({"Content-Type": "text/plain"}, "content_type"),
])
def test_http_rejects_nonlocal_mutations(http_server, headers, expected):
    app, port = http_server
    status, error = request(port, "POST", "/api/new", {"seed": 1}, headers)
    assert status in (403, 415) and error["code"] == expected
    assert len(app.sessions) == 0


def test_http_rejects_oversized_or_nonobject_json(http_server):
    app, port = http_server
    status, error = request(port, "POST", "/api/new", {"padding": "x" * 4096})
    assert status == 413 and error["code"] == "body_size"
    status, error = request(port, "POST", "/api/new", [1, 2])
    assert status == 400 and error["code"] == "invalid_json"
    assert len(app.sessions) == 0
