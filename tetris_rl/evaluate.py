"""Seed-paired evaluation. This module never trains or changes checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=json_default, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         default=json_default, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RandomPolicy:
    """Uniform valid action sampling with a private evaluation-only RNG."""

    def __init__(self, seed: int = 1729):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: dict, deterministic: bool = True, **_: Any) -> int:
        valid = np.flatnonzero(obs["mask"])
        if not len(valid):
            raise RuntimeError("Nonterminal observation has no valid action")
        return int(self.rng.choice(valid))


class HeuristicPolicy:
    def __init__(self):
        from .policy import heuristic_action
        self._choose = heuristic_action

    def predict(self, obs: dict, deterministic: bool = True, **_: Any) -> int:
        return int(self._choose(obs))


def predict_action(policy: Any, obs: dict) -> int:
    """Accept our policy wrapper and SB3-style (action, state) predictions."""
    prediction = policy.predict(obs, deterministic=True) if hasattr(policy, "predict") else policy(obs)
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    action = int(np.asarray(prediction).item())
    mask = np.asarray(obs["mask"], dtype=bool)
    if action < 0 or action >= len(mask) or not mask[action]:
        raise ValueError(f"Policy selected invalid action {action}")
    return action


def public_frame(env: Any) -> dict:
    """Only public game information is saved; hidden bag/RNG state is omitted."""
    if hasattr(env, "public_state"):
        state = env.public_state()
        return json.loads(json.dumps(state, default=json_default))
    from .engine import PIECE_NAMES

    def piece_name(piece: Any) -> str | None:
        return None if piece is None or piece == -1 else (piece if isinstance(piece, str) else PIECE_NAMES[int(piece)])

    return {
        "board": np.asarray(env.board).tolist(),
        "current": piece_name(env.current),
        "hold": piece_name(env.hold),
        "next": [piece_name(piece) for piece in env.next_queue],
        "lines": int(env.lines),
        "pieces": int(env.pieces),
        "input_ticks": int(getattr(env, "input_ticks", 0)),
        "game_time_seconds": float(getattr(env, "input_ticks", 0)) / 60,
    }


def run_episode(policy: Any, seed: int, *, target_lines: int = 40,
                max_pieces: int = 500, record: bool = False,
                reward_config: dict | None = None,
                env_factory: Callable[..., Any] | None = None) -> tuple[dict, dict | None]:
    from .env import TetrisEnv
    env_factory = env_factory or TetrisEnv
    started = time.perf_counter()
    env_kwargs = {"target_lines": target_lines, "max_pieces": max_pieces}
    if reward_config is not None:
        env_kwargs["reward_config"] = reward_config
    env = env_factory(**env_kwargs)
    inference_seconds = 0.0
    rewards: list[float] = []
    actions: list[dict] = []
    frames: list[dict] = []
    try:
        obs, info = env.reset(seed=int(seed))
        if record:
            frames.append(public_frame(env))
        terminated = truncated = False
        while not (terminated or truncated):
            tick = time.perf_counter()
            action = predict_action(policy, obs)
            inference_seconds += time.perf_counter() - tick
            if record:
                candidate = env.candidate(action)
                actions.append({
                    "action": action, "inputs": list(env.action_path(action)),
                    "piece": int(candidate["piece"]),
                    "rotation": int(candidate["rotation"]),
                    "x": int(candidate["x"]), "y": int(candidate["y"]),
                    "hold": bool(candidate["hold"]),
                })
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            if record:
                frames.append(public_frame(env))
            if len(rewards) > max_pieces:
                raise RuntimeError("Environment exceeded its maximum placement limit")
        elapsed = time.perf_counter() - started
        success = bool(info.get("success", False))
        episode = {
            "seed": int(seed), "success": success,
            "reason": str(info.get("reason", "success" if success else "unknown")),
            "lines": int(info.get("lines", getattr(env, "lines", 0))),
            "pieces": int(info.get("pieces", len(rewards))),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "input_ticks": int(info.get("input_ticks", getattr(env, "input_ticks", 0))),
            "game_time_seconds": float(info.get("game_time_seconds", getattr(env, "input_ticks", 0) / 60)),
            "episode_return": float(info.get("episode_return", sum(rewards))),
            "inference_seconds": inference_seconds,
            "wall_seconds": elapsed,
        }
        replay = None
        if record:
            replay = {
                "format_version": 1, "seed": int(seed), "target_lines": target_lines,
                "max_pieces": max_pieces, "hidden_rows": 4,
                "reward_config": dict(env.reward_config),
                "game_time_definition": "One input token = one 1/60 second tick; no real-time gravity or line-clear delay.",
                "episode": episode, "frames": frames, "actions": actions,
            }
        return episode, replay
    finally:
        env.close()


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    if not n:
        raise ValueError("At least one evaluation episode is required")
    successes = [e for e in episodes if e["success"]]
    failures = [e for e in episodes if not e["success"]]
    count = len(successes)
    rate = count / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator

    def quantile(group: list[dict], key: str, percentile: float) -> float | None:
        return float(np.percentile([e[key] for e in group], percentile)) if group else None

    return {
        "success_rate": rate, "successes": count, "episodes": n,
        "success_rate_wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
        "successful_pieces_median": quantile(successes, "pieces", 50),
        "successful_pieces_p90": quantile(successes, "pieces", 90),
        "successful_game_time_seconds_median": quantile(successes, "game_time_seconds", 50),
        "successful_game_time_seconds_p90": quantile(successes, "game_time_seconds", 90),
        "failure_lines_median": quantile(failures, "lines", 50),
        "failure_lines_p90": quantile(failures, "lines", 90),
        "failure_reasons": dict(sorted(Counter(e["reason"] for e in failures).items())),
        "all_lines_mean": float(np.mean([e["lines"] for e in episodes])),
        "all_episode_return_mean": float(np.mean([e["episode_return"] for e in episodes])),
        "all_pieces": sum(e["pieces"] for e in episodes),
        "all_input_ticks": sum(e["input_ticks"] for e in episodes),
        "total_inference_seconds": sum(e["inference_seconds"] for e in episodes),
        "total_episode_wall_seconds": sum(e["wall_seconds"] for e in episodes),
    }


def evaluate_policy(policy: Any, seeds: Iterable[int], *, target_lines: int = 40,
                    max_pieces: int = 500, agent_name: str = "model", capture: int = 0,
                    reward_config: dict | None = None,
                    env_factory: Callable[..., Any] | None = None,
                    progress: bool = False, episode_log: str | Path | None = None) -> dict:
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique")
    episodes, replays = [], []
    start = time.perf_counter()
    stream = None
    if episode_log:
        Path(episode_log).parent.mkdir(parents=True, exist_ok=True)
        stream = Path(episode_log).open("w", encoding="utf-8")
    try:
        for index, seed in enumerate(seeds):
            # Action randomness is independent of the environment's RNG and future bag.
            episode_policy = RandomPolicy(1729 + index) if isinstance(policy, RandomPolicy) else policy
            episode, replay = run_episode(episode_policy, seed, target_lines=target_lines,
                                          max_pieces=max_pieces, record=index < capture,
                                          reward_config=reward_config,
                                          env_factory=env_factory)
            episodes.append(episode)
            if stream:
                stream.write(json.dumps(episode, default=json_default, allow_nan=False) + "\n")
                stream.flush()
            if replay:
                replay["agent"] = agent_name
                replays.append(replay)
            if progress and (index == 0 or (index + 1) % 10 == 0 or index + 1 == len(seeds)):
                good = sum(e["success"] for e in episodes)
                print(f"{agent_name}: {index + 1}/{len(seeds)} episodes, {good} successes, "
                      f"{time.perf_counter() - start:.1f}s", flush=True)
    finally:
        if stream:
            stream.close()
    return {"agent": agent_name, "summary": summarize(episodes), "episodes": episodes,
            "replays": replays, "evaluation_wall_seconds": time.perf_counter() - start}


def parse_checkpoint_map(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            name, path = "model", value
        if name in result:
            raise ValueError(f"Checkpoint name repeated: {name}")
        result[name] = Path(path).resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", default=["random", "heuristic", "model"])
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="PATH for model, or NAME=PATH; may be repeated")
    parser.add_argument("--seeds", type=Path, help="JSON list, JSON object with seeds, or one seed per line")
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-set-name", default="validation")
    parser.add_argument("--target-lines", type=int, default=40)
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=int, default=1, help="Save replay for first N seeds per agent")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--config", type=Path, help="Optional JSON config included in provenance")
    parser.add_argument("--reward-config", type=Path, help="JSON reward parameters actually applied to every agent")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.torch_threads)
    if args.seeds:
        raw = args.seeds.read_text(encoding="utf-8-sig")
        try:
            loaded = json.loads(raw)
            seeds = loaded["seeds"] if isinstance(loaded, dict) else loaded
        except json.JSONDecodeError:
            seeds = [int(line.strip()) for line in raw.splitlines() if line.strip()]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("A nonempty list of unique seeds is required")
    checkpoints = parse_checkpoint_map(args.checkpoint)
    from .env import DEFAULT_REWARD
    reward_config = dict(DEFAULT_REWARD)
    if args.reward_config:
        reward_config.update(json.loads(args.reward_config.read_text(encoding="utf-8-sig")))
    run_config = {"target_lines": args.target_lines, "max_pieces": args.max_pieces,
                  "deterministic_policy": True, "seed_set_name": args.seed_set_name,
                  "seeds": seeds, "reward_config": reward_config}
    if args.config:
        run_config["source_config"] = json.loads(args.config.read_text(encoding="utf-8-sig"))
    result = {"format_version": 1, "seed_set_name": args.seed_set_name, "seeds": seeds,
              "seed_sha256": fingerprint(seeds), "evaluation_config": run_config,
              "config_sha256": fingerprint(run_config),
              "source_sha256": {name: file_sha256(Path(__file__).with_name(name))
                                for name in ("engine.py", "env.py", "policy.py", "evaluate.py")},
              "game_time_definition": "Input token count / 60; simulation excludes timed gravity, DAS/ARR and line-clear delay.",
              "results": {}}
    for name in args.agents:
        metadata = {}
        if name == "random":
            policy = RandomPolicy()
        elif name == "heuristic":
            policy = HeuristicPolicy()
        else:
            from .policy import load_policy
            if name not in checkpoints:
                parser.error(f"Missing --checkpoint {name}=PATH")
            checkpoint = checkpoints[name]
            policy = load_policy(checkpoint, device=args.device)
            metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                        "policy_metadata": getattr(policy, "metadata", {})}
        evaluated = evaluate_policy(policy, seeds, target_lines=args.target_lines,
                                    max_pieces=args.max_pieces, agent_name=name,
                                    reward_config=reward_config,
                                    capture=args.capture, progress=True,
                                    episode_log=args.output.with_name(f"{args.output.stem}_{name}_episodes.jsonl"))
        for replay in evaluated.pop("replays"):
            replay.update(metadata)
            replay["seed_set_name"] = args.seed_set_name
            replay["config_sha256"] = result["config_sha256"]
            replay_path = args.output.parent / "replays" / f"{name}_{replay['seed']}.json"
            write_json(replay_path, replay)
            from .replay import build_html
            build_html(replay_path, replay_path.with_suffix(".html"))
        evaluated.update(metadata)
        result["results"][name] = evaluated
        write_json(args.output, result)
    print(json.dumps({name: entry["summary"] for name, entry in result["results"].items()},
                     ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
