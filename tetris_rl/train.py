"""Budgeted, reproducible policy-search RL and masked PPO training entry point."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .budget import Budget
from .env import TetrisEnv
from .policy import (CandidateMaskablePolicy, DELLACHERIE_WEIGHTS,
                     LinearCandidatePolicy, PPOPolicyAdapter, load_policy)


class BudgetExhausted(RuntimeError):
    pass


class RunBudget:
    def __init__(self, budget, steps):
        self.shared = budget
        self.limit = int(steps)
        self.used = 0

    def reserve(self):
        if self.used >= self.limit or not self.shared.record(1):
            raise BudgetExhausted("Training/validation transition or wall-clock budget exhausted")
        self.used += 1

    @property
    def exhausted(self):
        return self.used >= self.limit or self.shared.exhausted


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def append_log(path, data):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(episodes):
    successes = [e for e in episodes if e["success"]]
    failures = [e for e in episodes if not e["success"]]
    pieces = [e["pieces"] for e in successes]
    return {
        "episodes": len(episodes), "successes": len(successes),
        "success_rate": len(successes) / max(len(episodes), 1),
        "success_pieces_mean": float(np.mean(pieces)) if pieces else None,
        "success_pieces_median": float(np.median(pieces)) if pieces else None,
        "success_pieces_p90": float(np.percentile(pieces, 90)) if pieces else None,
        "mean_lines": float(np.mean([e["lines"] for e in episodes])) if episodes else 0.,
        "failure_lines_mean": float(np.mean([e["lines"] for e in failures])) if failures else None,
        "mean_return": float(np.mean([e["return"] for e in episodes])) if episodes else 0.,
    }


def selection_key(metrics):
    successful_pieces = metrics["success_pieces_mean"]
    return (metrics["success_rate"],
            -successful_pieces if successful_pieces is not None else metrics["mean_lines"],
            metrics["mean_lines"], metrics["mean_return"])


def run_episodes(policy, seeds, budget: RunBudget, *, target_lines=40, reward_config=None):
    env = TetrisEnv(target_lines=target_lines, max_pieces=500, reward_config=reward_config)
    episodes = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            total_reward = 0.
            started = time.perf_counter()
            while True:
                action = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
                budget.reserve()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    episodes.append({
                        "seed": int(seed), "success": bool(info.get("success", False)),
                        "reason": info.get("reason", "unknown"), "pieces": int(info["pieces"]),
                        "lines": int(info["lines"]), "return": total_reward,
                        "input_ticks": int(info.get("input_ticks", 0)),
                        "game_time_seconds": float(info.get("game_time_seconds", 0.)),
                        "wall_seconds": time.perf_counter() - started,
                    })
                    break
    finally:
        env.close()
    return summarize(episodes), episodes


def load_validation_seeds(args):
    if args.validation_seeds:
        obj = json.loads(Path(args.validation_seeds).read_text(encoding="utf-8-sig"))
        seeds = obj.get("validation", obj.get("seeds")) if isinstance(obj, dict) else obj
    else:
        seeds = list(range(1_000_000, 1_000_000 + args.validation_episodes))
    seeds = [int(x) for x in seeds]
    if not seeds or any(not 1_000_000 <= x < 2_000_000 for x in seeds):
        raise ValueError("Validation seeds must lie in [1000000, 2000000), separate from training/test")
    return seeds[:args.validation_episodes]


def create_reference_checkpoints(run_dir, rng):
    untrained = run_dir / "untrained.pt"
    initialized = run_dir / "initialized.pt"
    # Keep training RNG consumption identical whether reference files already exist.
    random_weights = rng.normal(0, 0.1, 14)
    if not untrained.exists():
        LinearCandidatePolicy(random_weights).save(untrained, {
            "stage": "untrained", "rl_updated": False, "source": "seeded random linear weights"})
    if not initialized.exists():
        LinearCandidatePolicy(DELLACHERIE_WEIGHTS).save(initialized, {
            "stage": "heuristic_initialization", "rl_updated": False,
            "source": "explicit Dellacherie six-feature coefficients; hold cost -0.001",
            "behavior_cloning": False})


def incumbent_validation(run_dir, checkpoint_name, report_name, validation_seeds, budget, reward_config):
    """Keep prior best on resume; compare scores only on identical validation seeds."""
    checkpoint, report = run_dir / checkpoint_name, run_dir / report_name
    if not checkpoint.exists() or not report.exists():
        return None
    saved = json.loads(report.read_text(encoding="utf-8"))
    old_seeds = [episode["seed"] for episode in saved.get("episodes", [])]
    if old_seeds == validation_seeds:
        return saved["metrics"]
    metrics, episodes = run_episodes(load_policy(checkpoint), validation_seeds, budget,
                                     reward_config=reward_config)
    saved.update(metrics=metrics, episodes=episodes)
    saved["revalidated_due_to_seed_set_change"] = True
    write_json(report, saved)
    return metrics


def inherit_cem_incumbent(source_dir, run_dir):
    """A resumed search must carry its best policy, not just its scalar score."""
    source_dir, run_dir = Path(source_dir).resolve(), Path(run_dir).resolve()
    if source_dir == run_dir:
        return
    source_checkpoint, source_report = source_dir / "best.pt", source_dir / "best_validation.json"
    checkpoint, report = run_dir / "best.pt", run_dir / "best_validation.json"
    # An existing complete destination incumbent is handled by incumbent_validation.
    if checkpoint.exists() and report.exists():
        return
    if not source_checkpoint.exists() or not source_report.exists():
        return  # No score is inherited without its actual checkpoint.
    shutil.copy2(source_checkpoint, checkpoint)
    saved = json.loads(source_report.read_text(encoding="utf-8"))
    saved["inherited_from"] = str(source_checkpoint)
    write_json(report, saved)


def restore_torch_rng(state):
    """Older checkpoints remain loadable but cannot reproduce their Torch RNG stream."""
    if state is not None and "torch_rng_state" in state:
        torch.set_rng_state(torch.tensor(state["torch_rng_state"], dtype=torch.uint8))
        return True
    return False


def cem_train(args, run_dir, budget, validation_seeds, reward_config):
    """Cross-entropy method: sample policies, rank episode return, update distribution."""
    rng = np.random.default_rng(args.seed)
    create_reference_checkpoints(run_dir, rng)
    initial = DELLACHERIE_WEIGHTS.astype(np.float64)
    initial /= np.linalg.norm(initial)
    mean = initial.copy()
    std = np.maximum(np.abs(initial) * args.sigma, 0.008)
    generation = 0
    best_metrics = None
    if args.resume:
        resume = Path(args.resume)
        if resume.is_dir():
            resume = resume / "cem_state.json"
        if resume.suffix == ".json":
            state = json.loads(resume.read_text(encoding="utf-8"))
            mean = np.asarray(state["mean"], dtype=np.float64)
            std = np.asarray(state["std"], dtype=np.float64)
            rng.bit_generator.state = state["rng_state"]
            generation = state["generation"]
            inherit_cem_incumbent(resume.parent, run_dir)
        else:
            mean = load_policy(resume).weights.detach().cpu().numpy().astype(np.float64)
            mean /= np.linalg.norm(mean)
    existing_best = incumbent_validation(run_dir, "best.pt", "best_validation.json",
                                         validation_seeds, budget, reward_config)
    if existing_best is not None:
        best_metrics = existing_best
    if not (run_dir / "initial_validation.json").exists():
        metrics, episodes = run_episodes(LinearCandidatePolicy(initial), validation_seeds, budget,
                                         reward_config=reward_config)
        write_json(run_dir / "initial_validation.json", {"metrics": metrics, "episodes": episodes,
                   "eligible_for_rl_selection": False, "source": "explicit heuristic reference before policy search"})
        print(json.dumps({"event": "initial_validation", **metrics}), flush=True)

    def save_state():
        write_json(run_dir / "cem_state.json", {
            "method": "cem", "generation": generation, "mean": mean.tolist(), "std": std.tolist(),
            "rng_state": rng.bit_generator.state, "best_validation": best_metrics,
            "initial_weights_normalized": initial.tolist(), "run_steps": budget.used,
            "budget": budget.shared.snapshot(), "validation_seeds": validation_seeds,
        })

    try:
        for _ in range(args.generations):
            if budget.exhausted:
                break
            # Common random numbers improve the comparison without exposing seeds to policies.
            train_seeds = rng.integers(0, 1_000_000, size=args.train_episodes).tolist()
            noises = rng.normal(size=(math.ceil(args.population / 2), len(mean)))
            noises = np.concatenate((noises, -noises), axis=0)[:args.population]
            population = mean[None, :] + std[None, :] * noises
            population /= np.linalg.norm(population, axis=1, keepdims=True).clip(min=1e-8)
            # A deterministic reference candidate helps measure improvements from this generation.
            candidates = [mean.copy(), *population]
            returns = []
            candidate_metrics = []
            for index, weights in enumerate(candidates):
                metrics, episodes = run_episodes(LinearCandidatePolicy(weights), train_seeds, budget,
                                                 target_lines=args.target_lines,
                                                 reward_config=reward_config)
                returns.append(metrics["mean_return"])
                candidate_metrics.append(metrics)
                append_log(run_dir / "training.jsonl", {
                    "event": "cem_candidate", "generation": generation + 1, "candidate": index,
                    "train_seeds": train_seeds, "metrics": metrics, "episodes": episodes,
                    "weights": weights.tolist(), "run_steps": budget.used,
                    "elapsed_seconds": budget.shared.elapsed_seconds})
            elite_count = max(2, math.ceil(len(candidates) * args.elite_fraction))
            order = np.argsort(returns)[::-1]
            elite = np.asarray(candidates)[order[:elite_count]]
            updated_mean = (1 - args.update_rate) * mean + args.update_rate * elite.mean(axis=0)
            updated_mean /= max(np.linalg.norm(updated_mean), 1e-8)
            updated_std = (1 - args.update_rate) * std + args.update_rate * elite.std(axis=0)
            mean = updated_mean
            std = np.maximum(updated_std, args.min_std)
            generation += 1
            # The mean is a reward-driven update. The elite is reward-selected, not heuristic fallback.
            contenders = [("mean", mean)]
            champion = np.asarray(candidates[order[0]])
            if not np.allclose(champion, initial, atol=1e-8):
                contenders.append(("reward_selected_elite", champion))
            for candidate_name, weights in contenders:
                metrics, episodes = run_episodes(LinearCandidatePolicy(weights), validation_seeds,
                                                 budget, reward_config=reward_config)
                drift = float(np.linalg.norm(weights - initial))
                metadata = {
                    "stage": "rl_updated", "rl_updated": True, "method": "cross_entropy_policy_search",
                    "generation": generation, "candidate": candidate_name,
                    "objective": "mean episodic environment reward", "validation": metrics,
                    "training_return_before": returns[0], "training_return_elite": float(max(returns)),
                    "parameter_l2_drift_from_normalized_initial": drift,
                    "source_initialization": "explicit Dellacherie heuristic, no behavior cloning",
                    "behavior_cloning": False, "run_steps": budget.used,
                    "elapsed_seconds": budget.shared.elapsed_seconds,
                }
                if drift <= 1e-8:
                    metadata["rl_updated"] = False
                checkpoint = run_dir / f"cem_{args.invocation_id}_gen_{generation:03d}_{candidate_name}.pt"
                LinearCandidatePolicy(weights).save(checkpoint, metadata)
                improved = metadata["rl_updated"] and (
                    best_metrics is None or selection_key(metrics) > selection_key(best_metrics))
                if improved:
                    best_metrics = metrics
                    LinearCandidatePolicy(weights).save(run_dir / "best.pt", metadata)
                    write_json(run_dir / "best_validation.json", {
                        "checkpoint": str(checkpoint.resolve()), "metrics": metrics,
                        "episodes": episodes, "metadata": metadata,
                        "selection": "success rate, then fewer successful placements; failure progress tie-break"})
                append_log(run_dir / "training.jsonl", {"event": "cem_validation", **metadata,
                           "checkpoint": str(checkpoint), "selected_best": bool(improved)})
                print(json.dumps({"event": "cem_validation", "generation": generation,
                                  "candidate": candidate_name, "selected_best": bool(improved),
                                  "drift": drift, **metrics, "steps": budget.used}), flush=True)
            save_state()
    finally:
        save_state()
        LinearCandidatePolicy(mean).save(run_dir / "latest.pt", {
            "stage": "rl_updated" if generation else "heuristic_initialization",
            "rl_updated": generation > 0, "method": "cross_entropy_policy_search",
            "generation": generation,
            "parameter_l2_drift_from_normalized_initial": float(np.linalg.norm(mean - initial))})


def ppo_train(args, run_dir, budget, validation_seeds, reward_config):
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback

    if CandidateMaskablePolicy is None:
        raise ImportError("MaskablePPO dependencies are not installed")
    rng = np.random.default_rng(args.seed)
    create_reference_checkpoints(run_dir, rng)
    ppo_state_path = run_dir / "ppo_state.json"
    previous_state = None
    if args.resume and not ppo_state_path.exists():
        source_state = Path(args.resume).resolve().parent / "ppo_state.json"
        if source_state.exists():
            previous_state = json.loads(source_state.read_text(encoding="utf-8"))
    if args.resume and ppo_state_path.exists():
        previous_state = json.loads(ppo_state_path.read_text(encoding="utf-8"))
    if previous_state is not None:
        rng.bit_generator.state = previous_state["rng_state"]

    class BudgetedEnv(gym.Wrapper):
        def action_masks(self):
            return self.env.action_masks()

        def reset(self, *, seed=None, options=None):
            # Every reset has an explicit training-range seed; no final-test seed can leak in.
            return self.env.reset(seed=int(rng.integers(0, 1_000_000)), options=options)

        def step(self, action):
            budget.reserve()
            return self.env.step(action)

    class ProgressCallback(BaseCallback):
        def _on_step(self):
            for info in self.locals.get("infos", []):
                if info.get("episode"):
                    append_log(run_dir / "training.jsonl", {"event": "ppo_episode",
                               "timesteps": self.num_timesteps, "episode": info["episode"]})
            return not budget.exhausted

    env = BudgetedEnv(TetrisEnv(target_lines=args.target_lines, max_pieces=500,
                               reward_config=reward_config))
    initial = DELLACHERIE_WEIGHTS / np.linalg.norm(DELLACHERIE_WEIGHTS)
    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device="cpu")
    else:
        model = MaskablePPO(CandidateMaskablePolicy, env, policy_kwargs={"initial_weights": initial.tolist()},
            seed=args.seed, device="cpu", n_steps=args.n_steps, batch_size=min(64, args.n_steps),
            learning_rate=args.learning_rate, gamma=1.0, gae_lambda=0.95, n_epochs=4,
            ent_coef=0.001, verbose=1)
        model.save(run_dir / "initialized_ppo.zip")
    initial_actor = model.policy.actor.weight.detach().clone()
    best_metrics = incumbent_validation(run_dir, "best_ppo.zip", "best_ppo_validation.json",
                                         validation_seeds, budget, reward_config)
    # Model construction/load seeds Torch; restore only after all model loading.
    torch_rng_restored = restore_torch_rng(previous_state)
    last_path = run_dir / "latest_ppo.zip"
    try:
        for update in range(args.generations):
            if budget.exhausted:
                break
            model.learn(total_timesteps=args.ppo_chunk_steps, reset_num_timesteps=False,
                        callback=ProgressCallback(), progress_bar=False)
            model.save(last_path)
            drift = float(torch.linalg.vector_norm(model.policy.actor.weight - initial_actor).item())
            metrics, episodes = run_episodes(PPOPolicyAdapter(model), validation_seeds, budget,
                                             reward_config=reward_config)
            eligible = drift > 1e-8 and model._n_updates > 0
            improved = eligible and (best_metrics is None or selection_key(metrics) > selection_key(best_metrics))
            metadata = {"method": "maskable_ppo", "update": update + 1,
                        "rl_updated": eligible, "parameter_l2_drift_from_run_start": drift,
                        "optimizer_updates": model._n_updates, "validation": metrics,
                        "run_steps": budget.used, "elapsed_seconds": budget.shared.elapsed_seconds}
            if improved:
                best_metrics = metrics
                model.save(run_dir / "best_ppo.zip")
                write_json(run_dir / "best_ppo_validation.json", {"metrics": metrics,
                           "episodes": episodes, "metadata": metadata})
            append_log(run_dir / "training.jsonl", {"event": "ppo_validation", **metadata,
                       "selected_best": bool(improved)})
            print(json.dumps({"event": "ppo_validation", **metadata}), flush=True)
    finally:
        model.save(last_path)
        write_json(ppo_state_path, {"rng_state": rng.bit_generator.state,
                   "torch_rng_state": torch.get_rng_state().tolist(),
                   "torch_rng_restored_on_resume": torch_rng_restored,
                   "best_validation": best_metrics, "validation_seeds": validation_seeds,
                   "optimizer_updates": model._n_updates, "num_timesteps": model.num_timesteps,
                   "resume_semantics": "optimizer/policy, NumPy seed generator and CPU Torch RNG retained; "
                                       "resume begins a fresh episode (not a bitwise continuation of a partial episode)"})
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["cem", "ppo"], default="cem")
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--budget-dir", type=Path, default=Path("artifacts/run"),
                        help="Shared ledger for ALL experiments, independent of checkpoint output directory")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Maximum training plus validation transitions in this invocation")
    parser.add_argument("--max-seconds", type=float, default=18_000)
    parser.add_argument("--max-transitions", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--validation-seeds", type=Path)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--train-episodes", type=int, default=6)
    parser.add_argument("--target-lines", type=int, default=40)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--update-rate", type=float, default=0.7)
    parser.add_argument("--min-std", type=float, default=0.002)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--ppo-chunk-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--reward-config", type=Path)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed >= 1_000_000:
        parser.error("--seed must be in training range [0, 1000000)")
    if args.population < 2 or args.train_episodes < 1 or args.validation_episodes < 1 or args.steps < 1:
        parser.error("positive episode/step counts and population >= 2 are required")
    if args.generations < 1 or args.target_lines < 1 or args.n_steps < 2 or args.ppo_chunk_steps < args.n_steps:
        parser.error("generations/target-lines must be positive; n-steps >= 2; ppo chunk >= n-steps")
    if args.sigma <= 0 or args.min_std <= 0 or args.learning_rate <= 0:
        parser.error("sigma, min-std, and learning-rate must be positive")
    if not 0 < args.max_seconds <= 18000 or not 0 < args.max_transitions <= 5000000:
        parser.error("Shared budget must be within the authorized 18000 seconds / 5000000 transitions")
    if not 0 < args.elite_fraction <= 1 or not 0 < args.update_rate <= 1:
        parser.error("elite-fraction and update-rate must be in (0,1]")
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    shared = Budget(args.budget_dir.resolve(), max_seconds=args.max_seconds, max_transitions=args.max_transitions)
    budget = RunBudget(shared, args.steps)
    validation_seeds = load_validation_seeds(args)
    reward_config = json.loads(args.reward_config.read_text(encoding="utf-8-sig")) if args.reward_config else None
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["validation_seed_values"] = validation_seeds
    config["torch_version"] = str(torch.__version__)
    config["budget_at_start"] = shared.snapshot()
    args.invocation_id = str(time.time_ns())
    write_json(run_dir / f"config_{args.method}_{int(time.time())}.json", config)
    outcome = "finished_requested_generations"
    try:
        if args.method == "cem":
            cem_train(args, run_dir, budget, validation_seeds, reward_config)
        else:
            ppo_train(args, run_dir, budget, validation_seeds, reward_config)
    except BudgetExhausted as error:
        outcome = "budget_exhausted"
        print(str(error), flush=True)
    finally:
        shared.save()
        summary = {"method": args.method, "outcome": outcome, "invocation_transitions": budget.used,
                   "budget": shared.snapshot()}
        write_json(run_dir / f"last_{args.method}_run.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
