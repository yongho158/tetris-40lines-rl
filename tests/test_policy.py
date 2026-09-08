import numpy as np
import torch

from tetris_rl.env import TetrisEnv
from tetris_rl.engine import MAX_ACTIONS, N_FEATURES
from tetris_rl.policy import (CandidateMaskablePolicy, DELLACHERIE_WEIGHTS,
                              LinearCandidatePolicy, load_policy)


def public_observation():
    obs = {
        "board": np.zeros((24, 10), dtype=np.uint8),
        "context": np.zeros(12, dtype=np.float32),
        "candidates": np.zeros((MAX_ACTIONS, N_FEATURES), dtype=np.float32),
        "mask": np.zeros(MAX_ACTIONS, dtype=np.int8),
    }
    obs["mask"][[2, 7]] = 1
    obs["candidates"][2, 1] = 1
    obs["candidates"][7, 1] = 2
    obs["candidates"][3, 1] = 1_000_000  # Best score is invalid and must never win.
    return obs


def test_linear_save_load_and_mask(tmp_path):
    obs = public_observation()
    policy = LinearCandidatePolicy(DELLACHERIE_WEIGHTS)
    assert policy.predict(obs) == 7
    path = tmp_path / "policy.pt"
    policy.save(path, {"rl_updated": False})
    restored = load_policy(path)
    assert torch.equal(policy.weights, restored.weights)
    assert restored.predict(obs) == 7


def test_ppo_serialization_and_mask(tmp_path):
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    env = TetrisEnv()
    model = MaskablePPO(CandidateMaskablePolicy, env, n_steps=8, batch_size=8,
                       policy_kwargs={"initial_weights": DELLACHERIE_WEIGHTS.tolist()},
                       device="cpu", seed=19)
    obs = public_observation()
    before, _ = model.predict(obs, action_masks=obs["mask"], deterministic=True)
    assert int(before) == 7
    path = tmp_path / "ppo.zip"
    model.save(path)
    restored = load_policy(path)
    assert restored.predict(obs) == 7
    env.close()
