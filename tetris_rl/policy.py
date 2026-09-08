"""Masked candidate policies. Learned checkpoints never invoke a fallback policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


# Published Dellacherie coefficients, explicitly an initialization/baseline.
DELLACHERIE_WEIGHTS = np.array(
    [-4.500158825, 3.41812681, -3.217888, -9.348695, -7.899265,
     -3.385597, 0., 0., 0., 0., 0., 0., -0.001, 0.], dtype=np.float32
)


def valid_mask(obs: dict, action_masks=None) -> np.ndarray:
    mask = np.asarray(obs["mask"] if action_masks is None else action_masks, dtype=bool)
    if not np.any(mask):
        raise ValueError("No valid placement action is available")
    return mask


def heuristic_action(obs: dict, deterministic: bool = True, action_masks=None) -> int:
    """Fixed, named reference agent; not used by learned-policy inference."""
    scores = np.asarray(obs["candidates"], dtype=np.float32) @ DELLACHERIE_WEIGHTS
    return int(np.argmax(np.where(valid_mask(obs, action_masks), scores, -np.inf)))


class LinearCandidatePolicy(nn.Module):
    """An action is the highest scoring reachable candidate under learned weights."""

    def __init__(self, weights, *, device="cpu", metadata: dict | None = None):
        super().__init__()
        self.weights = nn.Parameter(torch.as_tensor(weights, dtype=torch.float32).clone())
        self.metadata = metadata or {}
        self.to(device)

    @torch.inference_mode()
    def predict(self, obs: dict, deterministic=True, action_masks=None) -> int:
        candidates = torch.as_tensor(obs["candidates"], dtype=torch.float32,
                                     device=self.weights.device)
        mask = torch.as_tensor(valid_mask(obs, action_masks), dtype=torch.bool,
                               device=self.weights.device)
        logits = (candidates @ self.weights).masked_fill(~mask, -torch.inf)
        if deterministic:
            return int(logits.argmax().item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    def save(self, path, metadata: dict | None = None):
        from .engine import FEATURE_NAMES
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1, "kind": "linear_candidate", "inference": "pytorch",
            "weights": self.weights.detach().cpu(), "feature_names": list(FEATURE_NAMES),
            "metadata": {**self.metadata, **(metadata or {})},
        }
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temp)
        temp.replace(path)


class PPOPolicyAdapter:
    def __init__(self, model):
        self.model = model
        self.metadata = {"kind": "maskable_ppo", "inference": "pytorch"}

    def predict(self, obs: dict, deterministic=True, action_masks=None) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic,
                                      action_masks=valid_mask(obs, action_masks))
        return int(np.asarray(action).item())


def load_policy(checkpoint, device="cpu"):
    """Load an explicit policy checkpoint; invalid paths never fall back silently."""
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".zip":
        from sb3_contrib import MaskablePPO
        return PPOPolicyAdapter(MaskablePPO.load(checkpoint, device=device))
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    if data.get("kind") != "linear_candidate":
        raise ValueError(f"Unsupported checkpoint kind: {data.get('kind')}")
    from .engine import FEATURE_NAMES
    if data["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Checkpoint feature order does not match this environment")
    return LinearCandidatePolicy(data["weights"], device=device, metadata=data["metadata"])


def make_maskable_policy_class():
    """Return the SB3 class while allowing linear policy use without importing SB3."""
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from stable_baselines3.common.torch_layers import CombinedExtractor

    class CandidateMaskablePolicy(MaskableActorCriticPolicy):
        """A shared linear actor scores each placement; a small critic reads public state."""

        def __init__(self, *args, initial_weights=None, **kwargs):
            self.initial_weights = initial_weights
            kwargs["features_extractor_class"] = CombinedExtractor
            kwargs["normalize_images"] = False
            super().__init__(*args, **kwargs)

        def _build(self, lr_schedule):
            n_features = self.observation_space["candidates"].shape[-1]
            context_size = int(np.prod(self.observation_space["context"].shape))
            self.actor = nn.Linear(n_features, 1, bias=False)
            self.value_net = nn.Sequential(nn.Linear(10 + 1 + context_size, 64), nn.Tanh(),
                                           nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
            with torch.no_grad():
                if self.initial_weights is not None:
                    self.actor.weight.copy_(torch.tensor(self.initial_weights).reshape(1, -1))
                else:
                    nn.init.normal_(self.actor.weight, std=0.01)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                                  **self.optimizer_kwargs)

        def _distribution(self, obs, action_masks=None):
            logits = self.actor(obs["candidates"].float()).squeeze(-1)
            dist = self.action_dist.proba_distribution(action_logits=logits)
            if action_masks is None:
                action_masks = obs["mask"].bool()
            dist.apply_masking(action_masks)
            return dist

        def predict_values(self, obs):
            board = obs["board"].float()
            # Public board summaries, no RNG or private queue information.
            column_fill = (board > 0).float().mean(dim=1)
            occupancy = (board > 0).float().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
            context = obs["context"].float()
            critic_input = torch.cat((column_fill, occupancy, context), dim=1)
            return self.value_net(critic_input)

        def forward(self, obs, deterministic=False, action_masks=None):
            dist = self._distribution(obs, action_masks)
            actions = dist.get_actions(deterministic=deterministic)
            return actions, self.predict_values(obs), dist.log_prob(actions)

        def evaluate_actions(self, obs, actions, action_masks=None):
            dist = self._distribution(obs, action_masks)
            return self.predict_values(obs), dist.log_prob(actions), dist.entropy()

        def get_distribution(self, obs, action_masks=None):
            return self._distribution(obs, action_masks)

        def _get_constructor_parameters(self) -> dict[str, Any]:
            data = super()._get_constructor_parameters()
            data["initial_weights"] = self.initial_weights
            return data

    return CandidateMaskablePolicy


# A module-level class is needed for reliable model serialization/imports.
try:
    CandidateMaskablePolicy = make_maskable_policy_class()
except ImportError:
    CandidateMaskablePolicy = None
