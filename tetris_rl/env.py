"""Gymnasium macro-placement environment with public-state action features."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .engine import (
    FEATURE_NAMES, HEIGHT, HIDDEN_ROWS, MAX_ACTIONS, N_FEATURES, PIECE_NAMES,
    SPAWN_X, SPAWN_Y, STATES, WIDTH, board_features, candidate_features,
    decode_action, fits, get_reachable, lock_piece, path_from_reachability,
    replay_path,
)

DEFAULT_REWARD = {
    "line": 1.0, "success": 100.0, "topout": -100.0,
    "placement": -0.05, "invalid": -1.0,
    "potential_scale": 0.1, "height_weight": 0.02,
    "holes_weight": 0.5, "potential_bound": 10.0,
}


class TetrisEnv(gym.Env):
    """40 Lines with one optional hold and one reachable lock per action.

    Context order: current ID, hold ID (-1 empty), next five IDs,
    hold_available, lines, target_lines, pieces, max_pieces. The agent never
    observes the bag, RNG state, seed, or future pieces beyond next five.
    Action IDs encode (hold, rotation, y anchor, x anchor), see engine.py.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 60}

    def __init__(self, target_lines=40, max_pieces=500, render_mode=None,
                 reward_config=None):
        super().__init__()
        if target_lines < 1 or max_pieces < 1:
            raise ValueError("target_lines and max_pieces must be positive")
        if render_mode not in (None, "ansi", "rgb_array"):
            raise ValueError("render_mode must be ansi, rgb_array, or None")
        self.target_lines = int(target_lines)
        self.max_pieces = int(max_pieces)
        self.render_mode = render_mode
        self.reward_config = dict(DEFAULT_REWARD)
        if reward_config:
            unknown = set(reward_config) - set(DEFAULT_REWARD)
            if unknown:
                raise ValueError(f"Unknown reward keys: {sorted(unknown)}")
            self.reward_config.update(reward_config)
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 7, (HEIGHT, WIDTH), np.uint8),
            "context": spaces.Box(-1, np.inf, (12,), np.float32),
            "candidates": spaces.Box(0, np.inf, (MAX_ACTIONS, N_FEATURES), np.float32),
            "mask": spaces.MultiBinary(MAX_ACTIONS),
        })
        self.board = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.current, self.hold = 0, -1
        self._queue, self._bag = [], []
        self.lines = self.pieces = self.input_ticks = self.invalid_actions = 0
        self.episode_return = 0.0
        self._done = False
        self._mask = np.zeros(MAX_ACTIONS, dtype=np.bool_)
        self._features = np.zeros((MAX_ACTIONS, N_FEATURES), dtype=np.float32)
        self._reach = [None, None]
        self._pieces_for_hold = [0, 0]
        self._reason = "playing"

    @property
    def next_queue(self):
        return tuple(self._queue[:5])

    @property
    def hold_available(self):
        # Every observation is a placement boundary; a hold is optional once.
        return not self._done

    def _draw(self):
        if not self._bag:
            self._bag = list(map(int, self.np_random.permutation(7)))
        return self._bag.pop()

    def _take_next(self):
        result = self._queue.pop(0)
        self._queue.append(self._draw())
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self._bag = []
        self.current = self._draw()
        self._queue = [self._draw() for _ in range(5)]
        self.hold = -1
        self.lines = self.pieces = self.input_ticks = self.invalid_actions = 0
        self.episode_return = 0.0
        self._done = False
        self._reason = "playing"
        self._refresh_candidates()
        return self._observation(), self._info()

    def _refresh_candidates(self):
        self._mask.fill(False)
        self._features.fill(0)
        if self._done:
            self._reach = [None, None]
            return
        self._pieces_for_hold = [self.current, self.hold if self.hold >= 0 else self._queue[0]]
        for hold_used, piece in enumerate(self._pieces_for_hold):
            reach = get_reachable(self.board, piece)
            self._reach[hold_used] = reach
            start = hold_used * STATES
            self._mask[start:start + STATES] = reach[0]
            self._features[start:start + STATES] = candidate_features(
                self.board, piece, reach[0], reach[4], hold_used)

    def action_masks(self):
        return self._mask.copy()

    def _observation(self):
        context = np.array([self.current, self.hold, *self._queue[:5],
                            float(self.hold_available), self.lines,
                            self.target_lines, self.pieces, self.max_pieces],
                           dtype=np.float32)
        return {"board": self.board.copy(), "context": context,
                "candidates": self._features.copy(),
                "mask": self._mask.astype(np.int8)}

    def _potential(self):
        f = board_features(self.board)
        cfg = self.reward_config
        cost = cfg["height_weight"] * f[4] + cfg["holes_weight"] * f[2]
        return -cfg["potential_scale"] * min(cfg["potential_bound"], cost)

    def _info(self, **extra):
        return {"success": self.lines >= self.target_lines,
                "reason": self._reason, "lines": self.lines,
                "pieces": self.pieces, "input_ticks": self.input_ticks,
                "game_time_seconds": self.input_ticks / 60.0,
                "episode_return": self.episode_return,
                "invalid_actions": self.invalid_actions, **extra}

    def candidate(self, action):
        action = int(action)
        if not self.action_space.contains(action) or not self._mask[action]:
            raise ValueError("Action is not currently valid")
        hold_used, r, x, y = decode_action(action)
        piece = self._pieces_for_hold[hold_used]
        after, cleared, eroded, topout = lock_piece(self.board, piece, r, x, y)
        return {"piece": piece, "piece_name": PIECE_NAMES[piece],
                "rotation": r, "x": x, "y": y, "hold": bool(hold_used),
                "path": self.action_path(action),
                "input_count": int(self._features[action, 13]),
                "features": self._features[action].copy(),
                "board_after": after, "cleared": int(cleared),
                "eroded": int(eroded), "topout": bool(topout)}

    def action_path(self, action):
        action = int(action)
        if not self.action_space.contains(action) or not self._mask[action]:
            raise ValueError("Action is not currently valid")
        hold_used, _, _, _ = decode_action(action)
        path = path_from_reachability(self._reach[hold_used], action % STATES)
        return (["hold"] if hold_used else []) + path

    def replay_action(self, action):
        path = self.action_path(action)
        piece = self.current
        if path[0] == "hold":
            piece = self.hold if self.hold >= 0 else self._queue[0]
            path = path[1:]
        r, x, y = replay_path(self.board, piece, path)
        _, expected_r, expected_x, expected_y = decode_action(action)
        if (r, x, y) != (expected_r, expected_x, expected_y):
            raise AssertionError("Replayed path did not reach encoded placement")
        return lock_piece(self.board, piece, r, x, y)[0]

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() after an episode ends")
        action = int(action)
        if not self.action_space.contains(action) or not self._mask[action]:
            self.invalid_actions += 1
            self.input_ticks += 1
            reward = float(self.reward_config["invalid"])
            self.episode_return += reward
            truncated = self.invalid_actions >= self.max_pieces
            if truncated:
                self._done = True
                self._reason = "invalid_action_limit"
                self._refresh_candidates()
            return self._observation(), reward, False, truncated, self._info(invalid_action=True)
        previous_potential = self._potential()
        hold_used, r, x, y = decode_action(action)
        input_count = int(self._features[action, 13])
        if hold_used:
            outgoing = self.current
            self.current = self.hold if self.hold >= 0 else self._take_next()
            self.hold = outgoing
        self.board, cleared, _, lockout = lock_piece(self.board, self.current, r, x, y)
        rewarded_lines = min(int(cleared), max(0, self.target_lines - self.lines))
        self.lines += int(cleared)
        self.pieces += 1
        self.input_ticks += input_count
        self.current = self._take_next()
        success = self.lines >= self.target_lines
        # Block out occurs on the newly spawned current piece; hold cannot
        # rescue an already failed spawn. Usually partial lock out fires first.
        blocked = not fits(self.board, self.current, 0, SPAWN_X, SPAWN_Y)
        terminated = bool(success or lockout or blocked)
        truncated = bool(not terminated and self.pieces >= self.max_pieces)
        self._done = terminated or truncated
        self._reason = ("success" if success else "lock_out" if lockout else
                        "block_out" if blocked else "piece_limit" if truncated else "playing")
        cfg = self.reward_config
        reward = cfg["line"] * rewarded_lines + cfg["placement"]
        if success:
            reward += cfg["success"]
        elif terminated:
            reward += cfg["topout"]
        # Undiscounted potential difference telescopes; terminal Phi is zero.
        # A completion/failure cannot retain a shaping-only terminal windfall.
        reward += (0.0 if self._done else self._potential()) - previous_potential
        self.episode_return += float(reward)
        self._refresh_candidates()
        return self._observation(), float(reward), terminated, truncated, self._info(cleared=int(cleared))

    def public_state(self):
        return {"board": self.board.tolist(), "current": PIECE_NAMES[self.current],
                "hold": PIECE_NAMES[self.hold] if self.hold >= 0 else None,
                "next": [PIECE_NAMES[p] for p in self._queue[:5]],
                "lines": self.lines, "pieces": self.pieces,
                "input_ticks": self.input_ticks,
                "game_time_seconds": self.input_ticks / 60.0,
                "hold_available": self.hold_available}

    def render(self):
        if self.render_mode == "rgb_array":
            palette = np.array([[18, 23, 38], [38, 204, 218], [70, 103, 223],
                                [245, 164, 45], [244, 209, 53], [104, 196, 73],
                                [174, 92, 215], [224, 84, 89]], dtype=np.uint8)
            image = palette[self.board[HIDDEN_ROWS:]]
            return np.repeat(np.repeat(image, 24, axis=0), 24, axis=1)
        rows = ["|" + "".join(" .IJLOSTZ"[int(v) + 1] if v else " " for v in row) + "|"
                for row in self.board[HIDDEN_ROWS:]]
        return "\n".join(rows + ["+----------+", f"lines={self.lines} pieces={self.pieces}"])
