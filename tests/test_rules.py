"""Independent fixtures for game rules, reachable inputs, and evaluation math."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tetris_rl.engine import (
    HEIGHT, HIDDEN_ROWS, MAX_ACTIONS, PIECE_NAMES, SHAPES, SPAWN_X, SPAWN_Y,
    STATES, WIDTH, board_features, decode_action, encode_action, fits,
    get_reachable, lock_piece, rotated,
)
from tetris_rl.env import TetrisEnv
from tetris_rl.evaluate import RandomPolicy, predict_action, summarize
from tetris_rl.replay import build_html


def board():
    return np.zeros((24, 10), dtype=np.uint8)


def test_piece_geometry_four_cells_and_rotation_cycle():
    assert SHAPES.shape == (7, 4, 4, 2)
    for shape in SHAPES.reshape(-1, 4, 2):
        assert len({tuple(cell) for cell in shape}) == 4
    np.testing.assert_array_equal(SHAPES[0, 0], [[0, 1], [1, 1], [2, 1], [3, 1]])
    assert {tuple(v) for v in SHAPES[5, 1]} == {(1, 0), (1, 1), (2, 1), (1, 2)}
    for r in range(1, 4):
        np.testing.assert_array_equal(SHAPES[3, r], SHAPES[3, 0])


def test_collision_walls_floor_and_occupied_cells():
    b = board()
    assert fits(b, 0, 0, 0, 22)
    assert not fits(b, 0, 0, -1, 22)
    assert not fits(b, 0, 0, 7, 22)
    assert not fits(b, 0, 0, 0, 23)
    assert not fits(b, 0, 0, 0, -2)
    b[23, 1] = 7
    assert not fits(b, 0, 0, 0, 22)
    assert fits(b, 0, 0, 2, 22)


@pytest.mark.parametrize("piece,x,expected_x", [(0, -2, 0), (5, -1, 0), (1, -1, 0)])
def test_srs_left_wall_kicks(piece, x, expected_x):
    b = board()
    assert fits(b, piece, 1, x, 8)
    assert not fits(b, piece, 0, x, 8)
    ok, rotation, nx, ny = rotated(b, piece, 1, x, 8, -1)
    assert (ok, rotation, nx, ny) == (True, 0, expected_x, 8)


def test_srs_floor_kicks_use_vertical_offsets():
    b = board()
    # T spawn orientation rests on the floor. 0->R third kick is (-1,+1 up).
    assert fits(b, 5, 0, 3, 22)
    assert rotated(b, 5, 0, 3, 22, 1) == (True, 1, 2, 21)
    # I 0->R fifth kick is (+1,+2 up), unlike JLSTZ.
    assert fits(b, 0, 0, 3, 22)
    assert rotated(b, 0, 0, 3, 22, 1) == (True, 1, 4, 20)


def test_line_clear_compaction_and_eroded_piece_cells():
    b = board()
    b[23, :] = 2
    b[23, 3:7] = 0
    b[22, 9] = 3
    after, cleared, eroded, topout = lock_piece(b, 0, 0, 3, 22)
    expected = board()
    expected[23, 9] = 3
    np.testing.assert_array_equal(after, expected)
    assert (cleared, eroded, topout) == (1, 4, False)
    # The input board is not mutated during candidate construction.
    assert b[23, 3] == 0 and b[22, 9] == 3


def test_four_line_clear_and_post_clear_topout():
    b = board()
    b[20:24, :] = 1
    b[20:24, 5] = 0
    after, cleared, eroded, topout = lock_piece(b, 0, 1, 3, 20)
    assert not after.any()
    assert (cleared, eroded, topout) == (4, 16, False)
    b = board()
    _, _, _, topout = lock_piece(b, 3, 0, 3, 2)
    assert topout
    _, _, _, topout = lock_piece(b, 3, 0, 3, 4)
    assert not topout


def test_7bag_is_permutation_for_many_bags():
    env = TetrisEnv()
    env.reset(seed=739)
    # Public current+next are the first six draws. Complete that first bag,
    # then verify many complete subsequent bags from the same generator.
    draws = [env.current, *env.next_queue] + [env._draw() for _ in range(64)]
    for start in range(0, 70, 7):
        assert sorted(draws[start:start + 7]) == list(range(7))


def valid_with_hold(env, hold):
    valid = np.flatnonzero(env.action_masks())
    return int(next(a for a in valid if decode_action(int(a))[0] == hold))


def test_empty_hold_consumes_next_and_filled_hold_swaps_once():
    env = TetrisEnv()
    env.reset(seed=65)
    before_current, before_next = env.current, env.next_queue
    action = valid_with_hold(env, 1)
    assert env.candidate(action)["piece"] == before_next[0]
    assert env.action_path(action).count("hold") == 1
    env.step(action)
    assert env.hold == before_current
    assert env.current == before_next[1]
    assert env.next_queue[:3] == before_next[2:]
    outgoing, stored, before_next = env.current, env.hold, env.next_queue
    action = valid_with_hold(env, 1)
    assert env.candidate(action)["piece"] == stored
    env.step(action)
    assert env.hold == outgoing
    assert env.current == before_next[0]
    assert env.next_queue[:4] == before_next[1:]
    assert env.hold_available
    for a in np.flatnonzero(env.action_masks()):
        assert env.action_path(int(a)).count("hold") <= 1


def test_without_hold_only_one_next_consumed():
    env = TetrisEnv()
    env.reset(seed=91)
    next_before = env.next_queue
    action = valid_with_hold(env, 0)
    assert "hold" not in env.action_path(action)
    env.step(action)
    assert env.hold == -1 and env.current == next_before[0]
    assert env.next_queue[:4] == next_before[1:]


def test_piece_limit_is_truncation_and_terminal_mask_empty():
    env = TetrisEnv(max_pieces=1)
    env.reset(seed=5)
    _, _, terminated, truncated, info = env.step(valid_with_hold(env, 0))
    assert not terminated and truncated and not info["success"]
    assert info["reason"] == "piece_limit" and info["pieces"] == 1
    assert not env.action_masks().any()
    assert not env.hold_available
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_success_at_least_target_lines_precedes_piece_limit():
    env = TetrisEnv(target_lines=40, max_pieces=1)
    env.reset(seed=8)
    env.current = 0
    env.lines = 39
    env.board[20:24] = 1
    env.board[20:24, 5] = 0
    env._refresh_candidates()
    action = encode_action(0, 1, 3, 20)
    assert env.action_masks()[action]
    _, _, terminated, truncated, info = env.step(action)
    assert terminated and not truncated and info["success"]
    assert info["lines"] == 43 and info["reason"] == "success"


def test_lockout_is_termination_and_spawn_collision_has_no_reachable_path():
    env = TetrisEnv()
    env.reset(seed=32)
    env.current = 3
    env.board[4, 4:6] = 1
    env._refresh_candidates()
    action = encode_action(0, 0, 3, 2)
    assert env.action_masks()[action]
    _, _, terminated, truncated, info = env.step(action)
    assert terminated and not truncated and not info["success"]
    assert info["reason"] == "lock_out"
    b = board()
    b[SPAWN_Y, SPAWN_X + 1] = 1
    assert not fits(b, 5, 0, SPAWN_X, SPAWN_Y)
    assert not get_reachable(b, 5)[0].any()


def test_geometrically_free_but_inaccessible_cavity_is_masked():
    env = TetrisEnv()
    env.reset(seed=0)
    env.current = 3
    env.board[12, :] = 1
    env._refresh_candidates()
    assert fits(env.board, 3, 0, 3, 22)
    action = encode_action(0, 0, 3, 22)
    assert not env.action_masks()[action]
    with pytest.raises(ValueError, match="valid"):
        env.candidate(action)


def test_seed_reproducibility_including_actions_and_rewards():
    env_a, env_b = TetrisEnv(), TetrisEnv()
    obs_a, _ = env_a.reset(seed=202)
    obs_b, _ = env_b.reset(seed=202)
    for _ in range(15):
        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key], obs_b[key])
        action = int(np.flatnonzero(obs_a["mask"])[0])
        obs_a, reward_a, term_a, trunc_a, info_a = env_a.step(action)
        obs_b, reward_b, term_b, trunc_b, info_b = env_b.step(action)
        assert (reward_a, term_a, trunc_a, info_a) == (reward_b, term_b, trunc_b, info_b)
        if term_a or trunc_a:
            break


def test_public_observation_does_not_depend_on_private_future_or_rng():
    env = TetrisEnv()
    env.reset(seed=123)
    baseline = env._observation()
    env._bag = list(reversed(env._bag))
    env.np_random.bit_generator.state = np.random.default_rng(999).bit_generator.state
    # Appended private queue entries beyond the five visible entries must not
    # affect the current action features, mask, or public observation.
    env._queue.extend([6, 6, 6, 6])
    env._refresh_candidates()
    after = env._observation()
    assert set(after) == {"board", "context", "candidates", "mask"}
    for key in after:
        np.testing.assert_array_equal(after[key], baseline[key])
    assert after["context"].shape == (12,)
    assert len(env.public_state()["next"]) == 5
    assert not any("seed" in key or "rng" in key or "bag" in key for key in env.public_state())


def test_observation_is_copy_and_gym_space_accepts_it():
    env = TetrisEnv()
    obs, _ = env.reset(seed=13)
    assert env.observation_space.contains(obs)
    obs["board"].fill(7)
    obs["mask"].fill(0)
    assert not env.board.any()
    assert env.action_masks().any()


# Independent SRS reference in screen coordinates, deliberately not imported
# from engine.KICKS or engine.replay_path.
REF_J = {
    (0, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (1, 0): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (1, 2): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (2, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (2, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (3, 2): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (3, 0): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (0, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
}
REF_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
}


def independent_input_board(initial, piece, path):
    starts = [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ]
    orientations = [starts[piece]]
    for _ in range(3):
        orientations.append(orientations[-1] if piece == 3 else
                            [((3 if piece == 0 else 2) - y, x) for x, y in orientations[-1]])

    def free(r, x, y):
        return all(0 <= x + dx < 10 and 0 <= y + dy < 24 and
                   initial[y + dy, x + dx] == 0 for dx, dy in orientations[r])

    r, x, y = 0, 3, 2
    assert free(r, x, y)
    assert path[-1] == "hard_drop" and path.count("hard_drop") == 1
    for token in path:
        if token == "hard_drop":
            while free(r, x, y + 1):
                y += 1
        elif token in ("left", "right", "down"):
            x += -1 if token == "left" else 1 if token == "right" else 0
            y += int(token == "down")
            assert free(r, x, y)
        else:
            assert token in ("cw", "ccw")
            nr = (r + (1 if token == "cw" else -1)) % 4
            options = [(0, 0)] if piece == 3 else (REF_I if piece == 0 else REF_J)[r, nr]
            valid = next(((x + dx, y + dy) for dx, dy in options if free(nr, x + dx, y + dy)), None)
            assert valid is not None
            r, (x, y) = nr, valid
    result = initial.copy()
    for dx, dy in orientations[r]:
        result[y + dy, x + dx] = piece + 1
    kept = result[np.any(result == 0, axis=1)]
    result = np.vstack([np.zeros((24 - len(kept), 10), dtype=np.uint8), kept])
    return result, (r, x, y)


@pytest.mark.parametrize("seed", [22, 94, 781])
def test_every_candidate_in_sampled_states_has_independently_executable_path(seed):
    env = TetrisEnv()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        valid = np.flatnonzero(obs["mask"])
        for action in valid:
            action = int(action)
            candidate = env.candidate(action)
            path = env.action_path(action)
            assert path.count("hold") <= 1
            expected_piece = env.current
            if path[0] == "hold":
                expected_piece = env.hold if env.hold >= 0 else env.next_queue[0]
                path = path[1:]
            assert candidate["piece"] == expected_piece
            result, pose = independent_input_board(env.board, expected_piece, path)
            assert pose == (candidate["rotation"], candidate["x"], candidate["y"])
            np.testing.assert_array_equal(result, candidate["board_after"])
            np.testing.assert_array_equal(result, env.replay_action(action))
            assert len(env.action_path(action)) == candidate["input_count"]
        obs, _, term, trunc, _ = env.step(int(rng.choice(valid)))
        if term or trunc:
            break


def test_input_time_separate_from_placement_count_and_reward_potential_telescopes():
    env = TetrisEnv(max_pieces=6)
    obs, _ = env.reset(seed=76)
    ticks, total_reward = 0, 0.0
    while True:
        action = int(np.flatnonzero(obs["mask"])[-1])
        ticks += len(env.action_path(action))
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            break
    assert info["input_ticks"] == ticks
    assert info["game_time_seconds"] == ticks / 60
    base = env.reward_config["line"] * min(info["lines"], env.target_lines)
    base += env.reward_config["placement"] * info["pieces"]
    if info["success"]:
        base += env.reward_config["success"]
    elif term:
        base += env.reward_config["topout"]
    assert total_reward == pytest.approx(base)


def test_invalid_policy_is_rejected_and_random_policy_only_samples_mask():
    obs = {"mask": np.array([0, 1, 0, 1])}
    policy = RandomPolicy(41)
    assert {policy.predict(obs) for _ in range(30)} == {1, 3}
    with pytest.raises(ValueError, match="invalid action"):
        predict_action(lambda obs: 0, obs)


def test_evaluation_summary_separates_successes_failures_and_timing():
    episodes = []
    for success, pieces, lines, reason in [(True, 100, 40, "success"),
                                          (True, 120, 40, "success"),
                                          (False, 500, 39, "piece_limit")]:
        episodes.append(dict(success=success, pieces=pieces, lines=lines, reason=reason,
                             game_time_seconds=pieces / 10, episode_return=lines,
                             input_ticks=pieces * 6, inference_seconds=.1, wall_seconds=.5))
    result = summarize(episodes)
    assert result["success_rate"] == 2 / 3
    assert result["episodes"] == 3 and result["successes"] == 2
    assert result["successful_pieces_median"] == 110
    assert result["successful_pieces_p90"] == 118
    assert result["failure_lines_median"] == 39
    assert result["failure_reasons"] == {"piece_limit": 1}
    assert result["total_episode_wall_seconds"] == 1.5


def test_replay_html_is_offline_and_escapes_script_closing_text(tmp_path):
    payload = {"frames": [{}], "actions": [], "agent": "</script><script>alert(1)</script>"}
    source = tmp_path / "replay.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = build_html(source)
    html = output.read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script>" in html
    assert '<canvas id="board"' in html
    assert '<input id="slider"' in html
    assert 'src="http' not in html
