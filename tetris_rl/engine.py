"""Deterministic SRS mechanics and Numba-compiled placement enumeration.

Coordinates increase right/down. Four hidden rows precede 20 visible rows.
Spawn origins are (3, 2); occupied cells must always remain inside the 10x24
matrix. Partial lock out occurs if any hidden cell remains after line clears.
Movement is untimed, with no automatic gravity/lock delay. A reachable action
is a sequence of left/right/down/CW/CCW followed by a hard drop. Each input is
one synthetic 1/60-second tick; these ticks are not commercial sprint times.
"""

from __future__ import annotations

import numpy as np
from numba import njit

WIDTH, HEIGHT, HIDDEN_ROWS = 10, 24, 4
SPAWN_X, SPAWN_Y = 3, 2
PIECE_NAMES = ("I", "J", "L", "O", "S", "T", "Z")
X_MIN, Y_MIN, X_COUNT, Y_COUNT = -2, -2, 12, 26
STATES = 4 * X_COUNT * Y_COUNT
MAX_ACTIONS = 2 * STATES
FEATURE_NAMES = (
    "landing_height", "eroded_piece_cells", "row_transitions",
    "column_transitions", "holes", "wells", "aggregate_height",
    "bumpiness", "cleared_lines", "max_height", "hole_depth",
    "rows_with_holes", "hold_used", "input_count",
)
N_FEATURES = len(FEATURE_NAMES)
MOVE_NAMES = ("left", "right", "down", "cw", "ccw")


def _make_shapes():
    starts = [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ]
    result = np.zeros((7, 4, 4, 2), dtype=np.int16)
    for p, shape in enumerate(starts):
        for r in range(4):
            result[p, r] = shape
            if p != 3:
                pivot2 = 3 if p == 0 else 2
                shape = [(pivot2 - y, x) for x, y in shape]
    return result


SHAPES = _make_shapes()
# SRS source tables use positive-up y; convert once to screen coordinates.
_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}
_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}
KICKS = np.zeros((2, 4, 2, 5, 2), dtype=np.int16)
for _group, _table in enumerate((_JLSTZ, _I)):
    for (_r, _nr), _offsets in _table.items():
        _direction = 0 if _nr == (_r + 1) % 4 else 1
        for _k, (_dx, _dy) in enumerate(_offsets):
            KICKS[_group, _r, _direction, _k] = (_dx, -_dy)


@njit(cache=True)
def state_index(rotation, x, y):
    return (rotation * Y_COUNT + y - Y_MIN) * X_COUNT + x - X_MIN


@njit(cache=True)
def state_decode(index):
    x = index % X_COUNT + X_MIN
    q = index // X_COUNT
    y = q % Y_COUNT + Y_MIN
    rotation = q // Y_COUNT
    return rotation, x, y


def encode_action(hold, rotation, x, y):
    if not (0 <= int(rotation) < 4 and X_MIN <= x < X_MIN + X_COUNT
            and Y_MIN <= y < Y_MIN + Y_COUNT and int(hold) in (0, 1)):
        raise ValueError("Action anchor is outside the fixed action domain")
    return int(hold) * STATES + int(state_index(rotation, x, y))


def decode_action(action):
    if not 0 <= int(action) < MAX_ACTIONS:
        raise ValueError("Action index is outside the action space")
    rotation, x, y = state_decode(int(action) % STATES)
    return int(action) // STATES, int(rotation), int(x), int(y)


@njit(cache=True)
def fits(board, piece, rotation, x, y):
    for k in range(4):
        xx = x + SHAPES[piece, rotation, k, 0]
        yy = y + SHAPES[piece, rotation, k, 1]
        if xx < 0 or xx >= WIDTH or yy < 0 or yy >= HEIGHT:
            return False
        if board[yy, xx] != 0:
            return False
    return True


@njit(cache=True)
def rotated(board, piece, rotation, x, y, direction):
    nr = (rotation + direction) % 4
    if piece == 3:
        return True, nr, x, y
    group = 1 if piece == 0 else 0
    di = 0 if direction == 1 else 1
    for k in range(5):
        nx = x + KICKS[group, rotation, di, k, 0]
        ny = y + KICKS[group, rotation, di, k, 1]
        if fits(board, piece, nr, nx, ny):
            return True, nr, nx, ny
    return False, rotation, x, y


@njit(cache=True)
def get_reachable(board, piece):
    """Return lock mask, BFS parents/moves, hard-drop sources, input counts.

    The index domain contains every in-bounds SRS anchor. A shortest input
    sequence is retained for each reachable final (rotation, x, y). O rotation
    duplicates are omitted because rotating O never changes occupied cells.
    """
    valid = np.zeros(STATES, dtype=np.bool_)
    parents = np.full(STATES, -2, dtype=np.int16)
    moves = np.full(STATES, -1, dtype=np.int8)
    sources = np.full(STATES, -1, dtype=np.int16)
    lengths = np.zeros(STATES, dtype=np.int16)
    if not fits(board, piece, 0, SPAWN_X, SPAWN_Y):
        return valid, parents, moves, sources, lengths
    start = state_index(0, SPAWN_X, SPAWN_Y)
    queue = np.empty(STATES, dtype=np.int16)
    depth = np.zeros(STATES, dtype=np.int16)
    queue[0] = start
    parents[start] = -1
    head, tail = 0, 1
    while head < tail:
        si = queue[head]
        head += 1
        r, x, y = state_decode(si)
        drop_y = y
        while fits(board, piece, r, x, drop_y + 1):
            drop_y += 1
        final = state_index(r, x, drop_y)
        if not valid[final]:
            valid[final] = True
            sources[final] = si
            lengths[final] = depth[si] + 1
        for move in range(5):
            nr, nx, ny = r, x, y
            if move == 0:
                nx -= 1
            elif move == 1:
                nx += 1
            elif move == 2:
                ny += 1
            else:
                if piece == 3:
                    continue
                ok, nr, nx, ny = rotated(board, piece, r, x, y,
                                          1 if move == 3 else -1)
                if not ok:
                    continue
            if not (X_MIN <= nx < X_MIN + X_COUNT and
                    Y_MIN <= ny < Y_MIN + Y_COUNT):
                continue
            ns = state_index(nr, nx, ny)
            if parents[ns] != -2 or not fits(board, piece, nr, nx, ny):
                continue
            parents[ns] = si
            moves[ns] = move
            depth[ns] = depth[si] + 1
            queue[tail] = ns
            tail += 1
    return valid, parents, moves, sources, lengths


@njit(cache=True)
def lock_piece(board, piece, rotation, x, y):
    """Lock a legal final pose, clear rows, and return board/lines/erosion/topout."""
    after = board.copy()
    for k in range(4):
        after[y + SHAPES[piece, rotation, k, 1],
              x + SHAPES[piece, rotation, k, 0]] = piece + 1
    full = np.zeros(HEIGHT, dtype=np.bool_)
    cleared = 0
    for yy in range(HEIGHT):
        complete = True
        for xx in range(WIDTH):
            if after[yy, xx] == 0:
                complete = False
                break
        full[yy] = complete
        if complete:
            cleared += 1
    eroded = 0
    for k in range(4):
        if full[y + SHAPES[piece, rotation, k, 1]]:
            eroded += 1
    dest = HEIGHT - 1
    for yy in range(HEIGHT - 1, -1, -1):
        if not full[yy]:
            after[dest] = after[yy]
            dest -= 1
    for yy in range(dest + 1):
        after[yy] = 0
    topout = False
    for yy in range(HIDDEN_ROWS):
        for xx in range(WIDTH):
            if after[yy, xx] != 0:
                topout = True
    return after, cleared, cleared * eroded, topout


@njit(cache=True)
def board_features(board):
    """Dellacherie-style raw afterstate features (borders treated occupied).

    Hole depth counts occupied cells above each hole. Cumulative wells sum
    1+...+depth for vertically consecutive empty cells bounded on both sides.
    Hidden rows participate; their empty-row transition terms are constant
    for normal play. These are derived solely from the public board.
    """
    heights = np.zeros(WIDTH, dtype=np.int16)
    hole_rows = np.zeros(HEIGHT, dtype=np.bool_)
    holes, hole_depth, row_trans, col_trans, wells = 0, 0, 0, 0, 0
    for x in range(WIDTH):
        occupied_above, previous = 0, 1
        for y in range(HEIGHT):
            occupied = 1 if board[y, x] != 0 else 0
            if occupied != previous:
                col_trans += 1
            previous = occupied
            if occupied:
                if occupied_above == 0:
                    heights[x] = HEIGHT - y
                occupied_above += 1
            elif occupied_above > 0:
                holes += 1
                hole_depth += occupied_above
                hole_rows[y] = True
        if previous == 0:
            col_trans += 1
        well_depth = 0
        for y in range(HEIGHT):
            if board[y, x] == 0 and (x == 0 or board[y, x - 1] != 0) and (x == WIDTH - 1 or board[y, x + 1] != 0):
                well_depth += 1
                wells += well_depth
            else:
                well_depth = 0
    for y in range(HEIGHT):
        previous = 1
        for x in range(WIDTH):
            occupied = 1 if board[y, x] != 0 else 0
            if occupied != previous:
                row_trans += 1
            previous = occupied
        if previous == 0:
            row_trans += 1
    aggregate, max_height, bumpiness, rows_with_holes = 0, 0, 0, 0
    for x in range(WIDTH):
        aggregate += heights[x]
        max_height = max(max_height, heights[x])
        if x > 0:
            bumpiness += abs(heights[x] - heights[x - 1])
    for y in range(HEIGHT):
        if hole_rows[y]:
            rows_with_holes += 1
    return row_trans, col_trans, holes, wells, aggregate, bumpiness, max_height, hole_depth, rows_with_holes


@njit(cache=True)
def candidate_features(board, piece, valid, lengths, hold_used):
    result = np.zeros((STATES, N_FEATURES), dtype=np.float32)
    for si in range(STATES):
        if not valid[si]:
            continue
        r, x, y = state_decode(si)
        after, cleared, eroded, _ = lock_piece(board, piece, r, x, y)
        rt, ct, holes, wells, agg, bump, mh, hd, rh = board_features(after)
        landing = 0.0
        for k in range(4):
            landing += HEIGHT - (y + SHAPES[piece, r, k, 1] + 0.5)
        result[si, 0] = landing / 4
        result[si, 1] = eroded
        result[si, 2] = rt
        result[si, 3] = ct
        result[si, 4] = holes
        result[si, 5] = wells
        result[si, 6] = agg
        result[si, 7] = bump
        result[si, 8] = cleared
        result[si, 9] = mh
        result[si, 10] = hd
        result[si, 11] = rh
        result[si, 12] = hold_used
        result[si, 13] = lengths[si] + hold_used
    return result


def path_from_reachability(reachability, final_index):
    valid, parents, moves, sources, _ = reachability
    if not valid[final_index]:
        raise ValueError("Placement is unreachable")
    si = int(sources[final_index])
    path = []
    while parents[si] >= 0:
        path.append(MOVE_NAMES[int(moves[si])])
        si = int(parents[si])
    path.reverse()
    path.append("hard_drop")
    return path


def replay_path(board, piece, path):
    """Interpret inputs, checking every intermediate pose; return final pose.

    Hold is deliberately handled by the environment before this interpreter.
    This does not use the BFS parent chain or trust the requested final pose.
    """
    r, x, y = 0, SPAWN_X, SPAWN_Y
    if not fits(board, piece, r, x, y):
        raise ValueError("Spawn collision")
    dropped = False
    for token in path:
        if dropped:
            raise ValueError("Input after hard drop")
        if token in ("left", "right", "down"):
            nx = x + (-1 if token == "left" else 1 if token == "right" else 0)
            ny = y + (1 if token == "down" else 0)
            if not fits(board, piece, r, nx, ny):
                raise ValueError(f"Blocked input: {token}")
            x, y = nx, ny
        elif token in ("cw", "ccw"):
            ok, nr, nx, ny = rotated(board, piece, r, x, y, 1 if token == "cw" else -1)
            if not ok:
                raise ValueError(f"Blocked rotation: {token}")
            r, x, y = nr, nx, ny
        elif token == "hard_drop":
            while fits(board, piece, r, x, y + 1):
                y += 1
            dropped = True
        else:
            raise ValueError(f"Unknown input token: {token}")
    if not dropped:
        raise ValueError("Action path must end in hard_drop")
    return int(r), int(x), int(y)
