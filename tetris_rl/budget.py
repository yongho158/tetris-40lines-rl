"""Persistent, conservative global budget shared by sequential experiments.

Call record BEFORE every training/validation environment transition. A reservation
watermark makes crashes conservative: a resumed process charges any reserved but
uncommitted transitions. Call save in a finally block on orderly shutdown.
The absolute wall deadline includes pauses between experiments, so it never
understates the user's five-hour upper bound.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path


class Budget:
    def __init__(self, run_dir, max_seconds=18000, max_transitions=5_000_000):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "budget.json"
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            self.started_at = float(state["started_at_unix"])
            self.max_seconds = min(float(max_seconds), float(state["max_seconds"]))
            self.max_transitions = min(int(max_transitions), int(state["max_transitions"]))
            self.transitions = int(state.get("reserved_transitions", state["transitions"]))
        else:
            self.started_at = time.time()
            self.max_seconds = float(max_seconds)
            self.max_transitions = int(max_transitions)
            self.transitions = 0
        if self.max_seconds <= 0 or self.max_transitions < 0:
            raise ValueError("Budget limits must be positive (transitions may be zero).")
        self._reserved = self.transitions
        self.save()

    @property
    def elapsed_seconds(self):
        return max(0.0, time.time() - self.started_at)

    @property
    def remaining_steps(self):
        return max(0, self.max_transitions - self.transitions)

    @property
    def exhausted(self):
        return self.remaining_steps == 0 or self.elapsed_seconds >= self.max_seconds

    def record(self, n=1):
        if not isinstance(n, int) or n < 0:
            raise ValueError("record(n) needs a nonnegative integer")
        if self.exhausted or n > self.remaining_steps:
            return False
        if self.transitions + n > self._reserved:
            self._reserved = min(self.max_transitions, max(self.transitions + n, self.transitions + 128))
            self._write()
        self.transitions += n
        return True

    def snapshot(self):
        return {
            "started_at_unix": self.started_at,
            "started_at_utc": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            "max_seconds": self.max_seconds,
            "max_transitions": self.max_transitions,
            "transitions": self.transitions,
            "reserved_transitions": self._reserved,
            "elapsed_seconds": self.elapsed_seconds,
            "remaining_steps": self.remaining_steps,
            "deadline_utc": datetime.fromtimestamp(self.started_at + self.max_seconds, timezone.utc).isoformat(),
            "exhausted": self.exhausted,
            "accounting": "all training and validation steps; elapsed wall time across sequential runs; crash reservations charged conservatively",
        }

    def _write(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, self.path)

    def save(self):
        self._reserved = self.transitions
        self._write()

