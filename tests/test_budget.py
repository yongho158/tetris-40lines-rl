import json

from tetris_rl.budget import Budget


def test_budget_never_overruns_and_resume_cannot_expand_limits(tmp_path):
    budget = Budget(tmp_path, max_seconds=10, max_transitions=3)
    assert budget.record(2)
    assert not budget.record(2)
    assert budget.record()
    assert not budget.record()
    budget.save()
    resumed = Budget(tmp_path, max_seconds=100, max_transitions=100)
    assert resumed.max_transitions == 3
    assert resumed.max_seconds == 10
    assert resumed.exhausted


def test_crash_reservation_is_charged_conservatively(tmp_path):
    first = Budget(tmp_path, max_transitions=1000)
    assert first.record()
    persisted = json.loads((tmp_path / "budget.json").read_text())
    assert persisted["reserved_transitions"] >= 1
    resumed = Budget(tmp_path, max_transitions=1000)
    assert resumed.transitions == persisted["reserved_transitions"]


def test_expired_wall_deadline_blocks_transitions(tmp_path):
    budget = Budget(tmp_path, max_seconds=10)
    budget.started_at -= 11
    assert budget.exhausted
    assert not budget.record()
