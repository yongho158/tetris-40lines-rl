import json

import torch

from tetris_rl.policy import DELLACHERIE_WEIGHTS, LinearCandidatePolicy, load_policy
from tetris_rl.train import inherit_cem_incumbent, restore_torch_rng


def test_cross_directory_resume_carries_actual_best_policy(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    weights = DELLACHERIE_WEIGHTS * 0.75
    LinearCandidatePolicy(weights).save(source / "best.pt", {"rl_updated": True})
    (source / "best_validation.json").write_text(json.dumps({
        "metrics": {"success_rate": 1.0}, "episodes": [{"seed": 1_000_000}],
    }))

    inherit_cem_incumbent(source, destination)

    assert torch.equal(load_policy(destination / "best.pt").weights, torch.tensor(weights))
    report = json.loads((destination / "best_validation.json").read_text())
    assert report["metrics"]["success_rate"] == 1.0
    assert report["inherited_from"] == str(source / "best.pt")


def test_cross_directory_resume_does_not_replace_existing_incumbent(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    for directory, marker in [(source, b"source"), (destination, b"destination")]:
        (directory / "best.pt").write_bytes(marker)
        (directory / "best_validation.json").write_text("{}")
    inherit_cem_incumbent(source, destination)
    assert (destination / "best.pt").read_bytes() == b"destination"


def test_resume_restores_cpu_torch_random_stream():
    original = torch.get_rng_state()
    try:
        torch.manual_seed(991)
        saved = {"torch_rng_state": torch.get_rng_state().tolist()}
        expected = torch.rand(32)
        torch.manual_seed(123)
        assert restore_torch_rng(saved)
        assert torch.equal(torch.rand(32), expected)
        assert not restore_torch_rng({"rng_state": {}})
    finally:
        torch.set_rng_state(original)
