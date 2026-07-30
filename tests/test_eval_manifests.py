from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_encoder_swap import _load_target_observations


def test_fixed_wall_manifest_has_exact_hash_selection_and_raw_action_indices(tmp_path):
    root = tmp_path / "raw"
    wall = root / "wall_single"
    (wall / "obses").mkdir(parents=True)
    episodes = 10
    frames = 70
    torch.save(torch.zeros(episodes, frames, 2), wall / "actions.pth")
    generator = torch.Generator().manual_seed(42)
    valid = sorted(torch.randperm(episodes, generator=generator).tolist()[9:])
    for episode in valid:
        torch.save(
            torch.zeros(frames, 3, 8, 8, dtype=torch.uint8),
            wall / "obses" / f"episode_{episode:03d}.pth",
        )
    output = tmp_path / "openloop_wall.jsonl"
    tool = Path(__file__).resolve().parents[1] / "eval_encoder_swap.py"
    subprocess.run(
        [
            "python3",
            str(tool),
            "manifest",
            "--env",
            "wall",
            "--root",
            str(root),
            "--n",
            "1000",
            "--seed",
            "20260714",
            "--horizons",
            "1,5,10",
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    metadata = json.loads(output.with_suffix(".meta.json").read_text())
    assert rows
    assert len(rows) == len({row["key"] for row in rows})
    assert metadata["selection"] == "smallest_sha256_start_keys"
    assert metadata["manifest_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert all(row["frameskip"] == 5 and row["horizons"] == [1, 5, 10] for row in rows)
    assert all(len(indices) == 5 for row in rows for indices in row["raw_action_indices"].values())
    ordered = sorted(
        rows,
        key=lambda row: (hashlib.sha256(row["key"].encode()).hexdigest(), row["key"]),
    )
    assert rows == ordered[: len(rows)]


def test_wall_target_loader_reads_terminal_observation_without_terminal_action(tmp_path):
    wall = tmp_path / "wall_single"
    (wall / "obses").mkdir(parents=True)
    images = torch.arange(51, dtype=torch.float32).reshape(51, 1, 1, 1)
    torch.save(images, wall / "obses" / "episode_003.pth")
    dataset = SimpleNamespace(
        data_path=wall,
        get_seq_length=lambda _episode: 50,
        proprios=torch.arange(50, dtype=torch.float32)
        .reshape(1, 50, 1)
        .repeat(4, 1, 1),
        transform=lambda value: value + 0.5,
        depth_reader=None,
    )

    observations = _load_target_observations(dataset, "wall", 3, [5, 25, 50])

    torch.testing.assert_close(
        observations["visual"].flatten(),
        torch.tensor([5, 25, 50], dtype=torch.float32) / 255 + 0.5,
    )
    assert observations["proprio"].flatten().tolist() == [5, 25, 49]
