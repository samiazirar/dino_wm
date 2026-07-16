from __future__ import annotations

import pytest

from tools import prestage_p2a
from tools.harness_common import HarnessError


def _spec() -> dict:
    root = "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm"
    return {
        "study_root": root,
        "code_root": f"{root}/code/dino_wm",
        "container": {"path": f"{root}/containers/test.sif", "sha256": "a" * 64},
        "artifacts": {
            "dinov2": {"path": f"{root}/models/dino.pth", "sha256": "b" * 64},
            "dinocular_student": {
                "path": f"{root}/checkpoints/student.pth",
                "sha256": "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc",
            },
        },
    }


def _cards() -> list[dict]:
    return [
        prestage_p2a._card(
            spec=_spec(), commit="c" * 40, source_hashes={"train.py": "d" * 64}, producer=producer
        )
        for producer in prestage_p2a.PRODUCERS
    ]


def _identities(cards: list[dict]) -> dict[str, str]:
    values = set()

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            start = value.find(prestage_p2a.PENDING_PREFIX)
            if start >= 0:
                values.add(value[start:])

    for card in cards:
        visit(card)
    result = {}
    for index, value in enumerate(sorted(values)):
        if value.endswith("_PATH") or value.endswith("_DIR"):
            result[value] = (
                "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/"
                f"synthetic/{index}"
            )
        else:
            result[value] = f"{index + 1:064x}"[-64:]
    return result


def test_prestage_fails_closed_then_is_structurally_launchable() -> None:
    cards = _cards()
    with pytest.raises(HarnessError, match="pending immutable evidence"):
        prestage_p2a.validate_cards(cards)
    result = prestage_p2a.validate_cards(cards, _identities(cards))
    assert result["state"] == "STRUCTURALLY_LAUNCHABLE"
    assert result["card_count"] == 2
    assert result["sbatch_calls"] == 0


def test_prestage_locks_pairing_resume_and_assumption() -> None:
    cards = _cards()
    assert {card["segment_steps"] for card in cards} == {11_000}
    assert {card["target_steps"] for card in cards} == {123_858}
    assert {card["seed"] for card in cards} == {1}
    assert cards[0]["fixed_evaluation_manifest"] == cards[1]["fixed_evaluation_manifest"]
    assert cards[1]["assumption_tags"] == [prestage_p2a.ASSUMPTION]
    assert all(card["done_28_execution_gate"] is False for card in cards)
    assert all(card["strict_determinism"]["num_workers"] == 0 for card in cards)
