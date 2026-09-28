"""Dataset audit: verifies the expert dataset loads and reports which built-in AI actions
fall outside the v0 action space (drives the legal-move enumeration improvements).

Requires rl/data/expert.jsonl.gz (generate with rl/gen_expert.py); skipped when missing.
"""

import gzip
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest

import encoding as enc  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "expert.jsonl.gz")


def test_expert_dataset_loads_and_maps():
    if not os.path.exists(DATA_PATH):
        pytest.skip("expert dataset not generated yet")

    total = 0
    mapped = 0
    unmapped = Counter()

    with gzip.open(DATA_PATH, "rt") as f:
        for line in f:
            record = json.loads(line)
            total += 1

            unit_cells = enc.unit_cells_map(record["state"]["units"])
            legal_slots = []
            for move in record["legal"]:
                act, args = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])
                slot = enc.action_index(act, args, unit_cells)
                if slot is not None:
                    legal_slots.append(slot)

            expert = None
            for move, count in zip(record["legal"], record["counts"]):
                if count > 0:
                    act, args = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])
                    expert = (act, args)
                    break

            assert expert is not None, "one-hot policy target must point at a legal move"

            slot = enc.action_index(expert[0], expert[1], unit_cells)
            if slot is None:
                unmapped[f"act={expert[0]}"] += 1
            else:
                mapped += 1
                assert slot in legal_slots, "the expert action must be present in the legal list"

    assert total > 0
    print(f"\ndataset: {total} records, mapped {mapped}, unmapped {sum(unmapped.values())} {dict(unmapped)}")
