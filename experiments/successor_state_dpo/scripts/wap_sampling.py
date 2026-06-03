import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def split_budget(total: int, groups: Sequence[Tuple]) -> Dict[Tuple, int]:
    groups = list(groups)
    base = total // len(groups)
    remainder = total % len(groups)
    return {
        group: base + (1 if idx < remainder else 0)
        for idx, group in enumerate(groups)
    }


def group_key(item: Dict, fields: Sequence[str]) -> Tuple:
    return tuple(item.get(field, "unknown") for field in fields)


def balanced_reservoir_sample(
    jsonl_path: str,
    max_samples: Optional[int],
    seed: int,
    group_fields: Sequence[str],
    groups: Sequence[Tuple],
) -> List[Dict]:
    groups = list(groups)
    group_set = set(groups)

    if max_samples is None:
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if group_key(item, group_fields) in group_set:
                    samples.append(item)
        return samples

    rng = random.Random(seed)
    budgets = split_budget(max_samples, groups)
    reservoirs = {group: [] for group in groups}
    seen = defaultdict(int)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            key = group_key(item, group_fields)
            if key not in budgets:
                continue

            seen[key] += 1
            budget = budgets[key]
            reservoir = reservoirs[key]

            if len(reservoir) < budget:
                reservoir.append(item)
                continue

            replace_idx = rng.randint(0, seen[key] - 1)
            if replace_idx < budget:
                reservoir[replace_idx] = item

    samples = []
    for group in groups:
        samples.extend(reservoirs[group])

    rng.shuffle(samples)
    return samples
