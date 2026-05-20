"""Multi-positive retrieval metrics used by Task 2."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def multipositive_retrieval_metrics(
    ranked_ids: Sequence[Sequence[str]],
    positive_ids: Sequence[Iterable[str]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K, mAP, and median rank for multi-positive retrieval."""

    if len(ranked_ids) != len(positive_ids):
        raise ValueError("ranked_ids and positive_ids must have the same length.")
    if not ranked_ids:
        raise ValueError("Cannot evaluate an empty retrieval result.")

    positives = [set(ids) for ids in positive_ids]
    recalls = {k: 0 for k in ks}
    ap_sum = 0.0
    ranks: list[int] = []

    for ranking, pos in zip(ranked_ids, positives):
        if not pos:
            continue
        hit_ranks = [idx + 1 for idx, item_id in enumerate(ranking) if item_id in pos]
        first_rank = hit_ranks[0] if hit_ranks else len(ranking) + 1
        ranks.append(first_rank)
        for k in ks:
            recalls[k] += int(first_rank <= k)

        hits = 0
        precision_sum = 0.0
        for idx, item_id in enumerate(ranking, start=1):
            if item_id in pos:
                hits += 1
                precision_sum += hits / idx
        ap_sum += precision_sum / min(len(pos), len(ranking)) if ranking else 0.0

    denom = len(ranked_ids)
    sorted_ranks = sorted(ranks)
    mid = len(sorted_ranks) // 2
    median_rank = (
        sorted_ranks[mid]
        if len(sorted_ranks) % 2
        else 0.5 * (sorted_ranks[mid - 1] + sorted_ranks[mid])
    )
    metrics = {f"R@{k}": recalls[k] / denom for k in ks}
    metrics["mAP"] = ap_sum / denom
    metrics["median_rank"] = float(median_rank)
    return metrics
