"""Clean effort ledger from benchmark JSON metric blocks.

Buckets (see findings / user spec):
  solo baseline  — script TELEPORT inventory (fixed on complete runs)
  human actions  — trips the human actually spends
  robot actions  — one tag per robot TELEPORT in result/env metrics
  net effort     — savings − (redirected + undo); positive = trips saved vs solo
  made harder    — corrected + corrected_repeated + cdc + redirected

Works from per_day rows (human_metrics + result_metrics) or finetune totals
(human_metrics + env_metrics) without loading full benchmark JSON.
"""
from __future__ import annotations

SOLO_KEYS = (
    "corrected",
    "corrected_repeated",
    "corrected_despite_correct",
    "done_as_is",
    "done_early",
    "accepted",
    "skipped",
    "skipped_repeated",
)

SAVINGS_KEYS = ("done_early", "accepted", "skipped", "skipped_repeated")

ROBOT_KEYS = (
    "skipped",
    "corrected",
    "corrected_despite_correct",
    "accepted",
    "corrected_through_vigilance",
    "redirected",
    "unseen",
    "unseen_at_end_of_day",
)


def _g(d: dict | None, key: str) -> int:
    return int((d or {}).get(key, 0) or 0)


def _undo(hm: dict | None, em: dict | None) -> int:
    return _g(em, "vigilance_corrections") or _g(hm, "vigilance_corrections")


def _redirected(hm: dict | None, em: dict | None) -> int:
    return _g(em, "redirected") or _g(hm, "redirected")


def _robot_ctv(em: dict | None) -> int:
    return _g(em, "corrected_through_vigilance") or _g(em, "vigilance_corrections")


def ledger_from_sources(hm: dict | None, em: dict | None) -> dict | None:
    """Return ledger dict, or None when solo baseline is zero."""
    hm = hm or {}
    em = em or {}
    g = lambda k: _g(hm, k)

    undo = _undo(hm, em)
    redirected = _redirected(hm, em)

    solo = sum(g(k) for k in SOLO_KEYS)
    if solo == 0:
        return None

    savings = sum(g(k) for k in SAVINGS_KEYS)
    human = (
        g("corrected")
        + g("corrected_repeated")
        + g("corrected_despite_correct")
        + g("done_as_is")
        + redirected
        + undo
    )
    robot = sum(_g(em, k) for k in ROBOT_KEYS if k != "corrected_through_vigilance")
    robot += _robot_ctv(em)

    net_abs = savings - redirected - undo
    harder_abs = (
        g("corrected")
        + g("corrected_repeated")
        + g("corrected_despite_correct")
        + redirected
    )

    return {
        "solo": solo,
        "human": human,
        "human_pct": 100.0 * human / solo,
        "robot": robot,
        "savings": savings,
        "savings_pct": 100.0 * savings / solo,
        "undo": undo,
        "redirected": redirected,
        "net_abs": net_abs,
        "net_pct": 100.0 * net_abs / solo,
        "harder_abs": harder_abs,
        "harder_pct": 100.0 * harder_abs / solo,
    }


def aggregate_dicts(dicts: list[dict | None]) -> dict:
    """Sum numeric keys across metric dicts."""
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in (d or {}).items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + int(v or 0)
    return out


def ledger_from_rows(rows: list[dict]) -> dict | None:
    """Ledger from pooled per_day rows (micro-aggregation before metric compute)."""
    if not rows:
        return None
    hm = aggregate_dicts([r.get("human_metrics") or {} for r in rows])
    em = aggregate_dicts([
        r.get("result_metrics") or r.get("env_metrics") or {} for r in rows
    ])
    return ledger_from_sources(hm, em)


_COUNT_PER_DAY_KEYS = frozenset({
    "human", "robot", "solo", "savings", "undo", "redirected",
    "net_abs", "harder_abs",
})


def ledger_per_day(ledger: dict, n_days: int) -> dict:
    """Scale count metrics to a per-day rate; leave % metrics unchanged."""
    if n_days <= 0:
        return ledger
    out = dict(ledger)
    for k in _COUNT_PER_DAY_KEYS:
        out[k] = ledger[k] / n_days
    return out


def ledger_from_row(row: dict) -> dict | None:
    hm = row.get("human_metrics") or {}
    em = row.get("result_metrics") or row.get("env_metrics") or {}
    return ledger_from_sources(hm, em)


def ledger_from_finetune(jf: str) -> dict | None:
    from final_deliverable import _extract_totals

    t = _extract_totals(jf).get("finetune") or {}
    hm = t.get("human_metrics") or {}
    em = t.get("env_metrics") or {}
    return ledger_from_sources(hm, em)
