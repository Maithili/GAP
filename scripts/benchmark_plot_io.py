"""Lightweight benchmark JSON reads for plotting (avoids OOM on huge STREAK files)."""
from __future__ import annotations

import json
import os

from final_deliverable import _extract_totals

_SLIM_ROW_KEYS = ("human_metrics", "result_metrics", "env_metrics")
_MAX_SMALL = 150_000_000


def _slim_row(row: dict) -> dict:
    return {k: row.get(k) or {} for k in _SLIM_ROW_KEYS}


def load_benchmark_plot_data(path: str) -> dict:
    """Return {totals, per_day, ckpt_only}.

    Small files: slim per_day rows for daily macro averaging.
    Large files: totals only; plots use finetune_by_ckpt window aggregates.
    """
    totals = _extract_totals(path)
    if os.path.getsize(path) > _MAX_SMALL:
        return {"totals": totals, "per_day": None, "ckpt_only": True}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    per_day = [_slim_row(r) for r in (data.get("per_day") or [])]
    return {"totals": totals, "per_day": per_day, "ckpt_only": False}


def ckpt_window_days(entry: dict, ckpt: int) -> int:
    ws = entry.get("window_start_day_idx", ckpt)
    we = entry.get("window_end_day_idx")
    return (we - ws + 1) if we is not None else 5


def finetune_day_count(totals: dict) -> int:
    bc = (totals or {}).get("finetune_by_ckpt") or {}
    indices: set[int] = set()
    for ckpt_str, entry in bc.items():
        ckpt = int(ckpt_str)
        ws = entry.get("window_start_day_idx", ckpt)
        we = entry.get("window_end_day_idx")
        chunk_end = (we + 1) if we is not None else ws + 5
        indices.update(range(ws, chunk_end))
    return len(indices) or 1


def finetune_per_day_ledger(jf: str, totals: dict):
    from effort_ledger import ledger_from_finetune, ledger_per_day

    ledger = ledger_from_finetune(jf)
    if ledger is None:
        return None
    return ledger_per_day(ledger, finetune_day_count(totals))


def ledger_from_ckpt_entry(entry: dict, ckpt: int):
    from effort_ledger import ledger_from_sources, ledger_per_day

    hm = entry.get("human_metrics") or {}
    em = entry.get("env_metrics") or {}
    ledger = ledger_from_sources(hm, em)
    if ledger is None:
        return None
    return ledger_per_day(ledger, ckpt_window_days(entry, ckpt))
