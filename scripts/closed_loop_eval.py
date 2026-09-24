"""
Closed-loop benchmark harness (library module).

TL;DR: ``evaluate_closed_loop`` runs robot→human each tick over script days;
``_aggregate_totals`` rolls up human-burden metrics and, when
``suppress_robot_action=True``, paper-style SLaTe-PRO tensor metrics.

Imported by ``scripts/benchmark_comparison.py`` (no standalone CLI here).
Formerly named ``evaluate_continual.py`` for the removed continual DPO model.
"""

import copy
import json
import hashlib
import os
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'rail_tasksim'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'rail_tasksim', 'simulation'))

import torch

from simulation import load_config
from agents.robot_policy_slatepro import RobotPolicySlatepro
from agents.robot_policy_streak import RobotPolicyStreak
from agents.robot_policy_hybrid import RobotPolicyHybrid
from agents.robot_policy_plainllm import RobotPolicyPlainLLM
from agents.robot_policy_oracle import RobotPolicyOracle
from agents.robot_policy_nopa import RobotPolicyNOPA
from agents.slatepro_utils import homers_script_path_to_processed_pt

from environment.world import print_graph_difference

import time
from model.vocab import VH_ID_TO_NAME
from scripts.visualize_comparison import plot_prediction_heatmap as _plot_prediction_heatmap
from datetime import datetime
from dataclasses import dataclass

# Debug plots: per_day (o) vs actionwise (^) at evaluation time
from collections import Counter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class ResumePrep:
    """Paths and padded state returned by ``prepare_resume_and_paths``."""
    out_stem: str | None
    resume_path: str | None
    start_day_idx: int
    phase: str
    per_day: list[dict]
    actionwise_evals_daily: list



def _atomic_write(path, write_fn):
    """Write via a temp file in the same dir, then os.replace() onto *path*.

    Every per-day artifact (observed day file, result json, resume checkpoint)
    is rewritten in place each day. A plain open('w')/torch.save truncates the
    previous good copy first, so a kill or ENOSPC mid-write leaves a corrupt
    file: the Plain-LLM wrapper's 4h self-resubmit exits while Python is still
    saving (corrupted a 300 MB resume checkpoint), and a full filesystem
    truncated a 1.6 GB result json. os.replace is atomic on POSIX, so readers
    and resumes only ever see the old complete file or the new complete one.
    The temp file is removed if the write fails, and the error re-raised."""
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _perday_jsonl_path(out_path):
    """Append-only per-day sidecar next to the run's result json."""
    stem = os.path.basename(out_path).replace(".json", "")
    return os.path.join(os.path.dirname(out_path), f"{stem}_perday.jsonl")


def _append_perday(out_path, day_idx, day_rec, actionwise, train_rec):
    """Append ONE day's records (~MBs) instead of rewriting the whole result.

    The result json holds every day's record, so rewriting it after each day
    costs the full file every time -- 3.6 GB/day by day 50, which on a shared
    node was the entire per-day runtime. The per-day records go here as one
    JSON object per line; `_assemble_result` turns them back into the same
    result json once, at the end.
    """
    line = json.dumps(_json_sanitize({
        "day": day_idx, "per_day": day_rec,
        "actionwise": actionwise, "training_dataset": train_rec}))
    with open(_perday_jsonl_path(out_path), "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_perday_jsonl(out_path, upto_day):
    """Replay the sidecar up to and including *upto_day* (resume path).

    Later duplicates of a day win, so a day re-simulated after a crash
    replaces the earlier attempt rather than appearing twice.
    """
    path = _perday_jsonl_path(out_path)
    if not os.path.isfile(path):
        raise RuntimeError(
            f"resume needs per-day records but {path} is missing; "
            f"delete the resume checkpoint to start this household over")
    by_day = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r["day"] <= upto_day:
                by_day[r["day"]] = r
    days = sorted(by_day)
    return ([by_day[d]["per_day"] for d in days],
            [by_day[d]["actionwise"] for d in days],
            {d: by_day[d]["training_dataset"] for d in days
             if by_day[d]["training_dataset"] is not None})


def _dump_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def backup_existing_output_json(path: str, *, log_prefix: str = "use_prior_checkpoints") -> str | None:
    """
    If `path` exists, rename it to add `_before_DDMM_HHMM` before the .json suffix.
    Returns the backup path if a backup was created; otherwise None.
    """
    if not path or not os.path.isfile(path):
        return None

    stamp = datetime.now().strftime("%d%m_%H%M")
    base = os.path.basename(path)
    if base.endswith(".json"):
        stem = base[:-5]
        ext = ".json"
    else:
        stem = base
        ext = ""

    candidate = os.path.join(os.path.dirname(path), f"{stem}_before_{stamp}{ext}")
    # Avoid collisions if called multiple times in the same minute.
    if os.path.exists(candidate):
        k = 2
        while True:
            cand2 = os.path.join(os.path.dirname(path), f"{stem}_before_{stamp}_{k}{ext}")
            if not os.path.exists(cand2):
                candidate = cand2
                break
            k += 1

    os.replace(path, candidate)
    print(f"[{log_prefix}] renamed existing output -> {candidate}")
    return candidate


def prepare_resume_and_paths(
    *,
    out_path: str | None,
    use_prior_checkpoints: bool,
    phase: str,
    phase_ends: dict[str, int],
    script_distribution: list[int],
    per_day: list[dict],
    actionwise_evals_daily: list,
    start_day_idx: int = 0,
) -> ResumePrep:
    """
    Centralizes output-path handling + resume policy for closed-loop runs.

    - If `out_path` is set, compute `{out_stem}_resume.pt` next to it.
    - If `use_prior_checkpoints` is True, we treat this as a "redone" run:
        - back up existing out_path JSON (if present)
        - never read/write a resume file
        - skip warm-start phase and pad per-day arrays accordingly
    """
    if use_prior_checkpoints and out_path:
        backup_existing_output_json(out_path, log_prefix="use_prior_checkpoints")

    if out_path:
        out_stem = os.path.basename(out_path).replace(".json", "")
        resume_path = os.path.join(os.path.dirname(out_path), f"{out_stem}_resume.pt")
    else:
        out_stem = None
        resume_path = None

    if use_prior_checkpoints:
        # Never read/write a resume file for redone runs
        resume_path = None

        # Skip the warm-start phase entirely
        start_day_idx = max(start_day_idx, phase_ends["warm-start"])
        if phase == "warm-start":
            phase = "finetune" if script_distribution[1] > 0 else "test"

        # Pad per_day / actionwise_evals_daily so day_idx indexing (and the
        # offsets _aggregate_totals uses) match a run that did process warm-start.
        per_day = [{} for _ in range(phase_ends["warm-start"])]
        actionwise_evals_daily = [{} for _ in range(phase_ends["warm-start"])]

    return ResumePrep(
        out_stem=out_stem,
        resume_path=resume_path,
        start_day_idx=start_day_idx,
        phase=phase,
        per_day=per_day,
        actionwise_evals_daily=actionwise_evals_daily,
    )

def _json_sanitize(obj):
    """
    Recursively convert common ML objects (Tensor/ndarray/scalars) into
    JSON-serializable Python types.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # torch.Tensor (and tensor-like)
    if torch.is_tensor(obj):
        t = obj.detach().cpu()
        if t.numel() == 1:
            return t.item()
        return t.tolist()

    # numpy arrays / scalars (optional dependency in some envs)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    # mapping / sequences
    if isinstance(obj, dict):
        # Coerce keys to strings to be safe for JSON.
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]

    # sets, Counters, etc.
    if isinstance(obj, set):
        return [_json_sanitize(v) for v in obj]

    # best-effort fallback
    return str(obj)

def name_from_vh_index(vh_index: int) -> str:
    """Map a VirtualHome node id to its class name via ``VH_ID_TO_NAME``."""
    return VH_ID_TO_NAME[vh_index]


def _classify_per_object_tensor(fwd, offline_edges, offline_times, current_time):
    """Per-(slate_obj_idx) tensor classification for one timestep.

    Returns {} when comparison cannot run (no fwd, no offline data,
    current_time absent from offline_times, or idx+1 out of range). Otherwise
    returns {slate_obj_idx -> {'1step': cat, 'anyFuture': cat}} for every
    dynamic-mask object.

    Categories (same taxonomy for both views):
        'precision_correct'           pred and gt both moved object to same dest
        'precision_wrong_destination' pred and gt both moved, different dests
        'precision_wrong_object'      pred moved, gt did not
        'moved_missed'                gt moved, pred did not
        'no_event'                    neither moved

    1step view: gt = offline_edges[idx+1][X] (one step ahead).
    anyFuture view: scan offline_edges[idx+1:end][X]; the prediction is treated
        as 'precision_correct' if X is ever observed at pred_loc; treated as
        'precision_wrong_destination' if X moves at any future step but never
        reaches pred_loc; 'precision_wrong_object' if X never moves at all.
        For objects the model didn't predict to move (cp=False), the view is
        'moved_missed' if X ever moves in the future, else 'no_event'.
    """
    if fwd is None or offline_edges is None:
        return {}
    match = (offline_times == float(current_time)).nonzero(as_tuple=False)
    if match.numel() == 0:
        return {}
    idx = int(match[0, 0].item())
    if idx + 1 >= offline_edges.shape[0]:
        return {}
    current_locs = offline_edges[idx].argmax(-1)
    gt_locs_next = offline_edges[idx + 1].argmax(-1)
    pred_locs    = fwd['pred_locations']
    dyn_mask     = fwd['dyn_mask_flat']
    # (F, N) future-only locations from idx+1 onward.
    future_locs  = offline_edges[idx + 1:].argmax(-1)

    out: dict[int, dict[str, str]] = {}
    for i in range(dyn_mask.shape[0]):
        if not bool(dyn_mask[i]):
            continue
        cur = int(current_locs[i])
        gt_next = int(gt_locs_next[i])
        p = int(pred_locs[i])
        cp = (p != cur)

        # 1-step view
        cg_1 = (gt_next != cur)
        ok_1 = (p == gt_next)
        if cp and cg_1 and ok_1:    cat_1 = 'precision_correct'
        elif cp and cg_1:           cat_1 = 'precision_wrong_destination'
        elif cp and not cg_1:       cat_1 = 'precision_wrong_object'
        elif not cp and cg_1:       cat_1 = 'moved_missed'
        else:                       cat_1 = 'no_event'

        # anyFuture view
        col = future_locs[:, i]
        ever_moves = bool((col != cur).any())
        ever_at_pred = bool((col == p).any())
        if cp and ever_moves and ever_at_pred:  cat_af = 'precision_correct'
        elif cp and ever_moves:                 cat_af = 'precision_wrong_destination'
        elif cp and not ever_moves:             cat_af = 'precision_wrong_object'
        elif not cp and ever_moves:             cat_af = 'moved_missed'
        else:                                   cat_af = 'no_event'

        out[int(i)] = {'1step': cat_1, 'anyFuture': cat_af}
    return out


_DEBUG_OUTCOME_COLORS = {
    "skipped": "#2ca02c",  # green
    "corrected": "#d62728",  # red
    "done_as_is": "#c4a661",  # beige / tan
    "accepted": "#17a2b8",  # cyan — LLM-human chose to go along
    "corrected_despite_correct": "#8c564b",  # brown — moved away though robot was right
    "redirected": "#e377c2",  # pink — moved straight to planned dest (one trip)
    "done_early": "#bbbbbb",  # grey — planned line already done by earlier redirect (ignored)
    "corrected_through_vigilance": "#ff9999",  # lighter red
    "no_movement": "#9ecae1",  # light blue
    "no_need_to_move": "#6baed6",  # mid blue
    "other": "#7f7f7f",  # grey
}


_DEBUG_TENSOR_COLORS = {
    "precision_correct":           "#1b7a1b",  # deep green
    "precision_wrong_destination": "#b8001f",  # deep red
    "precision_wrong_object":      "#8e44ad",  # purple
    "moved_missed":                "#806b3f",  # darker tan (sibling of done_as_is #c4a661)
    "no_event":                    "#bdbdbd",  # light grey
    "unknown":                     "#000000",  # black (non-teleport script lines)
    "NONE":                        "#000000",
}


def _debug_color_for(outcome: str) -> str:
    """Hex color for a human-outcome label in debug scatter plots."""
    return _DEBUG_OUTCOME_COLORS.get(outcome, _DEBUG_OUTCOME_COLORS["other"])


def _debug_tensor_color_for(tensor_cat: str) -> str:
    """Hex color for a tensor-eval category in debug scatter plots."""
    return _DEBUG_TENSOR_COLORS.get(tensor_cat, _DEBUG_TENSOR_COLORS["NONE"])


def _debug_extract_per_day_points(per_day_entry: dict) -> list[tuple[int, str, str, str, str]]:
    events = per_day_entry.get("human_action_events") or []
    out: list[tuple[int, str, str, str, str]] = []
    for e in events:
        t = e.get("time_minutes", e.get("timestep"))
        obj = e.get("object_name") or e.get("object")
        outcome = e.get("outcome")
        cat_1s = e.get("tensor_eval_category_1step",
                       e.get("tensor_eval_category", "no_event"))
        cat_af = e.get("tensor_eval_category_anyFuture", cat_1s)
        if t is None or not obj:
            continue
        out.append((int(t), str(obj),
                    str(outcome) if outcome is not None else "other",
                    str(cat_1s) if cat_1s is not None else "no_event",
                    str(cat_af) if cat_af is not None else "no_event"))
    return out


def _debug_extract_actionwise_points(day_actionwise: dict) -> list[tuple[int, str, str, str, str]]:
    """
    Plot actionwise points at evaluation time:
      t_eval = t_pred + eval_time_from_prediction
    """
    out: list[tuple[int, str, str, str, str]] = []
    for ts_str, actions in (day_actionwise or {}).items():
        try:
            t_pred = int(ts_str)
        except Exception:
            continue
        for a in actions:
            obj = a.get("object_name") or a.get("object")
            outcome = a.get("human_action_outcome") or a.get("label") or "other"
            cat_1s = a.get("tensor_eval_category_1step",
                           a.get("tensor_eval_category", "no_event"))
            cat_af = a.get("tensor_eval_category_anyFuture", cat_1s)
            if not obj:
                continue
            dt = a.get("eval_time_from_prediction", 0)
            try:
                dt_i = int(dt)
            except Exception:
                dt_i = 0
            t_eval = t_pred + dt_i
            out.append((t_eval, str(obj), str(outcome),
                        str(cat_1s) if cat_1s is not None else "no_event",
                        str(cat_af) if cat_af is not None else "no_event"))
    return out


def _debug_plot_day_actions_compare(
    *,
    per_day_entry: dict,
    day_actionwise: dict,
    out_path: str,
    title: str,
    y_gap: float = 0.28,
    max_objects: int = 25,
) -> None:
    per_pts = _debug_extract_per_day_points(per_day_entry)
    aw_pts = _debug_extract_actionwise_points(day_actionwise)

    obj_freq = Counter([o for _, o, _, _, _ in per_pts] + [o for _, o, _, _, _ in aw_pts])
    objects_sorted = [o for o, _ in obj_freq.most_common(max_objects)]
    objects_sorted = list(reversed(objects_sorted))  # most frequent at top
    obj_to_y = {o: i for i, o in enumerate(objects_sorted)}

    fig_h = max(4.5, 0.42 * max(8, len(objects_sorted)))
    fig, ax = plt.subplots(figsize=(18, fig_h))

    # per_day: marker 'o' — face = human outcome, edge = tensor 1step category
    xs = [t for t, o, _, _, _ in per_pts if o in obj_to_y]
    ys = [obj_to_y[o] for _, o, _, _, _ in per_pts if o in obj_to_y]
    cs = [_debug_color_for(out) for _, o, out, _, _ in per_pts if o in obj_to_y]
    edge_cs = [_debug_tensor_color_for(tc) for _, o, _, tc, _ in per_pts if o in obj_to_y]
    ax.scatter(
        xs, ys, c=cs, marker="o", s=120, alpha=0.95,
        edgecolors=edge_cs, linewidths=1.6, zorder=3,
    )
    # Inner dot: anyFuture tensor category
    inner_cs = [_debug_tensor_color_for(tc) for _, o, _, _, tc in per_pts if o in obj_to_y]
    ax.scatter(
        xs, ys, c=inner_cs, marker="o", s=20, alpha=1.0,
        edgecolors="none", zorder=5,
    )

    # actionwise: marker '^' with Y offset — face = human outcome, edge = 1step
    xs2 = [t for t, o, _, _, _ in aw_pts if o in obj_to_y]
    ys2 = [obj_to_y[o] + y_gap for _, o, _, _, _ in aw_pts if o in obj_to_y]
    cs2 = [_debug_color_for(out) for _, o, out, _, _ in aw_pts if o in obj_to_y]
    edge_cs2 = [_debug_tensor_color_for(tc) for _, o, _, tc, _ in aw_pts if o in obj_to_y]
    alphas2 = [
        0.6 if out == "corrected_through_vigilance" else 0.9
        for _, o, out, _, _ in aw_pts
        if o in obj_to_y
    ]
    ax.scatter(
        xs2, ys2, c=cs2, marker="^", s=130, alpha=1.0,
        edgecolors=edge_cs2, linewidths=1.6, zorder=4,
    )
    # Inner dot for triangles too: anyFuture
    inner_cs2 = [_debug_tensor_color_for(tc) for _, o, _, _, tc in aw_pts if o in obj_to_y]
    ax.scatter(
        xs2, ys2, c=inner_cs2, marker="o", s=20, alpha=1.0,
        edgecolors="none", zorder=5,
    )
    # Apply per-point alpha if backend supports it.
    try:
        coll = ax.collections[-1]
        fc = coll.get_facecolors()
        if len(fc) == len(alphas2):
            for i, a in enumerate(alphas2):
                fc[i, 3] = a
            coll.set_facecolors(fc)
    except Exception:
        pass

    ax.set_yticks(range(len(objects_sorted)))
    ax.set_yticklabels(objects_sorted, fontsize=10)
    ax.set_ylim(-0.5, len(objects_sorted) - 0.5 + y_gap)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Object")
    ax.set_title(title)

    # Legend (markers + outcome colors)
    from matplotlib.lines import Line2D

    outcome_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_DEBUG_OUTCOME_COLORS["skipped"],
            markersize=9,
            label="skipped",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_DEBUG_OUTCOME_COLORS["corrected"],
            markersize=9,
            label="corrected",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_DEBUG_OUTCOME_COLORS["done_as_is"],
            markersize=9,
            label="done_as_is",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_DEBUG_OUTCOME_COLORS["no_movement"],
            markersize=9,
            label="no_movement",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_DEBUG_OUTCOME_COLORS["no_need_to_move"],
            markersize=9,
            label="no_need_to_move",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_DEBUG_OUTCOME_COLORS["corrected_through_vigilance"],
            markersize=9,
            label="corrected_through_vigilance",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_DEBUG_OUTCOME_COLORS["other"],
            markersize=9,
            label="other",
        ),
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#333333",
            linestyle="none",
            markersize=8,
            label="per_day (o)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="#333333",
            linestyle="none",
            markersize=8,
            label="actionwise eval-time (^)",
        ),
    ]
    tensor_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="white", markeredgecolor=_DEBUG_TENSOR_COLORS[name],
            markeredgewidth=1.6, markersize=10,
            label=f"edge=1step / inner=anyFuture: {name}",
        )
        for name in (
            "precision_correct",
            "precision_wrong_destination",
            "precision_wrong_object",
            "moved_missed",
            "no_event",
            "unknown",
        )
    ]
    ax.legend(
        handles=source_handles + outcome_handles + tensor_handles,
        fontsize=8,
        loc="upper right",
        framealpha=0.9,
        ncol=2,
    )
    ax.grid(axis="x", alpha=0.25)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_prediction_heatmap(
    result_data: dict,
    day_indices: list[int],
    out_dir: str,
    household: str = "",
    condition: str = "",
    ref_result: dict | None = None,
    ref_label: str = "ref",
    path_suffix: str = "",
) -> str:
    """Render SLaTe-PRO prediction heatmap plus per-day debug reconciliation PNGs."""
    heatmap_path = _plot_prediction_heatmap(
        result_data,
        day_indices,
        out_dir,
        household=household,
        condition=condition,
        ref_result=ref_result,
        ref_label=ref_label,
        path_suffix=path_suffix,
    )

    # Emit debug plots next to the heatmap.
    try:
        if heatmap_path:
            per_day = result_data.get("per_day") or []
            aed = result_data.get("actionwise_evals_daily") or []
            base = os.path.basename(heatmap_path)
            debug_prefix = f"debug_{base}"
            for di in day_indices:
                if di < 0:
                    continue
                pde = per_day[di] if di < len(per_day) else {}
                awd = aed[di] if di < len(aed) else {}
                day_num = pde.get("day", di)
                script = pde.get("script", "")
                title = f"{debug_prefix} — day {day_num} {script}".strip()
                out_path = os.path.join(out_dir, f"{debug_prefix}_day{int(day_num):03d}.png")
                _debug_plot_day_actions_compare(
                    per_day_entry=pde,
                    day_actionwise=awd,
                    out_path=out_path,
                    title=title,
                    y_gap=0.28,
                    max_objects=25,
                )
    except Exception as e:
        print(f"[WARNING] Failed to write debug_* plots: {e}")

    return heatmap_path


def get_slate_positions_world(world_state, policy):
    """Return {slate_obj_idx: slate_loc_idx} from the live graph (same as SLaTe-PRO)."""
    from agents.slatepro_utils import get_object_location_matrix
    sorted_vh_ids = policy.common_data['node_ids']
    mapping = {vh_id: local_i for local_i, vh_id in enumerate(sorted_vh_ids)}
    n = len(mapping)
    edges = get_object_location_matrix(
        world_state, mapping, n, policy.common_data)
    locs = edges.argmax(dim=-1)
    return {i: locs[i].item() for i in range(n)}


def _vh_parent_locations_from_edges(world_state: dict) -> dict[int, int]:
    """Object / node from_id -> to_id for INSIDE/ON edges (VirtualHome graph dict)."""
    parents: dict[int, int] = {}
    for e in world_state.get('edges', []):
        if e['relation_type'] in ('INSIDE', 'ON'):
            parents[e['from_id']] = e['to_id']
    return parents


def build_world_state_snapshot(world_state: dict, robot_policy, time_minutes: int) -> dict:
    """Compact JSON-serializable real state after each timestep (robot + human).

    * slate_obj_to_loc: row in SLaTe matrix -> parent column (aligned across runs).
    * vh_obj_to_loc: VirtualHome object id -> parent location id for nodes in the
      policy's vh_id_to_slate_idx map (raw graph edges).
    """
    slate: dict[str, int] = {}
    vh_subset: dict[str, int] = {}
    if hasattr(robot_policy, 'history') and robot_policy.history is not None:
        slate_pos = get_slate_positions_world(world_state, robot_policy)
        slate = {str(k): int(v) for k, v in slate_pos.items()}
        mapping = robot_policy.history.get('vh_id_to_slate_idx') or {}
        parents = _vh_parent_locations_from_edges(world_state)
        for vh_id in mapping:
            if vh_id in parents:
                vh_subset[str(vh_id)] = int(parents[vh_id])
    return {
        'time_minutes': int(time_minutes),
        'slate_obj_to_loc': slate,
        'vh_obj_to_loc': vh_subset,
    }

def _dump_mismatch(name, actual, expected, current_time, idx, out_dump_dir, reason):
    """Save a history-vs-offline tensor mismatch to ``.pt`` for offline inspection."""
    os.makedirs(out_dump_dir, exist_ok=True)
    fname = os.path.join(out_dump_dir,
                         f'history_vs_offline_{name}_t{int(current_time)}_idx{idx}_{reason}.pt')
    torch.save({'actual': actual, 'expected': expected,
                'current_time': current_time, 'idx': idx,
                'reason': reason, 'channel': name}, fname)
    print(f"[DUMP] {fname}")


def _compare_history_to_offline(history, offline_data, current_time, out_dump_dir):
    """Compare model-input tensors (in `history`) to offline_data at the
    matching time index range. Asserts + dumps on mismatch."""
    offline_times = offline_data['times']
    match = (offline_times == float(current_time)).nonzero(as_tuple=False)
    if match.numel() == 0:
        return
    idx = int(match[0, 0].item())
    T = history['edges'].size(1)
    start = idx - (T - 1)
    if start < 0:
        return  # not enough offline frames before this point yet

    n_classes    = history['node_features'].size(-1)
    n_activities = history['activity_features'].size(-1)
    c_len        = history['time_features'].size(-1)

    # Build expected time_features same way training/batch eval does:
    # time_sine_informed at frequencies [10,30,60,180,360,720,1440] then
    # right-pad with zeros to c_len.
    _T_FREQ = torch.tensor([10, 30, 60, 60*3, 60*6, 60*12, 60*24], dtype=torch.float32)
    def _expected_time_features(times_slice):
        omegas = (2.0 * 3.141592653589793 / _T_FREQ).to(times_slice)
        ts = times_slice.float().unsqueeze(-1) * omegas  # (T, 7)
        feat = torch.cat([torch.sin(ts), torch.cos(ts)], dim=-1)  # (T, 14)
        if feat.size(-1) < c_len:
            pad = torch.zeros(feat.size(0), c_len - feat.size(-1),
                              device=feat.device, dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=-1)
        return feat

    # Some channels grow along T (per-timestep), others are stored as a single
    # broadcast frame. Build expected to match whatever history's actual shape is.
    def _match_T(channel_name, per_t_builder, single_builder):
        actual = history[channel_name][0].cpu().float()  # drop batch dim
        # If first dim equals T (per-timestep), build per-timestep; else single frame.
        if actual.shape[0] == T:
            return actual, per_t_builder()
        else:
            return actual, single_builder()

    checks = {
        'edges': _match_T('edges',
                           lambda: offline_data['edges'][start:idx+1].float(),
                           lambda: offline_data['edges'][idx:idx+1].float()),
        'dynamic_edges_mask': _match_T('dynamic_edges_mask',
                           lambda: offline_data['active_edges'].unsqueeze(0).expand(T, -1, -1).float(),
                           lambda: offline_data['active_edges'].unsqueeze(0).float()),
        'activity_features': _match_T('activity_features',
                           lambda: torch.nn.functional.one_hot(
                               offline_data['activity'][start:idx+1].long(),
                               num_classes=n_activities).float(),
                           lambda: torch.nn.functional.one_hot(
                               offline_data['activity'][idx:idx+1].long(),
                               num_classes=n_activities).float()),
        'node_features': _match_T('node_features',
                           lambda: torch.nn.functional.one_hot(
                               offline_data['nodes'].long(), num_classes=n_classes
                           ).float().unsqueeze(0).expand(T, -1, -1),
                           lambda: torch.nn.functional.one_hot(
                               offline_data['nodes'].long(), num_classes=n_classes
                           ).float().unsqueeze(0)),
        'time_features': _match_T('time_features',
                           lambda: _expected_time_features(offline_data['times'][start:idx+1]),
                           lambda: _expected_time_features(offline_data['times'][idx:idx+1])),
    }

    for name, (actual, exp_cpu) in checks.items():
        if actual.shape != exp_cpu.shape:
            _dump_mismatch(name, actual, exp_cpu, current_time, idx, out_dump_dir,
                           reason='shape_mismatch')
            print(
                f"[WARN][history vs offline] shape mismatch in '{name}': "
                f"actual={tuple(actual.shape)} expected={tuple(exp_cpu.shape)}")
            continue
        if not torch.allclose(actual, exp_cpu, atol=1e-3):
            diff = (actual - exp_cpu).abs()
            _dump_mismatch(name, actual, exp_cpu, current_time, idx, out_dump_dir,
                           reason='value_mismatch')
            print(
                f"[WARN][history vs offline] value mismatch in '{name}' at "
                f"current_time={current_time} idx={idx}: "
                f"max_abs_diff={float(diff.max()):.4g}  "
                f"n_diff={int((diff>1e-5).sum())}/{diff.numel()}")


def evaluate_closed_loop(robot_policy,
                          sim_config: dict,
                          script_files: list[str],
                          script_distribution: list[int] = [0,-1,0],    ## warm-start, finetune, test
                          suppress_robot_action: bool = False,
                          active_start_minutes: int = 0,
                          active_end_minutes: int = 1440,
                          train_every_n_days: int = 1,
                          reset_env: bool = True,
                          out_path: str = None,
                          use_prior_checkpoints: bool = False) -> dict:
    """
    Run World + HumanAgent closed-loop for each script and collect metrics.

    TL;DR: three phases (warm-start / finetune / test) over *script_files*;
    robot acts then human each tick; optional periodic ``robot_policy.train``.

    Parameters
    ----------
    robot_policy : RobotPolicyBase subclass (already constructed).
    sim_config : Full simulation config dict.
    script_files : Ordered list of script ``.txt`` paths (length must match
        ``sum(script_distribution)``; one entry may be ``-1`` = fill remainder).
    script_distribution : ``[warmup_days, finetune_days_or_-1, test_days]``.
    suppress_robot_action : If True, robot predictions are logged but not executed.
    reset_env : If True, ``World.reset()`` at each day boundary.
    out_path : Incrementally written benchmark JSON (optional).
    use_prior_checkpoints : Skip warm-start; load existing ``weights_day*.pt``.

    Returns
    -------
    dict with ``per_day``, ``totals`` (via ``_aggregate_totals``),
    ``actionwise_evals_daily``, ``training_dataset_per_day``.
    """
    from agents.robot_policy import RobotMoveAction, RobotPolicyBase
    from environment.human_agent import HumanAgent
    from environment.world import World
    from evolving_graph.environment import EnvironmentGraph
    from evolving_graph.execution import ScriptExecutor

    if not script_files:
        return {'per_day': [], 'totals': {}}

    max_time  = sim_config['simulation']['max_time']
    time_step = sim_config['simulation']['time_step_minutes']

    per_day: list[dict] = []
    actionwise_evals_daily = []
    training_dataset_per_day = {}

    if any(s == -1 for s in script_distribution):
        num_taken = sum([s for s in script_distribution if s != -1])
        assert script_distribution.count(-1) == 1, "Only one script distribution can be -1"
        script_distribution[script_distribution.index(-1)] = len(script_files) - num_taken

    assert sum(script_distribution) == len(script_files), "The sum of the script distribution must be equal to the number of scripts"

    phase = 'warm-start' if script_distribution[0] > 0 else 'finetune' if script_distribution[1] > 0 else 'test'
    phase_ends = {p:sum(script_distribution[:i+1]) for i, p in enumerate(['warm-start', 'finetune', 'test'])}
    # Phase semantics (see inline loop):
    #   warm-start — history only, no robot actions, builds gen-dataset .pt files
    #   finetune   — robot acts; human accommodates; train every N days
    #   test       — robot acts; no further training
    cfg   = copy.deepcopy(sim_config)
    store_world_trace = cfg.get('evaluation', {}).get('store_world_state_timeline', True)

    script_path = script_files[0]
    init_edges = torch.load(
        homers_script_path_to_processed_pt(script_path), map_location='cpu')['edges'][0]
    init_graph_dict = json.load(open(script_path.replace('/scripts_','/routines_').replace('.txt','.json')))['graphs'][0]
    world = World(cfg, init_graph_dict)

    # ── Dataset directory timestamp (must be consistent across resumes) ──
    timestr = time.strftime("%m%d_%H%M")
    timestr = timestr[:-1] + '0'

    # ── Resume from checkpoint if available ──────────────────────────────
    start_day_idx = 0
    # Per-household resume path (derived from out_path which includes the household)
    original_out_stem = os.path.basename(out_path).replace('.json', '') if out_path else None

    prep = prepare_resume_and_paths(
        out_path=out_path,
        use_prior_checkpoints=use_prior_checkpoints,
        phase=phase,
        phase_ends=phase_ends,
        script_distribution=script_distribution,
        per_day=per_day,
        actionwise_evals_daily=actionwise_evals_daily,
        start_day_idx=start_day_idx,
    )
    out_stem = prep.out_stem
    resume_path = prep.resume_path
    start_day_idx = prep.start_day_idx
    phase = prep.phase
    per_day = prep.per_day
    actionwise_evals_daily = prep.actionwise_evals_daily

    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        start_day_idx = ckpt['day_idx'] + 1
        if ckpt.get('perday_jsonl'):
            per_day, actionwise_evals_daily, training_dataset_per_day = _read_perday_jsonl(
                out_path, ckpt['day_idx'])
            print(f"Replayed {len(per_day)} days from "
                  f"{os.path.basename(_perday_jsonl_path(out_path))}")
        else:
            # Checkpoint written before the sidecar existed: it still embeds the
            # per-day records. Seed the sidecar from it so the next resume -- and
            # the final result json -- have a complete history either way.
            per_day = ckpt['per_day']
            actionwise_evals_daily = ckpt['actionwise_evals_daily']
            training_dataset_per_day = ckpt['training_dataset_per_day']
            print(f"Resumed from a pre-sidecar checkpoint ({len(per_day)} days embedded); "
                  f"seeding {os.path.basename(_perday_jsonl_path(out_path))}")
            if not os.path.isfile(_perday_jsonl_path(out_path)):
                for i, rec in enumerate(per_day):
                    _append_perday(out_path, rec.get('day', i), rec,
                                   actionwise_evals_daily[i] if i < len(actionwise_evals_daily) else None,
                                   training_dataset_per_day.get(rec.get('day', i)))
        phase = ckpt['phase']
        # Restore dataset directory timestamp so resumed runs use the same path
        timestr = ckpt.get('timestr', timestr)
        # Restore world state if not resetting
        if not reset_env and 'world_state_dict' in ckpt:
            world.env = EnvironmentGraph(ckpt['world_state_dict'])
            world.executor = ScriptExecutor(world.env, world.name_equivalence)
        # Restore model weights if training has happened
        if ckpt.get('model_weights_path') and os.path.isfile(ckpt['model_weights_path']):
            if isinstance(robot_policy, RobotPolicySlatepro):
                robot_policy.model.load_state_dict(
                    torch.load(ckpt['model_weights_path'], map_location='cpu'))
                robot_policy._setup_model_for_inference()
            elif isinstance(robot_policy, RobotPolicyStreak):
                robot_policy.load_artifacts(ckpt['model_weights_path'])
            elif isinstance(robot_policy, (RobotPolicyHybrid,
                                           RobotPolicyPlainLLM,
                                           RobotPolicyOracle,
                                           RobotPolicyNOPA)):
                robot_policy.load_artifacts(ckpt['model_weights_path'])
        last_model_weights_path = ckpt.get('model_weights_path')
        print(f"Resumed from day {start_day_idx} (phase={phase}, timestr={timestr})")
    else:
        last_model_weights_path = None
    heatmap_suffix = ''

    # When use_prior_checkpoints skips warm-start, the in-loop checkpoint-load
    # block (only fires at train_every_n_days boundaries inside finetune) never
    # reaches the last warm-start day. Load those weights once up front so the
    # robot has a trained model before day_idx==start_day_idx starts.
    if use_prior_checkpoints and out_path is not None and start_day_idx > 0:
        ckpt_stem = original_out_stem
        model_ckpt_dir = os.path.join(os.path.dirname(out_path), f'model_checkpoints_{ckpt_stem}')
        warm_end_day = start_day_idx - 1
        weights_path = os.path.join(model_ckpt_dir, f'weights_day{warm_end_day}.pt')
        loaded_path = None
        if isinstance(robot_policy, RobotPolicySlatepro):
            if os.path.isfile(weights_path):
                print(f"[use_prior_checkpoints] Loading warm-start weights: {weights_path}")
                robot_policy.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                loaded_path = weights_path
            else:
                # Fallback: PyTorch Lightning .ckpt under pl_checkpoints_day{warm_end_day}/.
                # Prefer the highest-epoch 'epoch=*.ckpt' file (the saved best checkpoint
                # from the last training round); fall back to 'last.ckpt' if none exists.
                pl_dir = os.path.join(model_ckpt_dir, f'pl_checkpoints_day{warm_end_day}')
                pl_ckpt = None
                if os.path.isdir(pl_dir):
                    epoch_cands = sorted([f for f in os.listdir(pl_dir)
                                          if f.startswith('epoch') and f.endswith('.ckpt')],
                                         reverse=True)
                    if epoch_cands:
                        pl_ckpt = os.path.join(pl_dir, epoch_cands[0])
                    elif os.path.isfile(os.path.join(pl_dir, 'last.ckpt')):
                        pl_ckpt = os.path.join(pl_dir, 'last.ckpt')
                if pl_ckpt is not None:
                    print(f"[use_prior_checkpoints] Falling back to PL ckpt: {pl_ckpt}")
                    ckpt_data = torch.load(pl_ckpt, map_location='cpu')
                    state_dict = ckpt_data.get('state_dict', ckpt_data)
                    robot_policy.model.load_state_dict(state_dict)
                    loaded_path = pl_ckpt
                else:
                    raise RuntimeError(f"[WARNING] use_prior_checkpoints set but weights missing at "
                          f"{weights_path} and no PL ckpt at {pl_dir}")
            if loaded_path is not None:
                robot_policy._setup_model_for_inference()
                last_model_weights_path = loaded_path
        elif isinstance(robot_policy, RobotPolicyStreak):
            if not os.path.isfile(weights_path):
                raise RuntimeError(
                    f"use_prior_checkpoints set but STREAK weights missing at {weights_path}")
            print(f"[use_prior_checkpoints] Loading STREAK weights: {weights_path}")
            robot_policy.load_artifacts(weights_path)
            last_model_weights_path = weights_path
        elif isinstance(robot_policy, (RobotPolicyHybrid, RobotPolicyPlainLLM,
                                       RobotPolicyOracle, RobotPolicyNOPA)):
            if not os.path.isfile(weights_path):
                raise RuntimeError(f"use_prior_checkpoints set but policy artifacts "
                                   f"missing at {weights_path}")
            print(f"[use_prior_checkpoints] Loading policy artifacts: {weights_path}")
            robot_policy.load_artifacts(weights_path)
            last_model_weights_path = weights_path

    dataset_dir = None

    for day_idx, script_path in enumerate(script_files):
        if day_idx < start_day_idx:
            continue
        if reset_env:
            # Reuse day-0 init graph (set at harness startup). Per-day
            # routines_*.json ``graphs[0]`` are identical on HOMER+ — the first
            # state at the start of any day is the same morning layout.
            world.reset()
        if day_idx >= phase_ends[phase]:
            phase = 'finetune' if phase == 'warm-start' else 'test'

        print(f"[{phase}] Day {day_idx}: {script_path}")
        
        offline_data_dirpath = script_path.replace('scripts_','offline_data/').replace('.txt','.pt')
        print(f"\n\n\nStarting day {day_idx} using human agent from \n{script_path}")
        ## if robot action is suppressed, don't accommodate for it
        from simulation import build_human_agent
        human = build_human_agent(cfg, world.node_map, script_path,
                                   accommodate_for_robot=(not suppress_robot_action))
        # Vigilance only active during finetune/test, not warm-start
        if phase == 'warm-start':
            human.vigilance_enabled = False
        # LLM-human is also inactive during warm-start (scripted decisions only)
        if hasattr(human, 'llm_active'):
            human.llm_active = (phase != 'warm-start')
        # Point the LLM audit log next to the benchmark JSON (one file per run).
        if hasattr(human, 'set_audit_log') and out_path is not None:
            _stem = os.path.basename(out_path).replace(".json", "")
            human.set_audit_log(os.path.join(
                os.path.dirname(out_path), f'llm_audit_{_stem}.txt'))
        # Per-run budget knob, editable live to raise/lower the per-day USD cap.
        if hasattr(human, 'set_budget_file') and out_path is not None:
            _stem = os.path.basename(out_path).replace(".json", "")
            human.set_budget_file(os.path.join(
                os.path.dirname(out_path), f'llm_budget_usd_{_stem}.txt'))
        # Reset per-day LLM bookkeeping (no-op for scripted human)
        if hasattr(human, 'reset_day'):
            human.reset_day(day_tag=f'day{day_idx}')

        # Clear stale per-episode refs on the policy
        if hasattr(robot_policy, 'reset'):
            # Pass the day's GT script to any policy whose reset accepts it
            # (the P0 oracle always; NOPA only for goal_source='oracle').
            import inspect as _inspect
            if 'script_path' in _inspect.signature(robot_policy.reset).parameters:
                robot_policy.reset(script_path=script_path)
            else:
                robot_policy.reset()

        result_metrics = {
            'skipped': 0,
            'corrected': 0,
            'unseen': 0,
            'vigilance_corrections': 0,
            # LLM-human outcomes a robot action can be tagged with (see
            # environment/llm_human_agent.py). Present so the increment at the
            # pending-action reconciliation below never KeyErrors.
            'accepted': 0,
            'corrected_despite_correct': 0,
            'redirected': 0,
            'done_early': 0,
        }
        if suppress_robot_action:
            # Paper-style metrics (1-step lookahead) vs offline GT edges.
            result_metrics.update({
            'precision_correct': 0,
            'precision_wrong_destination': 0,
            'precision_wrong_object': 0,
            'precision_total': 0,
            'moved_missed': 0,
            'moved_total': 0,
            'unmoved_fp': 0,
            'unmoved_total': 0,
            })
        actionwise_evals = {}
        pending_robot_actions_evaluation = []
        unmatched_tensor_events_today: list[dict] = []

        current_time  = active_start_minutes
        time_step_idx = 0
        world.current_time = current_time
        robot_actions = 0

        # Accumulators for raw HOMER_PLUS-format dataset generation
        raw_edges_list = []       # list of (N, N) binary edge matrices
        raw_movements_list_human = []
        raw_movements_list_robot = []
        raw_activity_list = []    # list of int activity indices
        raw_times_list = []       # list of float time values
        raw_nodes = None          # (N,) set once from first timestep
        raw_active_edges = None   # (N, N) set once from first timestep
        raw_class_names = None
        raw_node_ids = None
        
        world_state_prev_timestep = None
        day_world_trace: list[dict] = []

        if os.path.isfile(offline_data_dirpath):
            offline_data = torch.load(offline_data_dirpath)
            offline_edges = offline_data['edges']
            offline_activity = offline_data['activity']
            offline_nodes = offline_data['nodes']
            offline_active_edges = offline_data['active_edges']
            offline_class_names = offline_data['class_names']
            offline_node_ids = offline_data['node_ids']
            offline_times = offline_data['times']
            # Wire the policy's in-situ input probe. Only valid when robot
            # actions are suppressed AND we're in a fresh run — otherwise the
            # live world diverges from the offline trajectory by design (or by
            # offline-data regeneration) and the probe asserts spuriously.
            if (isinstance(robot_policy, RobotPolicySlatepro)
                    and suppress_robot_action
                    and not use_prior_checkpoints):
                robot_policy._debug_offline_data = offline_data
                # One-shot dump on the very first eval-phase call.
                if phase == 'finetune' and not hasattr(robot_policy, '_dumped_run_forward'):
                    dump_dir = os.path.dirname(out_path) if out_path else '.'
                    robot_policy._debug_dump_path = os.path.join(
                        dump_dir, f'live_run_forward_dump_{sim_config["benchmark"]["household"]}.pt')
        else:
            if suppress_robot_action:
                print(f"[WARNING] offline_data file missing: {offline_data_dirpath} — "
                      f"paper-style metric block will be skipped for day {day_idx}.")
            offline_edges = None
            offline_times = None
        

        while world.current_time < min(max_time, active_end_minutes):
            print(f"Current time: {world.current_time}")

            ## Print graph difference from last timestep
            if world_state_prev_timestep is not None:
                print_graph_difference(world_state_prev_timestep, world.get_state())

            world_state_prev_timestep = world.get_state()

            # TEMP: world_state vs offline divergence check (activity + raw time).
            # The history-vs-offline comparator below already covers edges; this
            # catches drifts the history layer might mask.
            # Only assert in fresh runs — with use_prior_checkpoints the offline
            # trajectory may have been regenerated since the checkpoints were
            # trained, in which case live/offline divergence is expected.
            if (offline_edges is not None
                    and isinstance(robot_policy, RobotPolicySlatepro)
                    and not use_prior_checkpoints):
                _ws = world.get_state()
                _match = (offline_times == float(current_time)).nonzero(as_tuple=False)
                if _match.numel() > 0:
                    _idx = int(_match[0, 0].item())
                    _activities = robot_policy.common_data['activities']
                    _ws_act_idx = (_activities.index(_ws['activity'])
                                   if _ws['activity'] in _activities else -1)
                    _off_act_idx = int(offline_activity[_idx].item())
                    if _ws_act_idx != _off_act_idx:
                        print(
                            f"[WARN] world_state activity mismatch at t={current_time} idx={_idx}: "
                            f"ws='{_ws['activity']}' (idx={_ws_act_idx}) vs offline_idx={_off_act_idx}")
                    if int(_ws['time']) != int(offline_times[_idx].item()):
                        print(
                            f"[WARN] world_state time mismatch at t={current_time}: "
                            f"ws={_ws['time']} vs offline={offline_times[_idx].item()}")

            pos_initial = get_slate_positions_world(world.get_state(), robot_policy)

            ### ROBOT ACTIONS ###
            robot_actions_tensor = None
            if phase != 'warm-start':
                actions_robot, movement_indices = robot_policy.get_actions(world.node_map, world_state=world.get_state())
                # TEMP: per-step input-tensor comparison vs offline_data.
                # Only meaningful when the robot does not act (live trajectory
                # tracks offline) and we're in a fresh run (offline data was
                # generated by the *current* code path, not stale from before).
                # With suppress_robot_action=False the world legitimately
                # diverges; with use_prior_checkpoints the offline trajectory
                # may post-date the checkpoints and disagree.
                if (isinstance(robot_policy, RobotPolicySlatepro)
                        and offline_edges is not None
                        and suppress_robot_action
                        and not use_prior_checkpoints):
                    _compare_history_to_offline(
                        robot_policy.history, offline_data, current_time,
                        out_dump_dir=os.path.dirname(out_path) if out_path else '.')
            else:
                # Warm-up: skip model inference, only update history for dataset generation
                robot_policy._update_history(world.get_state())
                actions_robot, movement_indices = [], ([], [], [])
            obj_idxs, loc_idxs, prev_loc_idxs = movement_indices

            # Paper-style 1-step metrics + tensor reconciliation: only when robot
            # actions are suppressed so live state tracks offline GT (cond1-style).
            fwd = getattr(robot_policy, '_last_fwd', None)
            per_obj_tensor_category = (
                _classify_per_object_tensor(fwd, offline_edges, offline_times, current_time)
                if (phase != 'warm-start'
                    and suppress_robot_action
                    and fwd is not None
                    and offline_edges is not None)
                else {}
            )
            vh_id_to_slate_idx = (
                robot_policy.history.get('vh_id_to_slate_idx', {})
                if getattr(robot_policy, 'history', None) is not None else {}
            )

            for action_robot, obj_idx, loc_idx, prev_loc_idx in zip(actions_robot, obj_idxs, loc_idxs, prev_loc_idxs):
                print(f"Robot action: {action_robot}")
                # Remove past movement (if any) of the same object from pending list and add to actionwise evals as undone movement
                if action_robot.object_id in [a['obj_idx'] for a in pending_robot_actions_evaluation]:
                    datapoint = [a for a in pending_robot_actions_evaluation if a['obj_idx'] == action_robot.object_id][0]
                    datapoint['human_action_outcome'] = 'unseen'
                    datapoint['eval_time'] = current_time
                    actionwise_evals.setdefault(datapoint['time'], []).append(datapoint)
                    result_metrics['unseen'] += 1
                    pending_robot_actions_evaluation.remove(datapoint)
                
                ## Add new robot action to pending list
                graph_delta = torch.zeros_like(robot_policy.history['edges'][0, -1])
                graph_delta[obj_idx, loc_idx] = 1
                robot_actions_tensor = graph_delta.to(torch.int64) if robot_actions_tensor is None else robot_actions_tensor + graph_delta.to(torch.int64)
                _cats = per_obj_tensor_category.get(int(obj_idx)) or {'1step': 'no_event', 'anyFuture': 'no_event'}
                pending_robot_actions_evaluation.append({
                            'obj_idx': action_robot.object_id,
                            'object_name': action_robot.object_name,
                            'slate_obj_idx': int(obj_idx),
                            'loc_idx': action_robot.dest_id,
                            'time': current_time,
                            'human_action_outcome': None,
                            'tensor_eval_category_1step': _cats['1step'],
                            'tensor_eval_category_anyFuture': _cats['anyFuture'],
                            'graph_delta': graph_delta.to(torch.int64)})
            
                ## Decide whether to undo this robot action (legacy vigilance
                ## or LLM-driven, dispatched via human.review_robot_action).
                vigilance_correction_initiated = False
                world_state_now = world.get_state() if hasattr(world, 'get_state') else {}
                review_decision, review_record = human.review_robot_action(
                    action_robot, world_state_now, current_time,
                    day_history=None)
                if review_decision == 'undo':
                    vigilance_correction_initiated = True
                    human.vigilance_corrections += 1
                    result_metrics['vigilance_corrections'] += 1
                    pending_robot_actions_evaluation[-1]['human_action_outcome'] = 'corrected_through_vigilance'
                    pending_robot_actions_evaluation[-1]['eval_time'] = current_time
                    actionwise_evals.setdefault(current_time, []).append(pending_robot_actions_evaluation[-1])
                    pending_robot_actions_evaluation.remove(pending_robot_actions_evaluation[-1])
                    print(f"  [REVIEW] Undid robot action: {action_robot.object_name} -> {action_robot.dest_id}")
                elif review_decision == 'redirect':
                    # v2: suppress the robot's (wrong) move, and have the human
                    # move the object straight to its next planned destination
                    # now (one trip). The later script line for that obj->dest is
                    # tagged 'done_early' (ignored) by the human agent.
                    vigilance_correction_initiated = True
                    tgt = (review_record or {}).get('redirect_target') or {}
                    tgt_name, tgt_id = tgt.get('name'), tgt.get('id')
                    from agents.robot_policy import RobotMoveAction
                    redirect_move = RobotMoveAction(
                        action_robot.object_name, action_robot.object_id,
                        tgt_name, tgt_id)
                    rr, _ = world.execute_action(redirect_move.script_action())
                    if rr is None:
                        # Redirect failed (dest unreachable). The robot's move is
                        # already suppressed, so count this as a vigilance undo
                        # rather than a (non-existent) successful redirect — don't
                        # credit a save that never happened.
                        print(f"[WARN] Redirect move failed at t={current_time}: "
                              f"{action_robot.object_name} -> {tgt_name}; counting as vigilance correction.")
                        result_metrics['vigilance_corrections'] += 1
                        human.vigilance_corrections += 1
                        pending_robot_actions_evaluation[-1]['human_action_outcome'] = 'corrected_through_vigilance'
                    else:
                        result_metrics['redirected'] += 1
                        pending_robot_actions_evaluation[-1]['human_action_outcome'] = 'redirected'
                    pending_robot_actions_evaluation[-1]['eval_time'] = current_time
                    actionwise_evals.setdefault(current_time, []).append(pending_robot_actions_evaluation[-1])
                    pending_robot_actions_evaluation.remove(pending_robot_actions_evaluation[-1])
                    print(f"  [REVIEW] Redirected: {action_robot.object_name} -> {tgt_name} (was -> {action_robot.dest_id})")

                robot_actions += 1
                if not suppress_robot_action and not vigilance_correction_initiated and not phase == 'warm-start':
                    script_action = action_robot.script_action()
                    result, _ = world.execute_action(script_action)
                    if result is None:
                        print(
                            f"[WARN] Robot action failed at t={current_time}: {script_action}\n"
                            f"  Error: {world.executor.info.get_error_string()}\n"
                            f"  Dropping action from pending eval list; world state unchanged."
                        )
                        result_metrics.setdefault('failed_action', 0)
                        result_metrics['failed_action'] += 1
                        if pending_robot_actions_evaluation and pending_robot_actions_evaluation[-1]['obj_idx'] == action_robot.object_id:
                            pending_robot_actions_evaluation.pop()

            ### HUMAN ACTIONS ###
            # Capture positions before human acts for movement diff
            pos_before = get_slate_positions_world(world.get_state(), robot_policy)
            # Determine human actions and set activity for this timestep
            n_outcomes_before = len(human.action_outcomes)
            actions_human = human.get_actions(
                current_time, world.get_state(),
                robot_actions=pending_robot_actions_evaluation,
                per_obj_tensor_category=per_obj_tensor_category,
                vh_id_to_slate_idx=vh_id_to_slate_idx,
            )
            world.set_human_activity(human.get_human_activity())
            for action in actions_human:
                print(f"Human action: {action}")
                result, _ = world.execute_action(action)
                if result is None:
                    err = world.executor.info.get_error_string()
                    # A human teleport that would create a cycle is skipped, not
                    # fatal: the robot can stack an object such that the human's
                    # otherwise-normal move would close a loop in the world tree
                    # (e.g. robot dumps onto <toothbrush>, human moves
                    # <toothbrush> -> <toothbrush_holder>). The human must not be
                    # able to crash the world; skip it (mirrors the robot-action
                    # path above). The containment-ban mask only blocks the common
                    # DIRECT inversions, so transitive/magnet-dump cycles still
                    # reach here. Any OTHER human failure still raises (no silent
                    # fallback for genuine bugs).
                    if 'create cycle' in err or 'ancestor of' in err:
                        print(
                            f"[WARN] Human action would create a cycle at t={current_time}: {action}\n"
                            f"  Error: {err}\n"
                            f"  Skipping action; world state unchanged."
                        )
                        result_metrics.setdefault('human_cycle_skipped', 0)
                        result_metrics['human_cycle_skipped'] += 1
                        continue
                    raise RuntimeError(
                        f"Human action failed at t={current_time}: {action}\n"
                        f"  Error: {err}"
                    )

            # Compute pos_delta_human after vigilance corrections so the
            # HOMER-format dataset sees the corrected world state.
            def tensor_diff(pos_after, pos_before):
                keys_a = list(pos_after.keys())
                vals_a = list(pos_after.values())
                keys_b = list(pos_before.keys())
                vals_b = list(pos_before.values())

                n = max(max(keys_a), max(keys_b)) + 1
                tensorA = torch.zeros((n,n))
                tensorB = torch.zeros((n,n))

                tensorA[keys_a, vals_a] = 1.0
                tensorB[keys_b, vals_b] = 1.0

                diff = tensorA - tensorB
                return diff.to(torch.int64)

            pos_after = get_slate_positions_world(world.get_state(), robot_policy)
            pos_delta_human = tensor_diff(pos_after, pos_before)
            pos_delta_human[pos_delta_human<0] = 0

            if robot_actions_tensor is None:
                robot_actions_tensor = torch.zeros_like(pos_delta_human)

            # Paper-style 1-step metrics: compare model's raw next-step prediction
            # against offline GT edges at idx+1. Mirrors SLaTe-PRO eval at step=0.
            if (suppress_robot_action
                    and phase != 'warm-start'
                    and fwd is not None
                    and offline_edges is not None):
                match = (offline_times == float(current_time)).nonzero(as_tuple=False)
                assert match.numel() > 0, (
                    f"current_time {current_time} not found in offline_times "
                    f"(range {offline_times[0].item()}..{offline_times[-1].item()})")
                idx = int(match[0, 0].item())

                if idx + 1 < offline_edges.shape[0]:
                    current_locs = offline_edges[idx].argmax(-1)
                    gt_locs      = offline_edges[idx + 1].argmax(-1)
                    pred_locs    = fwd['pred_locations']
                    dyn_mask     = fwd['dyn_mask_flat']

                    changes_gt       = (gt_locs   != current_locs) & dyn_mask
                    changes_pred     = (pred_locs != current_locs) & dyn_mask
                    correct          = (pred_locs == gt_locs)
                    used_pred_and_gt = changes_pred & changes_gt
                    used_pred_not_gt = changes_pred & ~changes_gt
                    not_pred_but_gt  = ~changes_pred & changes_gt
                    unmoved_gt       = ~changes_gt & dyn_mask

                    result_metrics['precision_correct']           += int((correct & used_pred_and_gt).sum())
                    result_metrics['precision_wrong_destination'] += int((~correct & used_pred_and_gt).sum())
                    result_metrics['precision_wrong_object']      += int(used_pred_not_gt.sum())
                    result_metrics['precision_total']             += int(changes_pred.sum())
                    result_metrics['moved_missed']                += int(not_pred_but_gt.sum())
                    result_metrics['moved_total']                 += int(changes_gt.sum())
                    result_metrics['unmoved_fp']                  += int((changes_pred & unmoved_gt).sum())
                    result_metrics['unmoved_total']               += int(unmoved_gt.sum())
            
            # Match pending predictions against human movements; remove matched
            # Skip during warm-start (no robot actions, metrics meaningless)
            if phase != 'warm-start':
                for robot_action_datapoint in pending_robot_actions_evaluation:
                    if robot_action_datapoint['human_action_outcome'] == None:
                        continue
                    result_metrics[robot_action_datapoint['human_action_outcome']] += 1
                    actionwise_evals.setdefault(robot_action_datapoint['time'], []).append(robot_action_datapoint)
                pending_robot_actions_evaluation = [a for a in pending_robot_actions_evaluation if a['human_action_outcome'] == None]

            # Reconcile per-iteration tensor events (suppress_robot_action only).
            if (suppress_robot_action
                    and phase != 'warm-start'
                    and per_obj_tensor_category):
                matched_slate_idxs: set[int] = set()
                for a in pending_robot_actions_evaluation:
                    if a.get('time') == current_time and 'slate_obj_idx' in a:
                        matched_slate_idxs.add(int(a['slate_obj_idx']))
                for ev in actionwise_evals.get(current_time, []):
                    if isinstance(ev, dict) and 'slate_obj_idx' in ev:
                        matched_slate_idxs.add(int(ev['slate_obj_idx']))
                for o in human.action_outcomes[n_outcomes_before:]:
                    sidx = o.get('slate_obj_idx')
                    if sidx is not None:
                        matched_slate_idxs.add(int(sidx))
                for sidx, cats in per_obj_tensor_category.items():
                    if cats['1step'] == 'no_event' and cats['anyFuture'] == 'no_event':
                        continue
                    if int(sidx) in matched_slate_idxs:
                        continue
                    unmatched_tensor_events_today.append({
                        'time': current_time,
                        'slate_obj_idx': int(sidx),
                        'tensor_eval_category_1step': cats['1step'],
                        'tensor_eval_category_anyFuture': cats['anyFuture'],
                    })

            # Accumulate raw data for HOMER_PLUS-format dataset
            h = robot_policy.history
            # Edges: binarize the row-normalized edge matrix (argmax → one-hot)
            edge_t = h['edges'][0, -1]  # (N, N)
            binary_edge = torch.zeros_like(edge_t)
            locs = edge_t.argmax(dim=-1)
            binary_edge[torch.arange(edge_t.size(0)), locs] = 1.0
            raw_edges_list.append(binary_edge)
            raw_movements_list_human.append(pos_delta_human)
            raw_movements_list_robot.append(torch.zeros_like(pos_delta_human))
            # Activity: integer index from one-hot
            act_t = h['activity_features'][0, -1]  # (n_activities,)
            raw_activity_list.append(int(act_t.argmax().item()))
            # Time: raw minutes
            assert current_time == float(h['times'][-1]), f"Current time {current_time} does not match history time {float(h['times'][-1])}"
            raw_times_list.append(float(h['times'][-1]))
            # One-time: nodes, active_edges, metadata
            if raw_nodes is None:
                raw_nodes = h['nodes'][0, 0].float()              # (N,) positional indices
                raw_active_edges = h['dynamic_edges_mask'][0, 0]  # (N, N)
                raw_class_names = list(robot_policy.common_data.get('node_classes', []))
                raw_node_ids = list(robot_policy.common_data.get('node_ids', []))

            if store_world_trace:
                snap_t = int(world.current_time)
                day_world_trace.append(
                    build_world_state_snapshot(world.get_state(), robot_policy, snap_t))

            time_step_idx += 1
            current_time       += time_step
            world.current_time  = current_time
        
        ## Bookkeep edges for dataset generation and finish the list
        if not suppress_robot_action:
            # Dataset-noise study (cond13–16): MISSED_GOOD_PERC drops a fraction
            # of skipped events from the training dataset; ADDED_BAD_PERC adds a
            # fraction of non-skipped events. Defaults of 0/0 reproduce the
            # legacy skipped-only behaviour exactly.
            missed_good_perc = float(os.environ.get('MISSED_GOOD_PERC', '0') or 0) / 100.0
            added_bad_perc   = float(os.environ.get('ADDED_BAD_PERC',   '0') or 0) / 100.0
            use_eval_time    = os.environ.get('SKIPPED_AT_EVAL_TIME', '') == '1'

            # Deterministic per-day RNG so resubmits replay identical draws.
            rng_seed = hash((int(day_idx),
                             str(sim_config['benchmark']['household']),
                             os.environ.get('MISSED_GOOD_PERC', '0'),
                             os.environ.get('ADDED_BAD_PERC',   '0'))) & 0xFFFFFFFF
            rng = random.Random(rng_seed)

            for _t, eval_list in actionwise_evals.items():
                for action_eval in eval_list:
                    if not isinstance(action_eval, dict):
                        continue
                    outcome = action_eval.get('human_action_outcome')
                    if outcome is None:
                        action_eval['human_action_outcome'] = 'unseen_at_end_of_day'
                        outcome = 'unseen_at_end_of_day'

                    if outcome == 'skipped':
                        include = (rng.random() >= missed_good_perc)
                    else:
                        include = (rng.random() <  added_bad_perc)
                    if not include:
                        continue

                    action_time = action_eval.get('eval_time') if use_eval_time else action_eval.get('time')
                    action_delta_matrix = action_eval.get('graph_delta')
                    if action_time is not None and action_time in raw_times_list and action_delta_matrix is not None:
                        timestep = raw_times_list.index(action_time)
                        raw_movements_list_robot[timestep] += action_delta_matrix

        if raw_nodes is not None and not use_prior_checkpoints:
            try:
                filtered_edges_list = [raw_edges_list[0]]
                for i in range(1, len(raw_edges_list)):
                    new_edges = filtered_edges_list[i-1].clone()
                    # The movements that turn filtered_edges[i-1] into filtered_edges[i]
                    # happened during iteration i-1 (between snapshots taken at start
                    # of i-1 and start of i), so index by i-1, not i.
                    movements_to_add = torch.zeros_like(new_edges).to(torch.int64)
                    if raw_movements_list_robot[i-1].shape == new_edges.shape:
                        movements_to_add += raw_movements_list_robot[i-1].to(torch.int64)
                    if raw_movements_list_human[i-1].shape == new_edges.shape:
                        movements_to_add += raw_movements_list_human[i-1].to(torch.int64)
                    if movements_to_add.sum() == 0:
                        filtered_edges_list.append(new_edges)
                        continue
                    nz = movements_to_add.nonzero(as_tuple=False)  # (K, 2): [row, col] pairs
                    objects_affected = nz[:, 0].unique()
                    for obj in objects_affected:
                        dest = movements_to_add[obj].nonzero(as_tuple=False).squeeze(-1)
                        new_edges[obj, :] = 0
                        new_edges[obj, dest] = 1
                    filtered_edges_list.append(new_edges)
                
                batch_dict = {
                    'nodes':        raw_nodes,                                            # (N,)
                    'raw_edges':    torch.stack(raw_edges_list, dim=0),                   # (T, N, N)
                    'edges':        torch.stack(filtered_edges_list, dim=0),              # (T, N, N)
                    'active_edges': raw_active_edges,                                     # (N, N)
                    'times':        torch.tensor(raw_times_list, dtype=torch.float32),    # (T,)
                    'activity':     torch.tensor(raw_activity_list, dtype=torch.int64),   # (T,)
                    'class_names':  raw_class_names,                                      # list[str]
                    'node_ids':     raw_node_ids,                                         # list[int]
                }
                vig_cfg = sim_config.get('human', {}).get('curiosity_vigilance', {}) or {}
                if vig_cfg.get('enabled', False):
                    vig_tag = f"_vig{vig_cfg.get('probability', 1.0)}w{vig_cfg.get('correction_window_hours', 6.0)}"
                else:
                    vig_tag = "_vigoff"
                # Mark datasets generated under the SKIPPED_AT_EVAL_TIME ablation
                # or the missed-good/added-bad noise study so they're
                # distinguishable from vanilla cond4-style datasets.
                ablation_tag = '_skipAtEval' if os.environ.get('SKIPPED_AT_EVAL_TIME', '') == '1' else ''
                _mg = os.environ.get('MISSED_GOOD_PERC', '') or ''
                _ab = os.environ.get('ADDED_BAD_PERC', '') or ''
                noise_tag = (f"_missedGood{_mg}" if _mg else '') + (f"_addedBad{_ab}" if _ab else '')
                # Mark datasets generated under a non-scripted human so an
                # LLM-human run (cond19/cond20) is never confused with the
                # scripted cond5/cond4 datasets it otherwise mirrors.
                _human_policy = sim_config.get('human', {}).get('policy', 'scripted')
                if _human_policy != 'scripted':
                    _llm = (sim_config.get('human', {}).get('llm', {}) or {})
                    _llm_model = _llm.get('model', '')
                    _obs_mode = _llm.get('observe_mode', 'undo_allow')
                    human_tag = (f"_human{_human_policy}"
                                 + (f"-{_llm_model}" if _llm_model else "")
                                 # distinguish v2 'redirect' datasets from v1
                                 + (f"-{_obs_mode}" if _obs_mode and _obs_mode != 'undo_allow' else ""))
                else:
                    human_tag = ""
                # noRep0: legacy tag segment (repetition suppression removed)
                # Run-unique suffix. The tag otherwise carries no policy name or
                # variant knobs and the clock is bucketed to 10 min, so two runs of
                # the same shape and human started close together wrote the SAME
                # observed-dataset dir and overwrote each other's day files --
                # which every policy that trains on dataset_path then read
                # (SLaTe-PRO, STREAK, the hybrid's curator, NOPA). A hash of out_path
                # is unique per config+household and stable across resubmits, so
                # resume still finds its own data.
                _run_id = hashlib.md5(str(out_path).encode()).hexdigest()[:8] if out_path else ''
                dataset_tag = f"warm{script_distribution[0]}_reset{reset_env}_suppress{suppress_robot_action}_noRep0_every{train_every_n_days}{vig_tag}{ablation_tag}{noise_tag}{human_tag}_{timestr}{'_' + _run_id if _run_id else ''}"
                dataset_out_path = script_path.replace("HOMER_PLUS", f"gen_{dataset_tag}").replace("scripts_","processed_seqLM_coarse/").replace('.txt','.pt')
                os.makedirs(os.path.dirname(dataset_out_path), exist_ok=True)
                if os.path.exists(dataset_out_path):
                    print(f"WARNING: Dataset already exists at {dataset_out_path}, overwriting")
                _atomic_write(dataset_out_path, lambda t: torch.save(batch_dict, t))
                dataset_dir = '/'.join(dataset_out_path.split('/')[:-2])
            except Exception as e:
                print(f"Error getting history: {e}")
                raise e

        # Collect per-object outcomes from the human agent
        outcomes = human.action_outcomes
        keys_list = list(set(o['outcome'] for o in outcomes))
        outcome_counts = {key: sum(1 for o in outcomes if o['outcome'] == key) for key in keys_list}
        expected_keys = ['skipped', 'corrected', 'done_as_is', 'no_movement', 'no_need_to_move',
                         'skipped_repeated', 'corrected_repeated', 'accepted',
                         'corrected_despite_correct', 'done_early', 'redirected']
        for key in expected_keys:
            if key not in outcome_counts:
                outcome_counts[key] = 0

        # Per-object detail for debugging visuals
        object_outcomes: dict[str, dict] = {}
        for o in outcomes:
            obj = o.get('object', '')
            if not obj:
                continue
            entry = object_outcomes.setdefault(obj, {k: 0 for k in expected_keys})
            entry[o['outcome']] = entry.get(o['outcome'], 0) + 1

        # LLM-human bookkeeping (no-op for scripted human)
        llm_decisions_today = list(getattr(human, 'llm_decisions', []))
        llm_stats_today = (
            human.stats() if hasattr(human, 'stats') else {}
        )

        per_day.append({
            'day':                day_idx,
            'script':             os.path.basename(script_path),
            'total_human':        len(outcomes),
            'total_robot':        robot_actions,
            'result_metrics':     result_metrics,
            'human_metrics':      outcome_counts,
            'total_script_steps': len(human.script.lines),
            'vigilance_observations': human.vigilance_observations,
            'vigilance_corrections': human.vigilance_corrections,
            'llm_decisions':      llm_decisions_today,
            'llm_stats':          llm_stats_today,
            'script_line_treatments': human.script_line_treatments,
            'human_action_events': outcomes,
            'object_outcomes':    object_outcomes,
            'world_state_timeline': day_world_trace,
            'unmatched_tensor_events': unmatched_tensor_events_today,
            'actionwise_evals_flat': [e for tlist in actionwise_evals.values() for e in tlist],
        })
        sorted_timestamps = sorted(actionwise_evals.keys())
        actionwise_evals_daily.append({k:actionwise_evals[k] for k in sorted_timestamps})

        # Per-day debug plot (paper-style tensor reconciliation; suppress only).
        if phase != 'warm-start' and suppress_robot_action and out_path is not None:
            try:
                debug_dir = os.path.join(os.path.dirname(out_path), 'debug_daily')
                debug_path = os.path.join(
                    debug_dir,
                    f"debug_{out_stem}_day{int(day_idx):03d}.png",
                )
                _debug_plot_day_actions_compare(
                    per_day_entry=per_day[-1],
                    day_actionwise=actionwise_evals_daily[-1],
                    out_path=debug_path,
                    title=f"day {day_idx} {os.path.basename(script_path)}",
                    y_gap=0.28,
                    max_objects=25,
                )
            except Exception as e:
                print(f"[WARNING] Failed to write per-day debug plot for day {day_idx}: {e}")

        if out_path is not None:
            totals = _aggregate_totals(
                per_day, script_distribution, train_every_n_days,
                suppress_robot_action=suppress_robot_action)
            result_so_far = {'done': 'upto_day' + str(day_idx), 'totals': totals, 'per_day': per_day, 'actionwise_evals_daily': actionwise_evals_daily, 'training_dataset_per_day': training_dataset_per_day}
            _append_perday(out_path, day_idx, per_day[-1],
                           actionwise_evals_daily[-1] if actionwise_evals_daily else None,
                           training_dataset_per_day.get(day_idx))
        
        next_day = day_idx + 1
        if (phase_ends['warm-start'] <= next_day < phase_ends['finetune']
                and (next_day - phase_ends['warm-start']) % train_every_n_days == 0):
            # Use plot_prediction_heatmap to visualize the predictions since the last checkpoint
            if out_path is not None:
                days = [d for d in range(day_idx - train_every_n_days, day_idx + 1)]
                checkpoint_num = (next_day - phase_ends['warm-start']) // train_every_n_days
                # Always use the stable ckpt suffix so reruns overwrite debug plots.
                heatmap_suffix = f'ckpt{checkpoint_num}'
                plot_prediction_heatmap(result_so_far, days, os.path.join(os.path.dirname(out_path)), household=sim_config['benchmark']['household'], path_suffix=heatmap_suffix)
            ckpt_stem = original_out_stem if use_prior_checkpoints else out_stem
            model_ckpt_dir = os.path.join(os.path.dirname(out_path), f'model_checkpoints_{ckpt_stem}')
            if use_prior_checkpoints:
                weights_path = os.path.join(model_ckpt_dir, f'weights_day{day_idx}.pt')
                if os.path.isfile(weights_path) and isinstance(robot_policy, RobotPolicySlatepro):
                    print(f"Loading prior checkpoint for day {day_idx}: {weights_path}")
                    robot_policy.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                    robot_policy._setup_model_for_inference()
                    last_model_weights_path = weights_path
                    training_dataset_per_day[day_idx] = {'train': -1, 'val': -1, 'loaded_from_prior': True}
                    print(f"Loaded prior checkpoint for day {day_idx}: {weights_path}")
                elif os.path.isfile(weights_path) and isinstance(robot_policy, RobotPolicyStreak):
                    print(f"Loading prior STREAK checkpoint for day {day_idx}: {weights_path}")
                    robot_policy.load_artifacts(weights_path)
                    last_model_weights_path = weights_path
                    training_dataset_per_day[day_idx] = {'train': -1, 'val': -1, 'loaded_from_prior': True}
                elif os.path.isfile(weights_path) and isinstance(
                        robot_policy, (RobotPolicyHybrid, RobotPolicyPlainLLM,
                                       RobotPolicyOracle, RobotPolicyNOPA)):
                    print(f"Loading prior policy checkpoint for day {day_idx}: {weights_path}")
                    robot_policy.load_artifacts(weights_path)
                    last_model_weights_path = weights_path
                    training_dataset_per_day[day_idx] = {'train': -1, 'val': -1, 'loaded_from_prior': True}
                else:
                    print(f"[WARNING] Prior checkpoint missing at {weights_path}; keeping current model")
            else:
                robot_policy.train('/'.join(dataset_out_path.split('/')[:-2]), output_dir=model_ckpt_dir, days=str(day_idx))
                training_dataset_per_day[day_idx] = robot_policy.data_quantity
                last_model_weights_path = os.path.join(model_ckpt_dir, f'weights_day{day_idx}.pt')


        # Save resume checkpoint after each completed day
        if resume_path:
            # per_day/actionwise/training_dataset are NOT stored here: they are
            # appended to the _perday.jsonl sidecar and replayed on resume. Keeping
            # them made every daily checkpoint ~0.9 GB on top of the result json.
            resume_data = {
                'day_idx': day_idx,
                'perday_jsonl': True,
                'phase': phase,
                'model_weights_path': last_model_weights_path,
                'timestr': timestr,
            }
            if not reset_env:
                resume_data['world_state_dict'] = world.env.to_dict()
            _atomic_write(resume_path, lambda t: torch.save(resume_data, t))

        # Write human-readable status file
        if out_path and not use_prior_checkpoints:
            dataset_files_on_disk = 0
            if dataset_dir:
                for split in ('train', 'test'):
                    split_dir = os.path.join(dataset_dir, split)
                    if os.path.isdir(split_dir):
                        dataset_files_on_disk += len([f for f in os.listdir(split_dir) if f.endswith('.pt')])
            status_path = os.path.join(os.path.dirname(out_path), f'{out_stem}_status.txt')
            with open(status_path, 'w') as sf:
                sf.write(f"day: {day_idx}/{len(script_files)-1}\n")
                sf.write(f"phase: {phase}\n")
                sf.write(f"dataset_dir: {dataset_dir}\n")
                sf.write(f"dataset_files: {dataset_files_on_disk}\n")
                sf.write(f"last_model_weights: {last_model_weights_path}\n")

    # Aggregate
    totals = _aggregate_totals(
        per_day, script_distribution, train_every_n_days,
        suppress_robot_action=suppress_robot_action)
    if out_path is not None:
        result_so_far = {'done': True, 'totals': totals, 'per_day': per_day, 'actionwise_evals_daily': actionwise_evals_daily, 'training_dataset_per_day': training_dataset_per_day}
        _atomic_write(out_path, lambda t: _dump_json(_json_sanitize(result_so_far), t))
        if script_distribution[1] == 0:
            days = [d for d in range(day_idx - script_distribution[0], day_idx + 1)]
            plot_prediction_heatmap(result_so_far, days, os.path.join(os.path.dirname(out_path)), household=sim_config['benchmark']['household'], path_suffix='warmstart')
    # Clean up resume checkpoint on successful completion
    if resume_path and os.path.isfile(resume_path):
        os.remove(resume_path)
        print("Simulation complete, resume checkpoint removed.")
    return {'totals': totals, 'per_day': per_day, 'actionwise_evals_daily': actionwise_evals_daily, 'training_dataset_per_day': training_dataset_per_day}


def _aggregate_totals(
    per_day: list[dict],
    script_distribution: list[int],
    train_every_n_days: int,
    *,
    suppress_robot_action: bool = False,
) -> dict:
    """Roll per-day records up into per-phase ('finetune' / 'test') totals.

    Produces, for each phase: the human-burden metrics (TP=skipped+accepted,
    FP=corrected+corrected_despite_correct+vigilance_corrections+redirected,
    FN=done_as_is; precision/recall/F1), raw env_metrics, and — when
    suppress_robot_action=True — paper-style 1-step SLaTe-PRO metrics plus
    (human_outcome x tensor_eval_category) confusion matrices for reconciling
    the human-F1 and robot-F1 views.
    redirected/vigilance are sourced from result_metrics (the per-day
    human_metrics block zeroes them); see the inline TP/FP/FN comment below.
    Warm-start days carry no robot actions and are excluded.
    """
    aggregated_totals_by_phase = {phase: {} for phase in ['finetune', 'test']}
    start_idx = 0

    def _metrics(rows: list[dict]) -> dict:
        # Can happen when script_distribution slices produce an empty phase window.
        # Return a zeroed summary instead of crashing.
        if not rows:
            env_metrics = {
                'vigilance_observations': 0,
                'vigilance_corrections': 0,
            }
            return {
                'human_metrics': {
                    'skipped': 0,
                    'corrected': 0,
                    'done_as_is': 0,
                    'no_movement': 0,
                    'no_need_to_move': 0,
                    'skipped_repeated': 0,
                    'corrected_repeated': 0,
                },
                'slatepro': None,
                'env_metrics': env_metrics,
                'human_prec': 0.0,
                'human_rec': 0.0,
                'human_f1': 0.0,
                'total_human': 0,
                'total_robot': 0,
                'total_script_steps': 0,
                'reconciliation': None,
            }

        # Human outcomes can be stored either at top-level (legacy) or under
        # r['human_metrics'] (current). Aggregate across both representations.
        hm_list = [(r.get('human_metrics') or {}) for r in rows]
        rm_list_h = [(r.get('result_metrics') or {}) for r in rows]
        # Use whichever representation is present — never sum both (double-count).
        def _cnt(key):
            return sum((hm[key] if key in hm else r.get(key, 0)) or 0
                       for r, hm in zip(rows, hm_list))
        s = _cnt('skipped')
        c = _cnt('corrected')
        d = _cnt('done_as_is')
        nm = _cnt('no_movement')
        nntm = _cnt('no_need_to_move')
        s_rep = sum((hm.get('skipped_repeated', 0) or 0) for hm in hm_list)
        c_rep = sum((hm.get('corrected_repeated', 0) or 0) for hm in hm_list)
        acc = sum((hm.get('accepted', 0) or 0) for hm in hm_list)
        cdc = sum((hm.get('corrected_despite_correct', 0) or 0) for hm in hm_list)
        vig_corr = sum((rm.get('vigilance_corrections', 0) or 0) for rm in rm_list_h)
        redirected = sum((rm.get('redirected', 0) or 0) for rm in rm_list_h)
        done_early = sum((hm.get('done_early', 0) or 0) for hm in hm_list)

        # ── TP / FP / FN categorisation ─────────────────────────────────────
        # TP — robot's action helped and the human kept it:
        #        skipped   (robot pre-empted the exact planned move)
        #      + accepted  (robot did something different; human adapted and kept it)
        # FP — robot acted but the human had to expend effort because of it:
        #        corrected                  (robot moved wrong obj/loc; human fixed)
        #      + corrected_despite_correct   (human moved obj away even though the
        #                                     robot was right — still human effort)
        #      + vigilance_corrections       (human proactively undid a robot action
        #                                     on observing it. For the LLM human this
        #                                     IS its 'undo' decision; for the scripted
        #                                     cond4/cond5 baselines vigilance is off so
        #                                     this term is 0.)
        #      + redirected                  (v2 LLM: human moved the robot's
        #                                     misplacement straight to its planned
        #                                     destination in one trip. One FP, like a
        #                                     scripted correction — and the later
        #                                     planned line is 'done_early', NOT a TP.)
        # FN — robot failed to help and the human did it unassisted:
        #        done_as_is
        #
        # EXCLUDED on purpose:
        #  - done_early : a planned line already satisfied by an earlier human
        #    redirect — NOT a robot success; the human did it. Ignored entirely.
        #  - undone_movements : robot overwriting its OWN prior prediction — robot
        #    self-churn, not human effort, so it has no place in a human-burden metric.
        #  - skipped_repeated / corrected_repeated : duplicate script lines for the
        #    same (obj,dest) after the first robot match — counting them inflates.
        #  - no_movement / no_need_to_move : not robot-relevant. Neutral.
        tp = s + acc
        fp = c + cdc + vig_corr + redirected
        fn = d
        human_prec = tp / (tp + fp + 1e-9)
        human_rec = tp / (tp + fn + 1e-9)
        human_f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)


        total_robot = sum(r.get('total_robot', 0) for r in rows)
        total_human = sum(r.get('total_human', 0) for r in rows)
        total_script_steps = sum(r.get('total_script_steps', 0) for r in rows)

        result_metrics = [(r.get('result_metrics') or {}) for r in rows]
        env_metric_keys = set()
        for rm in result_metrics:
            if isinstance(rm, dict):
                env_metric_keys.update(rm.keys())
        env_metrics = {key: sum((rm.get(key, 0) or 0) for rm in result_metrics if isinstance(rm, dict)) for key in env_metric_keys}
        env_metrics['vigilance_observations'] = sum(r.get('vigilance_observations', 0) for r in rows)
        env_metrics['vigilance_corrections'] = sum(r.get('vigilance_corrections', 0) for r in rows)

        slatepro_block = None
        reconciliation = None
        if suppress_robot_action:
            # Paper-style 1-step metrics: aggregate per-key sums across rows then
            # derive precision / recall / F1 with paper convention (wrong_destination
            # contributes to both FP and FN).
            def _sum(key):
                return sum((rm.get(key, 0) or 0) for rm in result_metrics)
            prec_correct           = _sum('precision_correct')
            prec_wrong_destination = _sum('precision_wrong_destination')
            prec_wrong_object      = _sum('precision_wrong_object')
            prec_total             = _sum('precision_total')
            moved_missed           = _sum('moved_missed')
            moved_total            = _sum('moved_total')
            unmoved_fp             = _sum('unmoved_fp')
            unmoved_total          = _sum('unmoved_total')

            slatepro_tp = prec_correct
            slatepro_fp = prec_wrong_destination + prec_wrong_object
            slatepro_fn = moved_missed + prec_wrong_destination
            slatepro_prec = slatepro_tp / (slatepro_tp + slatepro_fp + 1e-9)
            slatepro_rec  = slatepro_tp / (slatepro_tp + slatepro_fn + 1e-9)
            slatepro_f1   = 2 * slatepro_prec * slatepro_rec / (slatepro_prec + slatepro_rec + 1e-9)
            slatepro_block = {
                'precision_correct': prec_correct,
                'precision_wrong_destination': prec_wrong_destination,
                'precision_wrong_object': prec_wrong_object,
                'precision_total': prec_total,
                'moved_missed': moved_missed,
                'moved_total': moved_total,
                'unmoved_fp': unmoved_fp,
                'unmoved_total': unmoved_total,
                'precision': slatepro_prec,
                'recall': slatepro_rec,
                'f1': slatepro_f1,
            }

            # (human_outcome × tensor_eval_category) confusion matrices — one per
            # tensor view (1step and anyFuture). Three contributing streams:
            #   - actionwise_evals_flat: every robot action that got matched.
            #   - human_action_events: human script lines with no pending robot action.
            #     skipped/corrected variants are excluded here to avoid double-counting.
            #   - unmatched_tensor_events: tensor saw a movement but neither side
            #     recorded it (drop-by-multi-actions filter, ambient changes).
            _human_only_outcomes = ('done_as_is', 'no_movement', 'no_need_to_move')

            def _build_confusion(view_key: str) -> dict:
                cell: dict[tuple[str, str], int] = {}
                field = f'tensor_eval_category_{view_key}'
                for r in rows:
                    for ev in (r.get('actionwise_evals_flat') or []):
                        if not isinstance(ev, dict):
                            continue
                        key = (str(ev.get('human_action_outcome', 'NONE')),
                               str(ev.get(field, 'NONE')))
                        cell[key] = cell.get(key, 0) + 1
                    for o in (r.get('human_action_events') or []):
                        if not isinstance(o, dict):
                            continue
                        if o.get('outcome') not in _human_only_outcomes:
                            continue
                        key = (str(o.get('outcome', 'NONE')),
                               str(o.get(field, 'no_event')))
                        cell[key] = cell.get(key, 0) + 1
                    for ev in (r.get('unmatched_tensor_events') or []):
                        if not isinstance(ev, dict):
                            continue
                        key = ('NONE', str(ev.get(field, 'NONE')))
                        cell[key] = cell.get(key, 0) + 1
                return cell

            def _summarize(cell: dict) -> dict:
                return {
                    'confusion': {f'{h}__VS__{t}': v for (h, t), v in sorted(cell.items())},
                    'diagonal': {
                        'skipped_x_precision_correct':              cell.get(('skipped', 'precision_correct'), 0),
                        'corrected_x_precision_wrong_destination':  cell.get(('corrected', 'precision_wrong_destination'), 0),
                        'done_as_is_x_moved_missed':                cell.get(('done_as_is', 'moved_missed'), 0),
                    },
                    'common_off_diagonal': {
                        'skipped_but_tensor_no_event':              cell.get(('skipped', 'no_event'), 0),
                        'skipped_but_tensor_wrong_destination':     cell.get(('skipped', 'precision_wrong_destination'), 0),
                        'skipped_but_tensor_wrong_object':          cell.get(('skipped', 'precision_wrong_object'), 0),
                        'corrected_but_tensor_correct':             cell.get(('corrected', 'precision_correct'), 0),
                        'corrected_but_tensor_wrong_object':        cell.get(('corrected', 'precision_wrong_object'), 0),
                        'corrected_but_tensor_no_event':            cell.get(('corrected', 'no_event'), 0),
                        'done_as_is_but_tensor_predicted':          (cell.get(('done_as_is', 'precision_correct'), 0)
                                                                     + cell.get(('done_as_is', 'precision_wrong_destination'), 0)),
                        'done_as_is_but_tensor_no_event':           cell.get(('done_as_is', 'no_event'), 0),
                        'unseen_but_tensor_correct':                (cell.get(('unseen', 'precision_correct'), 0)
                                                                     + cell.get(('unseen_at_end_of_day', 'precision_correct'), 0)),
                        'tensor_only_moved_missed':                 cell.get(('NONE', 'moved_missed'), 0),
                        'tensor_only_wrong_object':                 cell.get(('NONE', 'precision_wrong_object'), 0),
                        'tensor_only_wrong_destination':            cell.get(('NONE', 'precision_wrong_destination'), 0),
                    },
                }

            reconciliation = {
                '1step': _summarize(_build_confusion('1step')),
                'anyFuture': _summarize(_build_confusion('anyFuture')),
            }

        out = {
            'human_metrics': {
                'skipped': s,
                'corrected': c,
                'done_as_is': d,
                'no_movement': nm,
                'no_need_to_move': nntm,
                'skipped_repeated': s_rep,
                'corrected_repeated': c_rep,
                'accepted': acc,
                'corrected_despite_correct': cdc,
                'vigilance_corrections': vig_corr,
                'redirected': redirected,
                'done_early': done_early,
                'tp': tp,
                'fp': fp,
                'fn': fn,
            },
            'slatepro': slatepro_block,
            'env_metrics': env_metrics,
            'human_prec': human_prec,
            'human_rec': human_rec,
            'human_f1': human_f1,
            'total_human': total_human,
            'total_robot': total_robot,
            'total_script_steps': total_script_steps,
            'reconciliation': reconciliation,
        }
        return out

    for phase, num_scripts in zip(['warm-start', 'finetune', 'test'], script_distribution):
        if phase == 'warm-start':
            start_idx += num_scripts
            continue
        per_day_this_phase = per_day[start_idx:start_idx + num_scripts]
        m = _metrics(per_day_this_phase)
        if len(per_day) < start_idx + num_scripts:
            m['day_start_idx'] = start_idx
            m['day_end_idx'] = len(per_day)
        else:
            m['day_start_idx'] = start_idx
            m['day_end_idx'] = start_idx + num_scripts
        print(f"Human precision: {m['human_prec']:.3f}, recall: {m['human_rec']:.3f}, f1: {m['human_f1']:.3f}")
        start_idx += num_scripts

        aggregated_totals_by_phase[phase] = m

    # Also break down finetune metrics by checkpoint (keyed by number of training files).
    # A checkpoint is created every train_every_n_days, so aggregate over *that window*
    # (non-overlapping chunks), clipped at finetune end if needed.
    finetune_by_ckpt: dict[int, dict] = {}
    warm_n = script_distribution[0] if len(script_distribution) > 0 else 0
    finetune_n = script_distribution[1] if len(script_distribution) > 1 else 0
    finetune_start = int(warm_n)
    finetune_end_excl = int(warm_n + finetune_n)

    chunk_start = finetune_start
    while chunk_start < finetune_end_excl:
        chunk_end_excl = min(chunk_start + int(train_every_n_days), finetune_end_excl)
        rows = per_day[chunk_start:chunk_end_excl]
        m = _metrics(rows)
        finetune_by_ckpt[int(chunk_start)] = {
            **m,
            "window_start_day_idx": int(chunk_start),
            "window_end_day_idx": int(chunk_end_excl-1),
        }

        chunk_start += int(train_every_n_days)

    aggregated_totals_by_phase["finetune_by_ckpt"] = finetune_by_ckpt
    return aggregated_totals_by_phase

# NOTE: This module is now library-only — it provides ``evaluate_closed_loop`` /
# ``_aggregate_totals`` to the benchmark drivers (e.g. scripts/benchmark_comparison.py).
# The standalone ``main()`` that drove the (removed) continual DPO model was deleted
# along with that model.
