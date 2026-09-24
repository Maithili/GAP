"""Component-level error breakdown for the hybrid policy on an eval run.

Decomposes performance into the three pipeline stages (post-hoc GT use OK):
  1. activity RECOGNITION  — eval_tag_log current_activity vs GT schedule at t
  2. next-activity ANTICIPATION — policy_diag pred_next vs the actual next GT
     activity in the schedule (incl. gap error, and accuracy when prep fired)
  3. within-activity ACTION selection — candidate outcomes by source, plus
     how many candidates the guards filtered (incl. already-done coverage)

Curated activity names are mapped to GT names by their dominant GT label in
the training memory (same curated->GT mapping method).

Usage:
  python scripts/analyze_hybrid_errors.py --memory_dir <...>/hybrid_memory_A \
      --household A --eval_day_offset 65
(eval day d in the logs corresponds to scripts_test/<d>.json when the run
used WARMUP_DAYS=65 ALL_SCRIPTS=1 with USE_PRIOR_CHECKPOINTS; days 65..74.)
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch


def curated_to_gt_mapping(memory_dir: str) -> dict:
    """Dominant GT label per curated activity, from the training tagged days."""
    mapping = defaultdict(Counter)
    act_names = None
    for tp in sorted(glob.glob(os.path.join(memory_dir, 'tagged_days', '*.json'))):
        day = json.load(open(tp))
        d = torch.load(day['source_pt'], map_location='cpu', weights_only=False)
        gt_act, times = d['activity'], d['times']
        if act_names is None:
            m = re.search(r'Household([A-C])', day['source_pt'])
            p = os.path.join(PROJECT_ROOT, 'external', 'HOMER_PLUS',
                             f'Household{m.group(1)}', 'processed_seqLM_coarse',
                             'common_data.json')
            act_names = json.load(open(p))['activities']
        for seg in day['segments']:
            for i in seg['event_idxs']:
                t_min = day['events'][i]['t_min']
                tick = max(0, int(round((t_min - float(times[0])) / 10.0)) - 1)
                tick = min(tick, len(gt_act) - 1)
                mapping[seg['activity_name']][act_names[int(gt_act[tick])]] += 1
    return {cur: ctr.most_common(1)[0][0] for cur, ctr in mapping.items()}


def load_schedules(household: str, eval_day_offset: int, n_days: int) -> dict:
    """day_in_process -> schedule list [{activity, start, end}] (test scripts)."""
    sched = {}
    for d in range(n_days):
        script_day = eval_day_offset + d
        # 65 train scripts then test scripts 000..009
        if script_day >= 65:
            p = os.path.join(PROJECT_ROOT, 'external', 'HOMER_PLUS',
                             f'Household{household}', 'converted_scripts_test',
                             f'{script_day - 65:03d}.json')
        else:
            p = os.path.join(PROJECT_ROOT, 'external', 'HOMER_PLUS',
                             f'Household{household}', 'converted_scripts_train',
                             f'{script_day:03d}.json')
        if os.path.isfile(p):
            sched[d] = json.load(open(p))['schedule']
    return sched


def gt_at(schedule: list, t: float) -> str:
    for e in schedule:
        if e['start'] <= t < e['end']:
            return e['activity']
    return 'idle'


def gt_next_after(schedule: list, t: float):
    for e in schedule:
        if e['start'] > t:
            return e['activity'], e['start'] - t
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memory_dir', required=True)
    ap.add_argument('--household', required=True)
    ap.add_argument('--eval_day_offset', type=int, default=65)
    args = ap.parse_args()

    mapping = curated_to_gt_mapping(args.memory_dir)

    # ---------------- stage 1: recognition ----------------
    tags = [json.loads(l) for l in
            open(os.path.join(args.memory_dir, 'eval_tag_log.jsonl'))]
    n_days = max(t['day_in_process'] for t in tags) + 1
    sched = load_schedules(args.household, args.eval_day_offset, n_days)
    rec = Counter()
    for t in tags:
        s = sched.get(t['day_in_process'])
        if s is None:
            continue
        cur = t['current_activity']
        if cur is None:
            rec['no_confident_tag'] += 1
            continue
        gt_now = gt_at(s, t['t_min'])
        gt_prev = gt_at(s, t['t_min'] - 10)  # events observed 1 tick late
        mapped = mapping.get(cur, cur)
        if mapped in (gt_now, gt_prev):
            rec['correct'] += 1
        elif gt_now == 'idle' and gt_prev == 'idle':
            rec['gt_idle'] += 1     # human between activities; tag unverifiable
        else:
            rec['wrong'] += 1
    tot = sum(rec.values())
    print(f"=== 1. activity recognition (eval tagger), {tot} calls ===")
    for k, v in rec.most_common():
        print(f"   {k:<18} {v:>5}  ({100*v/tot:.0f}%)")

    # ---------------- stage 2: anticipation ----------------
    diag = [json.loads(l) for l in
            open(os.path.join(args.memory_dir, 'policy_diag.jsonl'))]
    ant = Counter(); gap_errs = []; ant_prep = Counter()
    for r in diag:
        s = sched.get(r['day_in_process'])
        if s is None or 'pred_next' not in r:
            continue
        pred = r['pred_next']
        actual, actual_gap = gt_next_after(s, r['t_min'])
        if actual is None:
            continue
        mapped = mapping.get(pred['name'], pred['name'])
        ok = (mapped == actual)
        ant['correct' if ok else 'wrong'] += 1
        if pred['gap_min'] is not None:
            gap_errs.append(abs(pred['gap_min'] - actual_gap))
        if r.get('prep_fired'):
            ant_prep['correct' if ok else 'wrong'] += 1
    tot = sum(ant.values())
    print(f"\n=== 2. next-activity anticipation (transformer), {tot} ticks ===")
    print(f"   top-1 vs actual next GT activity: "
          f"{100*ant['correct']/max(tot,1):.0f}% correct")
    tp = sum(ant_prep.values())
    print(f"   when prep FIRED ({tp} ticks): "
          f"{100*ant_prep['correct']/max(tp,1):.0f}% correct")
    gap_errs.sort()
    if gap_errs:
        print(f"   |gap error| median={gap_errs[len(gap_errs)//2]:.0f} min, "
              f"p75={gap_errs[int(.75*len(gap_errs))]:.0f} min")

    # ---------------- stage 3: action selection + filters ----------------
    out_by_src = defaultdict(Counter)
    filt = Counter(); too_early = 0
    for r in diag:
        too_early += r.get('too_early', 0)
        for c in r['candidates']:
            filt[c['outcome']] += 1
            if c['outcome'] == 'emitted':
                out_by_src[c['source']]['emitted'] += 1
    print(f"\n=== 3. candidate filtering (incl. already-done guards) ===")
    print(f"   too_early (timing gate): {too_early}")
    for k, v in filt.most_common():
        print(f"   {k:<18} {v:>6}")
    print("   (already_at_dest = object already where it should be — covers "
          "actions the human or robot already completed)")


if __name__ == '__main__':
    main()
