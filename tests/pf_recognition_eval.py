"""Offline recognition eval for the particle filter.

Replays the act+reset eval days (clean GT streams from processed test .pt —
caveat: no robot perturbation in the stream) through GraphParticleFilter
using a given run's memory dir (its task graphs + tagged training days), and
scores current-activity recognition with the SAME methodology as
analyze_hybrid_errors.py stage 1: curated->GT mapping by dominant training
label, GT schedule at t (+/- 1 tick), gt_idle excluded, abstention bucket.

Usage:
  python tests/pf_recognition_eval.py \
      --memory_dir results/<run>/hybrid_memory_A --household A [--seed 0]
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from collections import Counter

from agents.task_graph import TaskGraphSet
from agents.llm_memory_curator import extract_events_from_pt
from agents.activity_particle_filter import GraphParticleFilter, BG_LABEL
from scripts.analyze_hybrid_errors import (curated_to_gt_mapping,
                                           load_schedules, gt_at)


def build_room_of(pt_path: str) -> dict:
    """node idx -> room idx from the initial frame's parent chains."""
    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    parent = d['edges'][0].argmax(dim=-1).tolist()
    classes = list(d['class_names'])
    home = [i for i, c in enumerate(classes) if c == 'home']
    home = home[0] if home else -1
    rooms = {i for i, p in enumerate(parent) if p == home and i != home}
    room_of = {}
    for i in range(len(parent)):
        j, hops = i, 0
        while j not in rooms and j != home and hops < 10:
            j = parent[j]
            hops += 1
        if j in rooms:
            room_of[i] = j
    return room_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memory_dir', required=True)
    ap.add_argument('--household', required=True)
    ap.add_argument('--eval_day_offset', type=int, default=65)
    ap.add_argument('--n_days', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--drop', type=float, default=0.0,
                    help='fraction of events removed (robot claimed them)')
    ap.add_argument('--noise', type=float, default=0.0,
                    help='fraction of extra noise moves injected (reactions)')
    args = ap.parse_args()

    mapping = curated_to_gt_mapping(args.memory_dir)
    sched = load_schedules(args.household, args.eval_day_offset, args.n_days)
    graphs = TaskGraphSet.load(os.path.join(args.memory_dir,
                                            'task_graphs.json'))
    import json, glob
    tagged = [json.load(open(p)) for p in sorted(
        glob.glob(os.path.join(args.memory_dir, 'tagged_days', '*.json')))]

    test_dir = os.path.join(PROJECT_ROOT, 'external', 'HOMER_PLUS',
                            f'Household{args.household}',
                            'processed_seqLM_coarse', 'test')
    room_of = build_room_of(os.path.join(test_dir, '000.pt'))

    rec = Counter()
    for d in range(args.n_days):
        pt = os.path.join(test_dir, f'{d:03d}.pt')
        if not os.path.isfile(pt) or d not in sched:
            continue
        events = extract_events_from_pt(pt)['events']
        import random as _r
        prng = _r.Random(1000 + args.seed + d)
        scored = [True] * len(events)
        if args.drop > 0:
            keep = []
            for e in events:
                if prng.random() >= args.drop:
                    keep.append(e)
            events = keep
            scored = [True] * len(events)
        if args.noise > 0:
            pairs = [(e['obj_idx'], e['dest_idx']) for e in events]
            t_lo = min(e['t_min'] for e in events)
            t_hi = max(e['t_min'] for e in events)
            n_noise = int(args.noise * len(events))
            for _ in range(n_noise):
                o, de = prng.choice(pairs)
                events.append({'obj_idx': o, 'dest_idx': prng.choice(pairs)[1],
                               't_min': prng.uniform(t_lo, t_hi)})
                scored.append(False)   # noise itself is not scored
            order = sorted(range(len(events)),
                           key=lambda i: events[i]['t_min'])
            events = [events[i] for i in order]
            scored = [scored[i] for i in order]
        pf = GraphParticleFilter(graphs, tagged, room_of=room_of,
                                 seed=args.seed + d)
        for e, sc in zip(events, scored):
            pf.observe(e['obj_idx'], e['dest_idx'], e['t_min'])
            if not sc:
                continue
            post = pf.posterior_last_assign()
            top, p_top = next(iter(post.items()))
            gt_now = gt_at(sched[d], e['t_min'])
            gt_prev = gt_at(sched[d], e['t_min'] - 10)
            if top == BG_LABEL:
                rec['no_confident_tag'] += 1
            elif mapping.get(top, top) in (gt_now, gt_prev):
                rec['correct'] += 1
            elif gt_now == 'idle' and gt_prev == 'idle':
                rec['gt_idle'] += 1
            else:
                rec['wrong'] += 1

    tot = sum(rec.values())
    print(f"=== PF recognition (clean replay), {tot} events, "
          f"{args.memory_dir} ===")
    for k, v in rec.most_common():
        print(f"   {k:<18} {v:>5}  ({100*v/tot:.0f}%)")
    committed = rec['correct'] + rec['wrong']
    if committed:
        print(f"   acc when committed: {100*rec['correct']/committed:.0f}%")


if __name__ == '__main__':
    main()
