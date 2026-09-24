#!/usr/bin/env python
"""Recompute the human-burden metric consistently across scripted / v1 / v2
LLM-human conditions and print a comparison table.

human_f1 with the FINAL formula:
    TP = skipped + accepted
    FP = corrected + corrected_despite_correct + vigilance_corrections + redirected
    FN = done_as_is
    (done_early, no_need_to_move, no_movement, *_repeated, undone_movements excluded)
    f1 = 2*TP / (2*TP + FP + FN)

Reads each condition's results/<dir>/benchmark_slatepro_{A,B,C}.json
totals.finetune block (which stores the raw per-outcome counts), so the metric
is recomputed identically regardless of which code version wrote the JSON.

Usage:
    python scripts/analyze_llm_human_conditions.py
    python scripts/analyze_llm_human_conditions.py --markdown   # emit findings.md-ready table
"""
import argparse
import glob
import json
import os

# label -> (dir glob prefix, expected finetune days for "complete" heuristic)
CONDITIONS = [
    ('cond4  scripted retrain',  'cond4_warm5_every5_resetF',                         70),
    ('cond5  scripted act+reset','cond5_warm65_suppressF_resetT',                      10),
    ('cond19 v1 act+reset',      'cond19_warm65_suppressF_resetT_humanllm',            10),
    ('cond20 v1 retrain',        'cond20_warm5_every5_resetF_humanllm',                70),
    ('cond21 v2 act+reset',      'cond21_warm65_suppressF_resetT_humanllm_redirect',   10),
    ('cond22 v2 retrain',        'cond22_warm5_every5_resetF_humanllm_redirect',       70),
    ('cond23 hybrid act+reset',  'cond23_warm65_suppressF_resetT_hybrid',              10),
    ('cond23w hybrid tuned',     'cond23winner',                                       10),
    ('cond24 hybrid+llmv2',      'cond24_warm65_suppressF_resetT_hybrid_humanllm_redirect', 10),
    ('cond25 hybrid retrain',    'cond25_warm5_every5_resetF_hybrid',                  70),
    ('cond23b hybrid+gate',      'cond23b_gate',                                       10),
    ('cond24b hybrid+gate+llmv2','cond24b_gate_humanllm_redirect',                     10),
    ('cond26b hybrid retrain+llmv2', 'cond26b_warm5_every5_resetF_hybrid_humanllm_redirect', 70),
    ('cond25b hybrid retrain+gate', 'cond25b_warm5_every5_resetF_hybrid',                  70),
    ('cond27 plainllm act+reset', 'cond27_warm65_suppressF_resetT_plainllm',     10),
    ('cond25c hybrid retrain t005c12', 'cond25c_warm5_every5_resetF_hybrid_t005c12', 70),
    ('cond26c hybrid retrain+llmv2 t005c12', 'cond26c_warm5_every5_resetF_hybrid_t005c12_humanllm_redirect', 70),
    ('cond28 plainllm+llmv2',    'cond28_warm65_suppressF_resetT_plainllm_warmup65_test0_every99999_all1_noresetoff_humanllm_redirect', 10),
    ('cond29 plainllm retrain',  'cond29_warm5_every5_resetF_plainllm',                70),
    ('cond30 plainllm retrain+llmv2', 'cond30_warm5_every5_resetF_plainllm_warmup5_test0_every5_all1_noreset1_humanllm_redirect', 70),
    ('cond23v21 hybrid v2.1+ranker', 'cond23v21_warmup65_test0_every99999_all1_noresetoff', 10),
    ('cond24v21 hybrid v2.1+ranker+llmv2', 'cond24v21_warmup65_test0_every99999_all1_noresetoff_modeevent_humanllm_redirect', 10),
    ('cond23v15 hybrid v1.5 (B/C)', 'cond23v15_warmup65_test0_every99999_all1_noresetoff', 10),
    ('cond24v15 hybrid v1.5+llmv2 (B/C)', 'cond24v15_warmup65_test0_every99999_all1_noresetoff_modeevent_humanllm_redirect', 10),
    ('cond25v15 hybrid v1.5 retrain', 'cond25v15_warm5_every5_resetF', 70),
    ('cond26v15 hybrid v1.5 retrain+llmv2', 'cond26v15_warm5_every5_resetF_warmup5_test0_every5_all1_noreset1_modeevent_humanllm_redirect', 70),
    ('cond23pf hybrid PF', 'cond23pf_warmup65_test0_every99999_all1_noresetoff', 10),
    ('cond24pf hybrid PF+llmv2', 'cond24pf_warmup65_test0_every99999_all1_noresetoff_modeevent_humanllm_redirect', 10),
    ('cond23pf2 hybrid PF2', 'cond23pf2_warmup65_test0_every99999_all1_noresetoff', 10),
    ('cond24pf2 hybrid PF2+llmv2', 'cond24pf2_warmup65_test0_every99999_all1_noresetoff_modeevent_humanllm_redirect', 10),
    ('cond25pf hybrid PF2 retrain', 'cond25pf_warm5_every5_resetF', 70),
    ('cond26pf hybrid PF2 retrain+llmv2', 'cond26pf_warm5_every5_resetF_warmup5_test0_every5_all1_noreset1_modeevent_humanllm_redirect', 70),
    ('cond25pfcl hybrid PFv2-CL retrain', 'cond25pfcl_warm5_every5_resetF', 70),
    ('cond26pfcl hybrid PFv2-CL retrain+llmv2', 'cond26pfcl_warm5_every5_resetF_warmup5_test0_every5_all1_noreset1_modeevent_humanllm_redirect', 70),
    ('cond31 ORACLE pure', 'cond31_oracle_pure', 10),
    ('cond31m1 ORACLE margin1', 'cond31m1_oracle_pure', 10),
    ('cond33 ORACLE pure retrain', 'cond33_oracle_pure_retrain', 70),
    ('cond35 ORACLE pure +llmv2', 'cond35_oracle_pure_llmhuman_warmup65_test0_every99999_all1_noresetoff_humanllm_redirect', 10),
    ('cond37 ORACLE pure retrain+llmv2', 'cond37_oracle_pure_retrain_llmhuman_warmup5_test0_every5_all1_noreset1_humanllm_redirect', 70),
    ('cond31t1 ORACLE pure 1tick', 'cond31t1_oracle_pure', 10),
    ('cond33t1 ORACLE pure retrain 1tick', 'cond33t1_oracle_pure_retrain', 70),
    ('cond35t1 ORACLE pure +llmv2 1tick', 'cond35t1_oracle_pure_llmhuman_warmup65_test0_every99999_all1_noresetoff_humanllm_redirect', 10),
    ('cond37t1 ORACLE pure retrain+llmv2 1tick', 'cond37t1_oracle_pure_retrain_llmhuman_warmup5_test0_every5_all1_noreset1_humanllm_redirect', 70),
]


def find_dir(prefix):
    # exclude redirect dirs when matching the non-redirect v1 prefix
    cands = sorted(d for d in glob.glob(f'results/{prefix}_*/') + glob.glob(f'results/{prefix}/'))
    if not prefix.endswith('redirect'):
        cands = [d for d in cands if 'redirect' not in d]
    return cands[-1] if cands else None


def recompute(d, hh):
    # hybrid-policy runs write benchmark_hybrid_*.json; all others slatepro
    p = os.path.join(d, f'benchmark_slatepro_{hh}.json')
    if not os.path.isfile(p):
        p = os.path.join(d, f'benchmark_hybrid_{hh}.json')
    if not os.path.isfile(p):
        p = os.path.join(d, f'benchmark_plainllm_{hh}.json')
    if not os.path.isfile(p):
        p = os.path.join(d, f'benchmark_oracle_{hh}.json')
    if not os.path.isfile(p):
        return None
    # The file exists (checked above) — a JSON error means it's corrupt, so
    # surface it instead of silently dropping the condition from the table.
    try:
        j = json.load(open(p))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"corrupt benchmark JSON {p}: {e}") from e
    t = (j.get('totals') or {}).get('finetune') or {}
    if not t:
        return None
    hm = t.get('human_metrics') or {}
    em = t.get('env_metrics') or {}
    g = lambda k: hm.get(k, 0) or 0
    skipped, acc = g('skipped'), g('accepted')
    corr, cdc = g('corrected'), g('corrected_despite_correct')
    vig = hm.get('vigilance_corrections', em.get('vigilance_corrections', 0)) or 0
    redirected = hm.get('redirected', em.get('redirected', 0)) or 0
    done = g('done_as_is')
    done_early = g('done_early')
    # Robot-side: actions the human never reconciled (robot moved an object the
    # human didn't reach/match in-window). Lives in env_metrics, not human_metrics.
    unseen = em.get('unseen', 0) or 0
    tp = skipped + acc
    fp = corr + cdc + vig + redirected
    fn = done
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    days = t.get('day_end_idx', 0) - t.get('day_start_idx', 0)
    # Effort metric (trip accounting): a correction/redirect/cdc is neutral
    # (the human makes exactly one trip, same as solo); an observe-time undo
    # is a trip with no solo-world counterpart. done_early is the downstream
    # half of a redirect pair (the human already paid that trip at redirect
    # time — closed_loop_eval.py "satisfied by an earlier human redirect"),
    # so it is neither a save nor a separate solo line.
    trips_saved = skipped + acc
    trips_caused = vig
    net_trips = trips_saved - trips_caused
    # actions made HARDER: trip-neutral but perturbed — the human still makes
    # one trip, but from the robot's wrong spot, plus the attention cost
    harder = corr + cdc + redirected
    solo_workload = skipped + acc + corr + cdc + redirected + done
    return dict(days=days, f1=f1, tp=tp, fp=fp, fn=fn, skipped=skipped, acc=acc,
                corr=corr, cdc=cdc, vig=vig, redirected=redirected,
                done=done, done_early=done_early, unseen=unseen,
                net_trips=net_trips, solo=solo_workload,
                trips_day=net_trips / max(days, 1),
                harder_day=harder / max(days, 1),
                workload_pct=100.0 * net_trips / max(solo_workload, 1),
                done_marker=os.path.isfile(os.path.join(d, f'.done_{hh}')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markdown', action='store_true')
    args = ap.parse_args()

    hdr = (f"{'condition':<26}{'HH':>3}{'days':>5}{'done':>5}{'F1':>7}"
           f"{'TP':>6}{'FP':>6}{'FN':>6} | {'skip':>5}{'acc':>4}{'corr':>5}"
           f"{'cdc':>4}{'undo':>5}{'redir':>6}{'dEarly':>7}{'done':>6}{'unseen':>7}"
           f" | {'trip/d':>7}{'wl%':>7}{'hard/d':>7}")
    print(hdr); print('-' * len(hdr))
    avg = {}
    for label, prefix, exp_days in CONDITIONS:
        d = find_dir(prefix)
        if not d:
            print(f"{label:<26}  (not run yet)"); continue
        rows = []
        for hh in 'ABC':
            m = recompute(d, hh)
            if not m or (m['tp'] + m['fp'] + m['fn']) == 0:
                print(f"{label:<26}{hh:>3}  (no finetune data yet)"); continue
            rows.append(m)
            dn = 'Y' if m['done_marker'] else '-'
            print(f"{label:<26}{hh:>3}{m['days']:>5}{dn:>5}{m['f1']:>7.3f}"
                  f"{m['tp']:>6}{m['fp']:>6}{m['fn']:>6} | {m['skipped']:>5}{m['acc']:>4}"
                  f"{m['corr']:>5}{m['cdc']:>4}{m['vig']:>5}{m['redirected']:>6}"
                  f"{m['done_early']:>7}{m['done']:>6}{m['unseen']:>7}"
                  f" | {m['trips_day']:>7.1f}{m['workload_pct']:>7.1f}"
                  f"{m['harder_day']:>7.1f}")
        if rows:
            mf = sum(r['f1'] for r in rows) / len(rows)
            mtd = sum(r['trips_day'] for r in rows) / len(rows)
            mwl = sum(r['workload_pct'] for r in rows) / len(rows)
            mhd = sum(r['harder_day'] for r in rows) / len(rows)
            avg[label] = (mf, mtd, mwl, mhd, len(rows))
            print(f"{label+' [avg]':<26}{'':>3}{'':>5}{'':>5}{mf:>7.3f}"
                  f"{'':>6}{'':>6}{'':>6} | {'':>5}{'':>4}{'':>5}{'':>4}{'':>5}"
                  f"{'':>6}{'':>7}{'':>6}{'':>7} | {mtd:>7.1f}{mwl:>7.1f}")
        print()

    print("=== avg summary: human_f1 | net trips saved/day | % workload removed | actions made harder/day ===")
    print("    (* = partial / not all HH .done)")
    for label, _, _ in CONDITIONS:
        if label in avg:
            mf, mtd, mwl, mhd, n = avg[label]
            star = '' if n == 3 else f'  *({n}/3 HH)'
            print(f"  {label:<26} {mf:.3f} | {mtd:>+6.1f}/day | {mwl:>+6.1f}% | "
                  f"{mhd:>5.1f} harder/day{star}")


if __name__ == '__main__':
    main()
