#!/usr/bin/env python
"""Rebuild results_final/ from the runs that back the paper tables.

Each entry below names the paper cell and the run directory it comes from, so
the mapping is explicit rather than implied by directory globs. A cell is only
exported when all three households carry a .done_<HH> marker -- the benchmark
jsons are written incrementally, so their existence does not mean a run
finished (see README).

Per cell we ship, matching build_public_results.sh:
  results_<HH>.json     slim `totals` block (every reported number)
  checkpoints_<HH>/     the FINAL retrain round's weights only
  memory_<HH>/          curator artifacts (hybrid family only)
  llm_audit_<HH>.txt.gz full transcripts, gzipped (~45x)

Run: python scripts/export_results_final.py [--dry-run]
"""
import argparse, glob, gzip, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_slim_results import extract_totals

# Every source that backs a paper cell lives in full_results/. That tree is local
# working data: it is gitignored, and it is kept as one tarball per cell outside
# the repo (see that archive's README for the rsync + restore commands). Only the
# export below is committed, so a fresh checkout reports "source dir missing"
# until the tarballs are restored.
RR = "full_results"
HV = f"{RR}/hybrid_retrain_warmup5_test0_every5_all1_noreset1_modeevent"
HVA = f"{RR}/hybrid_actreset_warmup65_test0_every99999_all1_noresetoff_modeevent"
NT = RR
# NOPA LLM-human retrain has NOT been re-run with corrected retraining yet; this
# source is the frozen-training run (invalid) and stays outside rerun_Jun15.
SP = f"{RR}/slatepro_noPrivilege_warmup5_test0_every5_look1_eval100000_multidefault_threshold0.0_suppressoff_generateoff_all1_train_every_n_days5_noreset1_vigoff_addedBad100"
SK = f"{RR}/streak_noPrivilege_invban_cll2_warmup5_test0_every5_look3_threshold0.0_addedBad100"

# paper cell -> (source dir, policy name in the json filenames)
CELLS = {
    # main table: ours, lead-time gate REMOVED
    "hybrid_scripted":            (f"{HV}_gate_pbg0.001_noleadgate", "hybrid"),
    "hybrid_llm_human":           (f"{HV}_humanllm_redirect_gate_pbg0.001_noleadgate", "hybrid"),
    # ablations, all against the same baseline
    "ablation_noAcceptGate_scripted":  (f"{HV}_pbg0.001_noleadgate_noacceptgate", "hybrid"),
    "ablation_noAcceptGate_llm_human": (f"{HV}_humanllm_redirect_pbg0.001_noleadgate_noacceptgate", "hybrid"),
    "ablation_noPF_scripted":          (f"{HV}_gate_pbg0.001_noleadgate_pfsingle", "hybrid"),
    "ablation_noPF_llm_human":         (f"{HV}_humanllm_redirect_gate_pbg0.001_noleadgate_pfsingle", "hybrid"),
    "ablation_noPrecGate_scripted":    (f"{HV}_gate_pbg0.001_noleadgate_noprec", "hybrid"),
    "ablation_noPrecGate_llm_human":   (f"{HV}_humanllm_redirect_gate_pbg0.001_noleadgate_noprec", "hybrid"),
    "ablation_noPrep_scripted":        (f"{HV}_gate_pbg0.001_noleadgate_noprep", "hybrid"),
    "ablation_noPrep_llm_human":       (f"{HV}_humanllm_redirect_gate_pbg0.001_noleadgate_noprep", "hybrid"),
    "ablation_noBuildOn_scripted":     (f"{HV}_gate_pbg0.001_noleadgate_nobuildon", "hybrid"),
    "ablation_noBuildOn_llm_human":    (f"{HV}_humanllm_redirect_gate_pbg0.001_noleadgate_nobuildon", "hybrid"),
    # act+reset table
    "actreset_hybrid_scripted":   (f"{HVA}_gate_pbg0.001_noleadgate", "hybrid"),
    # NOPA, task episodes + weighted_ce
    "nopa_scripted":              (f"{NT}/nopa_retrain_warmup5_test0_every5_K20_tprop15_wm5.0_max1_weighted_ce", "nopa"),
    "nopa_llm_human":             (f"{NT}/nopa_retrain_warmup5_test0_every5_K20_tprop15_wm5.0_max1_weighted_ce_humanllm_redirect", "nopa"),
    "actreset_nopa_scripted":     (f"{NT}/nopa_actreset_warmup65_test0_every99999_K20_tprop15_wm5.0_max1_weighted_ce", "nopa"),
    # main-table baselines
    "slatepro_scripted":          (f"{SP}", "slatepro"),
    "slatepro_llm_human":         (f"{SP}_humanllm_gpt-4o-mini_redirect", "slatepro"),
    "streak_scripted":            (f"{SK}", "streak"),
    "streak_llm_human":           (f"{SK}_max2_humanllm", "streak"),
    "oracle_scripted":            (f"{RR}/oracle_retrain_warmup5_test0_every5_all1_noreset1", "oracle"),
    "oracle_llm_human":           (f"{RR}/oracle_retrain_warmup5_test0_every5_all1_noreset1_humanllm_redirect", "oracle"),
    # Plain LLM: the runs contaminated by the shared observed-dataset dir are
    # deleted; both cells come from the corrected reruns (one observed-dataset
    # dir per household), finished 2026-11-20.
    "plainllm_scripted":          (f"{RR}/plainllm_retrain_warmup5_test0_every5_all1_noreset1", "plainllm"),
    "plainllm_llm_human":         (f"{RR}/plainllm_retrain_warmup5_test0_every5_all1_noreset1_humanllm_redirect", "plainllm"),
    # act+reset baselines (test-time distribution shift only)
    "actreset_slatepro_scripted": (f"{RR}/slatepro_actreset_warmup65_test0_every99999_look1_eval100000_multidefault_threshold0.0_suppressoff_generateoff_all1_train_every_n_days99999_noresetoff_vigoff", "slatepro"),
    "actreset_streak_scripted":   (f"{RR}/streak_noPrivilege_invban_cll2_warmup65_test0_every99999_look3_threshold0.0_addedBad100_actreset", "streak"),
    "actreset_plainllm_scripted": (f"{RR}/plainllm_actreset_warmup65_test0_every99999_all1_noresetoff", "plainllm"),
    "actreset_oracle_scripted":   (f"{RR}/oracle_actreset_warmup65_test0_every99999_all1_noresetoff", "oracle"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="results_final")
    a = ap.parse_args()
    done, skipped = [], []
    for cell, (src, pol) in sorted(CELLS.items()):
        if not os.path.isdir(src):
            skipped.append((cell, "source dir missing")); continue
        if len(glob.glob(f"{src}/.done_?")) != 3:
            skipped.append((cell, f"{len(glob.glob(f'{src}/.done_?'))}/3 markers")); continue
        dst = os.path.join(a.out, cell)
        if a.dry_run:
            done.append((cell, "would export")); continue
        # Rebuild from scratch so no file from an older build (different source,
        # checkpoint day or heatmap set) survives next to the new export.
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)
        for HH in "ABC":
            jf = f"{src}/benchmark_{pol}_{HH}.json"
            if not os.path.isfile(jf):
                continue
            t = extract_totals(jf)
            if t is None:
                skipped.append((cell, f"HH-{HH} totals unreadable")); continue
            json.dump({"totals": t}, open(f"{dst}/results_{HH}.json", "w"))
            ck = f"{src}/model_checkpoints_benchmark_{pol}_{HH}"
            ws = sorted(glob.glob(f"{ck}/weights_day*.pt"),
                        key=lambda p: int(p.split("weights_day")[1].split(".pt")[0]))
            if ws:
                os.makedirs(f"{dst}/checkpoints_{HH}", exist_ok=True)
                shutil.copy2(ws[-1], f"{dst}/checkpoints_{HH}/{os.path.basename(ws[-1])}")
            mem = f"{src}/hybrid_memory_{HH}"
            if os.path.isdir(mem):
                shutil.copytree(mem, f"{dst}/memory_{HH}", dirs_exist_ok=True)
            aud = f"{src}/llm_audit_benchmark_{pol}_{HH}.txt"
            if os.path.isfile(aud):
                with open(aud, "rb") as fi, gzip.open(f"{dst}/llm_audit_{HH}.txt.gz", "wb", 9) as fo:
                    shutil.copyfileobj(fi, fo)
        done.append((cell, "exported"))
    print(f"exported {len(done)} cells:")
    for c, m in done: print(f"  {c:<34} {m}")
    if skipped:
        print(f"\nSKIPPED {len(skipped)} (not marker-complete / missing):")
        for c, m in skipped: print(f"  {c:<34} {m}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
