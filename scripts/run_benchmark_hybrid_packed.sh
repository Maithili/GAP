#!/bin/bash
#SBATCH --job-name=bench_hybridpack
#SBATCH --partition=rail-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=14
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/bench_hybridpack_%j.out
#SBATCH --error=logs/bench_hybridpack_%j.err
#SBATCH --exclude=gundam,qt-1,clippy,chitti,conroy,nestor

# Packed hybrid: run all 3 households (A,B,C) of ONE setting concurrently on a
# single shared a40. The activity transformer is ~250k params and eval is
# particle-filter / curator(gpt-4o) bound, so the GPU sits near-idle -> 3 procs
# share one GPU with negligible slowdown and no OOM. Each household gets its own
# out_dir / curator-cache / memory dir (mandatory: avoids shared-cache contention).
# Submit on --qos=long (long wall) so no USR1 resubmit logic is needed here.
#
# Usage:  SETTING=actreset|retrain  sbatch --qos=long --time=1-00:00:00 scripts/run_benchmark_hybrid_packed.sh

set -uo pipefail
PYTHON="${COADAPT_PYTHON:-python}"
SETTING=${SETTING:?set SETTING=actreset|retrain}
OUT_BASE=${OUT_BASE:-full_results}
TRAIN_DAYS=${TRAIN_DAYS:-65}
HUMAN_POLICY=${HUMAN_POLICY:-}    # ''=scripted | 'llm'=gpt-4o-mini LLM human
LLM_OBSERVE_MODE=${LLM_OBSERVE_MODE:-}    # 'redirect' (v2; ONLY human we report) | 'undo_allow' (deprecated)
# redirect is the only human we care about: default any LLM-human run to it so a run
# can never silently use the superseded undo_allow mode. Scripted runs stay untagged.
if [ -n "$HUMAN_POLICY" ] && [ -z "$LLM_OBSERVE_MODE" ]; then
  LLM_OBSERVE_MODE=redirect
fi
GATE=${GATE:-}    # non-empty -> add --hybrid_accept_gate; tags dirs + caches *_gate
PROBTH=${PROBTH:-}    # optional hybrid_prob_threshold override; tags out dir + caches *_pt<v>
CURCACHE=${CURCACHE:-}    # optional literal curator-cache base (per-HH: "<base>_<HH>"); reuse a populated cache
BUILDON=${BUILDON:-}    # non-empty -> --hybrid_accept_buildon (gate v2: accept only build-upon); tags llm cache *_v2
PFPBG=${PFPBG:-}        # optional fixed PF background rate p_bg (ablation); tags out dir *_pbg<v>
NOLEADGATE=${NOLEADGATE:-}  # non-empty -> --hybrid_no_lead_gate (ablation: remove the lead-time/too-early gate); tags out dir *_noleadgate
NOPREC=${NOPREC:-}          # non-empty -> --hybrid_no_precedence_gate (ablation: remove the precedence ordering gate); tags out dir *_noprec
NOPREP=${NOPREP:-}          # non-empty -> --hybrid_no_prep (ablation: reactive only; also the effective no-transformer ablation); tags out dir *_noprep
RUNTAG=${RUNTAG:-}          # free-form suffix appended to BOTH the out dir and the
                            # human-LLM cache dir. Needed when a variant is defined by
                            # the ABSENCE of a knob (e.g. GATE= / BUILDON=), since the
                            # dir names are built from present knobs only and would
                            # otherwise collide with an existing run.
PFSINGLE=${PFSINGLE:-}      # non-empty -> --hybrid_pf_single_belief (remove PF: 1 deterministic belief); tags out dir *_pfsingle

# act+reset = isolate model quality (one train round, reset env, CL off/moot).
# retrain   = online adaptation under drift (train every 5d, no reset, CL on).
case "$SETTING" in
  actreset) WARMUP=65; EVERY=99999; NORESET="";  CL="";  OUTNAME="hybrid_actreset" ;;
  retrain)  WARMUP=5;  EVERY=5;     NORESET="1"; CL="1"; OUTNAME="hybrid_retrain" ;;
  *) echo "ERROR: bad SETTING='$SETTING' (want actreset|retrain)"; exit 1 ;;
esac

# Resolve the repo root without hardcoding it. Under sbatch the script is
# spooled, so $BASH_SOURCE does not point into the repo -- SLURM_SUBMIT_DIR
# does (submit from the repo root). REPO_ROOT can also be set explicitly.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs

echo "=== Packed hybrid [$SETTING] on ${SLURMD_NODENAME:-?}  GPU(CUDA_VISIBLE_DEVICES)=${CUDA_VISIBLE_DEVICES:-?} ==="
echo "warmup=$WARMUP every=$EVERY noreset=${NORESET:-off} cl=${CL:-off}"

declare -A PID OUTDIR
for HH in A B C; do
  OD="$OUT_BASE/${OUTNAME}_warmup${WARMUP}_test0_every${EVERY}_all1_noreset${NORESET:-off}_modeevent${HUMAN_POLICY:+_human$HUMAN_POLICY}${LLM_OBSERVE_MODE:+_$LLM_OBSERVE_MODE}${GATE:+_gate}${PROBTH:+_pt$PROBTH}${PFPBG:+_pbg$PFPBG}${NOLEADGATE:+_noleadgate}${NOPREC:+_noprec}${NOPREP:+_noprep}${PFSINGLE:+_pfsingle}${RUNTAG:+_$RUNTAG}"
  OUTDIR[$HH]="$OD"
  if [ -f "$OD/.done_$HH" ]; then
    echo "HH $HH: marker exists ($OD/.done_$HH) — skipping"
    continue
  fi
  mkdir -p "$OD"
  $PYTHON scripts/benchmark_comparison.py \
    --household "$HH" \
    --train_days "$TRAIN_DAYS" \
    --out_dir "$OD" \
    --run_hybrid --skip_slatepro --skip_noop \
    --time_step_minutes 10 \
    --all_scripts \
    --warmup_days "$WARMUP" \
    --test_days 0 \
    --train_every_n_days "$EVERY" \
    ${NORESET:+--no_reset_env} \
    ${CL:+--hybrid_cl} \
    --hybrid_memory_dir "$OD/hybrid_memory_$HH" \
    --hybrid_llm_cache_dir "${CURCACHE:-cache/llm_curator/${OUTNAME}${GATE:+_gate}${PROBTH:+_pt$PROBTH}}_$HH" \
    ${PROBTH:+--hybrid_prob_threshold "$PROBTH"} \
    ${PFPBG:+--hybrid_pf_p_bg "$PFPBG"} \
    ${NOLEADGATE:+--hybrid_no_lead_gate} \
    ${NOPREC:+--hybrid_no_precedence_gate} \
    ${NOPREP:+--hybrid_no_prep} \
    ${PFSINGLE:+--hybrid_pf_single_belief} \
    ${BUILDON:+--hybrid_accept_buildon} \
    ${HUMAN_POLICY:+--human_policy "$HUMAN_POLICY"} \
    ${HUMAN_POLICY:+--llm_model gpt-4o-mini} \
    ${HUMAN_POLICY:+--llm_max_usd_per_day 20} \
    ${LLM_OBSERVE_MODE:+--llm_observe_mode "$LLM_OBSERVE_MODE"} \
    ${GATE:+--hybrid_accept_gate} \
    ${HUMAN_POLICY:+--llm_cache_dir "cache/llm_human/${OUTNAME}_human${HUMAN_POLICY}${LLM_OBSERVE_MODE:+_$LLM_OBSERVE_MODE}${GATE:+_gate}${PROBTH:+_pt$PROBTH}${BUILDON:+_v2}${PFSINGLE:+_pfsingle}${NOPREP:+_noprep}${NOPREC:+_noprec}${NOLEADGATE:+_noleadgate}${RUNTAG:+_$RUNTAG}_$HH"} \
    > "logs/hybridpack_${SETTING}_${HUMAN_POLICY:-scripted}_${HH}_${SLURM_JOB_ID:-0}.out" 2>&1 &
  PID[$HH]=$!
  echo "HH $HH: launched pid ${PID[$HH]} -> $OD (log logs/hybridpack_${SETTING}_${HH}_${SLURM_JOB_ID:-0}.out)"
done

# Wait for each household; write its .done marker only on clean exit.
rc=0
for HH in A B C; do
  [ -z "${PID[$HH]:-}" ] && continue
  if wait "${PID[$HH]}"; then
    touch "${OUTDIR[$HH]}/.done_$HH"
    echo "HH $HH: DONE (marker written)"
  else
    ec=$?
    echo "HH $HH: FAILED (exit $ec) — see logs/hybridpack_${SETTING}_${HH}_${SLURM_JOB_ID:-0}.out"
    rc=1
  fi
done
echo "=== packed hybrid [$SETTING] finished rc=$rc ==="
exit $rc
