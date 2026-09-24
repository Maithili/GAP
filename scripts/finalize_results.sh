#!/bin/bash
# Move (not copy) the 10 reported result dirs into results/final/[method]_[human].
# mv = instant rename on the same filesystem: checkpoints ride along, nothing is
# duplicated. STREAK LLM = the max2 (capped) run (not the uncapped one).
# Idempotent: skips a mapping whose source is already gone / dest already exists.
set -uo pipefail
# Resolve the repo root without hardcoding it. Under sbatch the script is
# spooled, so $BASH_SOURCE does not point into the repo -- SLURM_SUBMIT_DIR
# does (submit from the repo root). REPO_ROOT can also be set explicitly.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"
DEST=results/final
mkdir -p "$DEST"

# dest-name : source-dir   (one per reported table cell)
map() {
cat <<'EOF'
slatepro_scripted|full_results/slatepro_noPrivilege_warmup5_test0_every5_look1_eval100000_multidefault_threshold0.0_suppressoff_generateoff_all1_train_every_n_days5_noreset1_vigoff_addedBad100
slatepro_llm_human|full_results/slatepro_noPrivilege_warmup5_test0_every5_look1_eval100000_multidefault_threshold0.0_suppressoff_generateoff_all1_train_every_n_days5_noreset1_vigoff_addedBad100_humanllm_gpt-4o-mini_redirect
streak_scripted|full_results/streak_noPrivilege_invban_cll2_warmup5_test0_every5_look3_threshold0.0_addedBad100
streak_llm_human|full_results/streak_noPrivilege_invban_cll2_warmup5_test0_every5_look3_threshold0.0_addedBad100_max2_humanllm
plainllm_scripted|full_results/plainllm_retrain_warmup5_test0_every5_all1_noreset1
plainllm_llm_human|full_results/plainllm_retrain_warmup5_test0_every5_all1_noreset1_humanllm_redirect
hybrid_scripted|results/hybrid_v2/hybrid_retrain_warmup5_test0_every5_all1_noreset1_modeevent_gate_pbg0.001
hybrid_llm_human|results/hybrid_v2/hybrid_retrain_warmup5_test0_every5_all1_noreset1_modeevent_humanllm_redirect_gate_pbg0.001
oracle_scripted|full_results/oracle_retrain_warmup5_test0_every5_all1_noreset1
oracle_llm_human|full_results/oracle_retrain_warmup5_test0_every5_all1_noreset1_humanllm_redirect
EOF
}

moved=0; skipped=0
while IFS='|' read -r name src; do
  [ -z "$name" ] && continue
  dst="$DEST/$name"
  if [ -e "$dst" ]; then echo "SKIP  $name (dest exists)"; skipped=$((skipped+1)); continue; fi
  if [ ! -d "$src" ]; then echo "SKIP  $name (source missing: $src)"; skipped=$((skipped+1)); continue; fi
  mv "$src" "$dst" && { echo "MOVED $name  <- $src"; moved=$((moved+1)); }
done < <(map)
echo "=== done: moved=$moved skipped=$skipped -> $DEST ==="
ls -la "$DEST"
