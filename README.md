# Co-Adaptation Simulation

A closed-loop simulation system to evaluate proactive robot policies with realistic
human models. Every LLM prompt in the repo, rendered with real data:
`docs/llm_prompts.md`. A robot watches a person's object-relocation routine and pre-places
objects; the human reacts (skip / correct / undo / redirect); metrics charge the
robot for the human's burden. Our method is **GAP**; the code, run directories and
result files spell it `hybrid`, which is its original name. Code-faithful architecture
reference, including every result table: `DESIGN.md`.

## COMPONENTS

### Scripted human
Replays HOMER+ daily scripts (65 train + 10 test per household A/B/C). At each
script line, if the object is already at the line's destination the action is
**skipped** (robot helped); if it is somewhere wrong the human **corrects** it
(moves it to the scripted destination). No observe-time reactions — corrections
happen only when the script reaches the object. `environment/human_agent.py`.

### LLM human
A gpt-4o-mini agent that *audits* robot actions as they happen: it sees its day
plan and each robot move, then **undoes** a move it dislikes, **redirects** an
almost-right move straight to the planned destination (the later script line is
then satisfied), or accepts. Harsher and more realistic than the scripted human —
every unjustified robot move gets billed. This auditing behaviour ("redirect") is the
only LLM human we report. Enable: `HUMAN_POLICY=llm LLM_OBSERVE_MODE=redirect`. `agents/llm_human_prompt.py`, config `human.llm`.

### Plain-LLM robot (baseline)
gpt-4o-mini (config default) sees the raw past-14-days observation text, today's
moves (or a morning-idle message before the first human relocation), and current
time — **every 10-min tick**. Directly predicts up to 10 relocations. No structure,
no learning; filters unmapped classes, unplaceable `home`, and already-at-destination.
Strong on the easy cell (act+reset/scripted f1 0.398), collapses whenever the human
audits or the world drifts (0.14–0.18, adds 40+ repair trips/day).
`agents/robot_policy_plainllm.py`.

### Oracle robot (P0 ceiling, measurement instrument — not a method)
Reads the day's ground-truth script by design and pre-places under the same
physics (10-min ticks, per-object line ordering, same reconciliation). Its f1
decomposes failure: `1 − oracle_f1` = structurally unclaimable (chains with no
free tick), `oracle_f1 − method_f1` = the method's true deficit. Pure ceiling
only — no policy-style guards.
`agents/robot_policy_oracle.py`.

### SLaTe-PRO robot (prior work baseline)
Graph-tensor sequence model from the SLaTe-PRO paper predicting next object
locations from world-state history. Solid open-loop, degrades closed-loop
(self-conditioning ping-pong, OOD destination dispersion — findings H1) and under
drift. Emits all above-threshold relocations each tick (no repetition-suppression
window). `agents/robot_policy_slatepro.py`, `external/SLaTe-PRO/`.

### NOPA robot (prior work baseline)
Puig et al., ICRA 2023. The only baseline here that reasons about **goal
uncertainty** rather than forecasting the next state: a goal proposal network
samples K candidate day-goals from `(day-start state, current state, clock)`, an
inverse-planning particle filter rejects the goals inconsistent with the human's
observed moves, and an uncertainty-aware planner scores candidate subgoals by
expected time saved, helper cost, and state disturbance — executing the argmax,
one subgoal per tick. Self-contained: no shared machinery with GAP.
The paper's MCTS rollout becomes a sampled arrival-clock plan (nothing in this
simulator charges a navigation cost) and its ablations (OursRG / Ours-InvPlan /
Ours-Return / HPGPN) are flags. `agents/robot_policy_nopa.py`,
`model/nopa_gpn.py`, `DESIGN.md` §10a.

### GAP (ours)
The method the paper reports. Named `hybrid` throughout the code and in every run
directory and result filename.
Train time (LLM used here only): a gpt-4o **curator** segments warmup days into an
open-vocabulary activity memory (no GT labels); code builds branching **task
graphs** over (object, destination) nodes; a 250k-param **activity transformer**
(frozen language embeddings of curated names) predicts the next activity + start
time for the **prep branch**.

Eval time (zero LLM calls): a **particle filter over the task graphs**
does online recognition — particles carry completed-activity history plus OPEN
activities with graph position markers; per-event likelihood mixes
continue/open-new/background with best-path partial credit, same-room partial credit
(`ROOM_MISS_DISCOUNT=0.1`), and clock priors; robot placements are virtual
observations (stochastic marker advance, re-armable nodes; human undos → reactions).
**Actuation runs every tick**: PF-weighted “current” candidates plus transformer
prep (including morning idle before any human move), gated by threshold/cap/precedence/
learned per-key lead times. Eval-time LLM tagger/ranker removed.
Offline activity recognition is 92–99%, against 49–73% for the eval-time LLM tagger it
replaced.

**Clean continual learning** (`HYBRID_CL=1` / `--hybrid_cl`): per-day decay γ on graph
counts, chosen each retrain round by a prequential backtest on training data only
(logged to `gamma_log.jsonl`, plotted by `scripts/plot_gamma.py`); robot events from
eval `attributed_days/` joined at retrain with reverted/superseded dropped; actor-split
node counts (`count_robot` → `scripts/plot_task_graphs.py`); recency-weighted
transformer warm-start mirroring γ (ablate with `HYBRID_NO_TR_RECENCY=1`).
`agents/activity_particle_filter.py`, `agents/robot_policy_hybrid.py`,
`agents/task_graph.py`, `agents/llm_memory_curator.py`, `model/activity_transformer.py`.

## METRICS

Computed identically for all conditions by `scripts/analyze_llm_human_conditions.py`:

- **human_f1**: TP = skipped+accepted, FP = corrected+corrected_despite_correct+
  vigilance_corrections+redirected, FN = done_as_is (excludes done_early,
  no_movement, no_need_to_move, *_repeated, unseen robot churn).
- **Effort**: net trips saved/day (= skipped+accepted − vigilance_corrections) and
  % of solo workload removed; corrections/redirects are trip-neutral, undos
  (vigilance) are caused trips.
- **Actions made harder/day**: corrected+corrected_despite_correct+redirected —
  trip-neutral but perturbed (human fetches from the robot's wrong spot).

Paper-style 1-step SLaTe-PRO tensor metrics in `benchmark_*.json` are emitted
**only** when `suppress_robot_action=True` (the offline comparison); closed-loop runs leave
`slatepro` / `reconciliation` null. See `DESIGN.md` §2.5.

**Shapes**: *act+reset* = train once on 65 warmup days, 10 eval days, env reset
daily. *retrain* = 5 warmup days, retrain every 5 days, 70 eval days, no reset
(drifting world). Compare ratios, not raw counts, across shapes.

---

## HOW-TOs

All runs go through SLURM (`sbatch`). The wrappers resolve the repo root
themselves (`REPO_ROOT`, falling back to `SLURM_SUBMIT_DIR` — submit from the
repo root) and take the interpreter from `$COADAPT_PYTHON`, defaulting to
whatever `python` is on PATH. Set it once:

```bash
export COADAPT_PYTHON=/path/to/your/envs/coadapt/bin/python
# SLaTe-PRO pretraining uses its own env: export SLATEPRO_PYTHON=...
```

Each wrapper
auto-resubmits on the 4 h time limit and writes `.done_<HH>` markers on success
(NOTE: retrain runs write `benchmark_*.json` incrementally — JSON existence does
not mean finished; trust the marker). The `visualize_comparison.py` ValueError
after a complete JSON is cosmetic.

### GAP (ours) — the commands that produced the paper cells

`run_benchmark_hybrid_packed.sh` runs households A/B/C inside one job and writes
a `.done_<HH>` marker per household. Every cell below is 3 households.

```bash
# Main table, scripted user
SETTING=retrain GATE=1 BUILDON=1 PFPBG=0.001 NOLEADGATE=1 \
OUT_BASE=results/<base> CURCACHE=cache/llm_curator/<tag> \
sbatch --gres=gpu:a40:1 -J gap_scr scripts/run_benchmark_hybrid_packed.sh

# Main table, LLM user: add
#   HUMAN_POLICY=llm            (observe mode defaults to redirect)

# act+reset (distribution shift at test time only)
SETTING=actreset GATE=1 BUILDON=1 PFPBG=0.001 NOLEADGATE=1 ...

# Ablations: drop exactly one component from the command above
#   - particle filter   PFSINGLE=1
#   - precedence gate   NOPREC=1
#   - prep branch       NOPREP=1
#   - acceptance gate   omit GATE and BUILDON entirely, add RUNTAG=noacceptgate
#   - build-upon credit keep GATE=1, omit BUILDON,  add RUNTAG=nobuildon
# RUNTAG is required whenever a variant is defined by the ABSENCE of a knob:
# the output dir name is built from the knobs that ARE set, so without it the
# run would land in the same directory as the full method.

# Confidence-threshold sweep (figures/cthresh_sweep.png): add PROBTH=<v>.
# c_thresh 0.1 is the config default, so the main run above IS the 0.1 point.
```

Cache rule: parallel jobs must NEVER share an LLM cache dir (NFS SQLite) — give
each job its own copy of `cache.db`.

### Oracle (P0 ceiling)

```bash
# act+reset ceiling
OUT_DIR=results/<cond> HOUSEHOLD=A WARMUP_DAYS=65 TEST_DAYS=0 \
TRAIN_EVERY_N_DAYS=99999 ALL_SCRIPTS=1 \
sbatch -J <name> scripts/run_benchmark_oracle.sh
# retrain shape: WARMUP_DAYS=5 TRAIN_EVERY_N_DAYS=5 NO_RESET_ENV=1
# LLM human: HUMAN_POLICY=llm LLM_OBSERVE_MODE=redirect LLM_CACHE_DIR=...
```

CPU-only, no API cost; a full act+reset cell finishes in ~10 min.

### Plain-LLM baseline

```bash
OUT_DIR=results/<cond> HOUSEHOLD=A WARMUP_DAYS=65 TEST_DAYS=0 \
TRAIN_EVERY_N_DAYS=99999 ALL_SCRIPTS=1 \
PLAINLLM_CACHE_DIR=cache/llm_plainllm/<cond>_A \
sbatch -J <name> scripts/run_benchmark_plainllm.sh
```

CPU-only (no GPU); LLM-human / retrain variants via the same env vars as above.

### SLaTe-PRO baseline

```bash
HOUSEHOLD=A OUT_DIR=results/<cond> sbatch scripts/run_benchmark_slatepro.sh
# training (one-time): sbatch scripts/run_pretrain_slatepro.sh
```

### Analysis & diagnostics

```bash
# the one table: f1 + effort (+trips/d, wl%, hard/d) for every condition
python scripts/analyze_llm_human_conditions.py

# recognition / anticipation / filtering breakdown for a hybrid-family run
python scripts/analyze_hybrid_errors.py \
    --memory_dir results/<run>/hybrid_memory_A --household A

# offline recognition test for the particle filter (clean replay vs GT schedules)
python tests/pf_recognition_eval.py \
    --memory_dir results/<run>/hybrid_memory_A --household A \
    [--drop 0.2 --noise 0.15]

# CL gamma trajectory plot (memory half-life per retrain round)
python scripts/plot_gamma.py --memory_dir results/<run>/hybrid_memory_A

# task-graph visualization
python scripts/plot_task_graphs.py --memory_dir <...> --min_freq 0.25 --min_p 0.1

# the two shipped figures, both rebuilt from results_final/ alone
python scripts/plot_over_time.py        # figures/over_time/retrain_{scripted,llm}.png
python scripts/plot_cthresh_sweep.py    # figures/cthresh_sweep.{png,pdf} + its CSVs
```

Results conventions: every condition lives in `results/<cond>_<shape-suffix>/`
with `benchmark_{policy}_<HH>.json` (e.g. `benchmark_hybrid_A.json`); the analyzer
knows conditions by prefix (see `CONDITIONS` in `analyze_llm_human_conditions.py`).
The method-comparison tables, the ablations and the per-branch breakdown live in
`DESIGN.md`; the exported cells behind them are in `results_final/`.

---

## ARCHITECTURE OVERVIEW

```
./
├── environment/                  # VirtualHome-based world + human agent
├── agents/
│   ├── robot_policy.py           # abstract policy interface
│   ├── robot_policy_hybrid.py    # GAP (ours): particle-filter eval + train-time curation
│   ├── activity_particle_filter.py  # particle filter over the task graphs
│   ├── task_graph.py             # branching task graphs + prequential γ (CL)
│   ├── llm_memory_curator.py     # train-time LLM curation; actor attribution
│   ├── robot_policy_plainllm.py  # plain-LLM baseline (every-tick LLM)
│   ├── robot_policy_oracle.py    # P0 GT-script ceiling (pure, no guards)
│   └── robot_policy_slatepro.py  # SLaTe-PRO baseline
├── model/activity_transformer.py # next-activity + clock head (prep branch; CL warm-start)
├── scripts/                      # run_benchmark_*.sh wrappers + analysis
│   └── closed_loop_eval.py       # closed-loop harness (library)
├── config/base.yaml              # defaults (robot.policy: hybrid)
├── DESIGN.md                     # code-faithful architecture reference
└── simulation.py                 # human agent factory + legacy driver
```
