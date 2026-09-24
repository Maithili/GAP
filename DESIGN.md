# Co-Adaptation Simulation — Design

Code-faithful architecture reference for the closed-loop benchmark. Every mechanism
below is traced to the implementation files named inline. When behavior differs
between drivers (e.g. `simulation.py` vs `closed_loop_eval.py`), both are noted.

---

## 1. Purpose and entry points

The benchmark studies **human–robot co-adaptation** in a HOMER+ household: a robot
pre-places objects; a human follows a daily relocation script and reacts to robot
moves. Outcomes are tagged per interaction and rolled up into **human burden**
(`human_f1`) and robot prediction metrics.


| Entry                                                           | Role                                                                             |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `scripts/benchmark_comparison.py`                               | CLI driver: script discovery, config wiring, dispatches policies                 |
| `scripts/closed_loop_eval.py::evaluate_closed_loop`             | Main closed-loop harness (library module, no standalone CLI)                     |
| `scripts/run_benchmark_slatepro.sh` / `run_benchmark_hybrid_packed.sh` / `run_benchmark_nopa.sh` | SLURM wrappers: env vars → `benchmark_comparison.py`                             |
| `simulation.py`                                                 | Legacy single-day loop; human-first tick order; not used for reported benchmarks |


Policy selection: `config['robot']['policy']` → `slate_pro` | `streak` | `hybrid` |
`plain_llm` | `oracle` | `nopa` | `noop`. Human selection: `config['human']['policy']` → `scripted` | `llm`
via `simulation.py::build_human_agent`.

Defaults live in `config/base.yaml` (`robot.policy` default: `hybrid`); CLI flags in
`benchmark_comparison.py` override them.

---

## 2. Closed-loop simulation

### 2.1 Script pool and day phases

`evaluate_closed_loop` iterates `script_files` (one `.txt` script per simulated day).

**Phase split** — `script_distribution = [warmup_days, finetune_days_or_-1, test_days]`:


| Index | Phase          | Behavior (`closed_loop_eval.py` ~788–791)                                         |
| ----- | -------------- | --------------------------------------------------------------------------------- |
| `[0]` | **warm-start** | Robot updates history only; no robot actions; builds `gen_*.pt` training datasets |
| `[1]` | **finetune**   | Robot acts; human accommodates; `robot_policy.train()` at periodic boundaries     |
| `[2]` | **test**       | Robot acts; no further training                                                   |


If one entry is `-1`, it is replaced by `len(script_files) - sum(other entries)`.
`phase_ends` maps each phase name to the **exclusive** end day index (cumulative sum).

Phase label advances at day start when `day_idx >= phase_ends[phase]` (~911–912).

**Benchmark default** (`benchmark_comparison.py` ~268): `[warmup_days, -1, test_days]`
so finetune length = total scripts − warmup − test.

### 2.2 World and day boundaries

At harness start (~795–798):

- `World` is constructed from **day 0's** script companion files only:
  - Initial edges: `processed_seqLM_coarse/{day0}.pt`
  - Initial graph: `routines_{day0}.json` → `graphs[0]`

Each day (~906–944):

1. `world.reset()` if `reset_env=True` (~909–910) — rebuilds graph from that same
  `init_graph` copy (`environment/world.py::reset`), **not** from the current day's
   routine JSON.
2. Phase transition (~911–912).
3. **New human agent** from today's script: `build_human_agent(cfg, ..., script_path)` (~919–921).
4. `robot_policy.reset()` (~943–944) — clears policy history / day state.

When `reset_env=False` (`--no_reset_env` / `NO_RESET_ENV=1`), world state persists;
resume checkpoints store `world_state_dict` (~~1564–1565) and restore it on resume (~~836–838).

### 2.3 Tick loop (robot then human)

Within each day, time runs from `active_start_minutes` to `min(max_time, active_end_minutes)`
stepping by `simulation.time_step_minutes` (default 10). The harness library defaults
`active_start_minutes=0`; `**benchmark_comparison.py` defaults to 360** (6 AM) unless
overridden via `--active_start_minutes`.

**Per tick** (`closed_loop_eval.py` ~1058–1330):

```
1. ROBOT
   - warm-start: robot_policy._update_history(world_state) only; no actions
   - finetune/test: robot_policy.get_actions(node_map, world_state=...)
   - For each emitted RobotMoveAction:
       a. Append to pending_robot_actions_evaluation
       b. human.review_robot_action(...) → undo | redirect | allow
       c. If undo/redirect: execute corrective human script action; skip robot execute
       d. Else if not suppress_robot_action and not warm-start: world.execute_action(robot TELEPORT)
2. HUMAN
   - human.get_actions(current_time, world_state,
       robot_actions=pending_robot_actions_evaluation, ...)
   - world.set_human_activity(human.get_human_activity())
   - Execute each returned script action via world.execute_action
3. Reconcile pending robot eval tags (corrected / skipped / unseen / etc.)
4. Accumulate dataset tensors + optional world trace
5. current_time += time_step_minutes
```

**Accommodation vs suppression:** the harness sets
`accommodate_for_robot = (not suppress_robot_action)` (~927). Closed-loop runs use
`suppress_robot_action=False` → the user skips script moves the robot has already
satisfied. The offline comparison in VI-B sets `suppress_robot_action=True` → robot moves
are scored but not executed, so the user always follows the script and the robot never
changes the state it is evaluated on.

**Warm-start LLM human** (~926–927): `human.llm_active = False` — defers to scripted
accommodation even when `policy=llm`.

**Vigilance** (~924): disabled during warm-start (`vigilance_enabled = False`).

### 2.4 Robot training schedule

Training fires **after a day completes**, not inside the tick loop (~1518–1550):

Conditions (all required):

- `next_day >= phase_ends['warm-start']` (finetune has started)
- `next_day < phase_ends['finetune']` (**no training in test phase**)
- `(next_day - phase_ends['warm-start']) % train_every_n_days == 0`

Calls `robot_policy.train(dataset_dir, model_ckpt_dir, days=str(day_idx))`.

Warm-start days write `gen_*.pt` files under `processed_seqLM_coarse/` (~1371–1437);
training reads that directory tree.

`train_every_n_days=99999` (act+reset shape) triggers effectively **one** train at the
first finetune boundary.

`use_prior_checkpoints=True`: skips warm-start simulation/dataset gen; loads
`weights_day{warm_end}.pt` up front (~859–902).

### 2.5 Metrics aggregation

`_aggregate_totals` (~1600+) rolls up **finetune** and **test** only; warm-start excluded.

**Human burden F1** (~1709–1714):


|     | Tags                                                                         |
| --- | ---------------------------------------------------------------------------- |
| TP  | `skipped + accepted`                                                         |
| FP  | `corrected + corrected_despite_correct + vigilance_corrections + redirected` |
| FN  | `done_as_is`                                                                 |


Excluded from F1: `done_early`, `no_movement`, `no_need_to_move`, `*_repeated`, robot
self-churn (`unseen`).

Also aggregates paper-style 1-step SLaTe-PRO metrics (`slatepro` block) and human-outcome ×
tensor-category confusion matrices (`reconciliation`) **only when**
`suppress_robot_action=True`. When off, those fields are `**null`** in the JSON —
closed-loop live state diverges from offline GT, so paper-style tensor reconciliation
would be invalid.

Output: incremental `benchmark_*.json`, optional resume `.pt`, per-checkpoint
`model_checkpoints_*/weights_day{N}.pt`.

---

## 3. The two evaluation conditions

The paper reports both: **drift during training and evaluation** (the main results) and
**drift at evaluation only** (paper Table VII). The second gives the robot 65 days of
passive observation before it may act, so it trains on data free of its own actions. Run
directories spell them `retrain` and `actreset`.

Configured via `benchmark_comparison.py` CLI / sweep env vars. Both use
`ALL_SCRIPTS=1`, `TEST_DAYS=0` in standard sweeps (finetune fills all post-warmup days).


|                      | **drift at evaluation only**    | **drift in training + eval**  |
| -------------------- | ------------------------------- | ----------------------------- |
| `WARMUP_DAYS`        | 65                              | 5                             |
| `TRAIN_EVERY_N_DAYS` | 99999 (train once)              | 5                             |
| `reset_env`          | True (default)                  | False (`NO_RESET_ENV=1`)      |
| World across days    | Reset to day-0 init graph daily | Persists (drifting world)     |
| Training             | Once at finetune start          | Every 5 finetune days         |
| Interpretation       | Isolate model quality           | Online adaptation under drift |


`--train_days` selects which scripts enter the pool when **not** using `--all_scripts`;
phase lengths come from `--warmup_days` and `--test_days` independently.

---

## 4. Human models

Selection: `simulation.py::build_human_agent` (~293–315).

- `human.policy == 'scripted'` → `HumanAgent`
- `human.policy == 'llm'` → forces `curiosity_vigilance.enabled = False`, then `LLMHumanAgent`

Both receive `accommodate_for_robot` (from `not suppress_robot_action` in the eval harness).

### 4.1 Scripted human (`environment/human_agent.py`)

**Accommodation** runs in `_execute_current_action` for **TELEPORT** lines only (~166–230).

Compares script destination to live world (or `pending_teleports` within the same tick).


| Situation                                  | Human outcome                                                                             | Robot pending tag |
| ------------------------------------------ | ----------------------------------------------------------------------------------------- | ----------------- |
| Already at dest, no robot match            | `no_need_to_move`                                                                         | —                 |
| Already at dest, robot matched (obj,dest)  | `skipped`                                                                                 | `skipped`         |
| Must move, robot matched exact (obj,dest)  | `skipped`; human TELEPORT dropped when `accommodate_for_robot=True` (closed-loop default) | `skipped`         |
| Must move, robot moved same obj wrong dest | `corrected`                                                                               | `corrected`       |
| Must move, no robot involvement            | `done_as_is`                                                                              | —                 |


Non-TELEPORT lines: `no_movement` in `script_line_treatments` only (~165–167); not added
to `action_outcomes`.

**Vigilance** (`review_robot_action`, ~302–324): if `curiosity_vigilance.enabled` and
Bernoulli draw succeeds, scans script within `correction_window_hours` for a matching
future TELEPORT; mismatch → `undo`. Used by `closed_loop_eval` for scripted runs when
vigilance is enabled; it is off in every reported cell.

### 4.2 LLM human (`environment/llm_human_agent.py`)

When `llm_active=True` (finetune/test in eval harness):

**Branch A — object already at script dest** (`_decide_at_dest`): LLM chooses `skip` or
`correct` → outcomes `skipped` / `corrected_despite_correct`.

**Branch B — object elsewhere** (`_decide_unexpected`): LLM chooses `correct`, `accept`, or
`skip` → `corrected`, `accepted`, or fallback to scripted path.

**Observe hook** (`review_robot_action`): replaces vigilance Bernoulli.


| `human.llm.observe_mode` | Decisions                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `undo_allow` (default)   | `undo` | `allow`                                                                                              |
| `redirect`               | `undo` | `allow` | `redirect` (redirect only to planned dest within `redirect_window_min`, floored at 90 min) |


`redirect` executes a human TELEPORT to the planned destination; tags robot outcome
`redirected`; registers `(obj, dest)` in `_moved_early` so later script line → `done_early`.

**Budget** (`_require_budget`): raises if live API calls today ≥
`max_llm_calls_per_day` or estimated USD ≥ `max_usd_per_day`. Cache hits do
not increment the call cap.

**Cache** (`agents/llm_human_prompt.py::LLMHumanClient`): disk cache keyed by
provider/model/temperature/schema/prompts.

Prompt builders and JSON schemas: `agents/llm_human_prompt.py`. A fully rendered example
of every prompt in the repo (curator, plain-LLM, and all three LLM-human prompts):
[docs/llm_prompts.md](docs/llm_prompts.md).

---

## 5. Robot policy contract

All benchmark policies subclass `agents/robot_policy.py::RobotPolicyBase`.

**Driver expects** (not fully reflected in base class docstring):

```python
get_actions(node_map, *, world_state) -> (
    list[RobotMoveAction],
    (obj_idxs, loc_idxs, prev_loc_idxs),  # slate row indices
)
reset()   # each day boundary
train(dataset_path, output_dir, days)   # finetune checkpoints
```

Shared history shim (SLaTe-format tensors): `history['edges']`, `activity_features`,
`time_features`, `times`, `sequence_length`, `vh_id_to_slate_idx`, etc. Built by
`agents/slatepro_utils.py` helpers.

`RobotMoveAction`: TELEPORT script line with VH `object_id` / `dest_id`.

---

## 6. SLaTe-PRO (`agents/robot_policy_slatepro.py`)

**Config** (`robot.slate_pro` in `base.yaml`):


| Key                    | Default | Use                                            |
| ---------------------- | ------- | ---------------------------------------------- |
| `confidence_threshold` | 0.0     | Min relocation confidence                      |
| `lookahead_steps`      | 3       | Autoregressive forward steps in `_run_forward` |
| `max_history_len`      | 144     | History window cap                             |
| `model_checkpoint`     | path    | Architecture + `common_data` discovery         |


Eval emits **all** above-threshold relocations (sorted by confidence); there is no
repetition-suppression window and no `multiple_actions` config knob.

### Init and weights

`_load_model_only_cfg` (~296–334): resolves `.ckpt` path for config discovery,
loads `common_data.json` and builds architecture from `model_configs*.json`.
Does **not** load tensor weights (random init until sim `train()` / resume).

### Eval (`get_actions`)

Every finetune/test tick:

1. `_update_history(world_state)`
2. `_run_forward()`:
  - Encode graph + activity latents with `time_context = time_features[:, 1:num_states]`
  - Loop `lookahead_steps`: `predict(latents, tc_pred)` with per-step shifted
  `time_context` (synthetic +Δt encodings when future GT times unavailable)
  - Track best destination + confidence per object across steps
  - `valid_relocations = changed & confidence ≥ threshold & dynamic`
3. Sort by confidence; emit every valid relocation (no cap beyond threshold/dynamic mask)

### Train

PyTorch Lightning `Trainer.fit` on `RoutinesDataset` from generated `gen_*.pt` dir;
`MAX_TRAIN_EPOCHS` env (default 450); EarlyStopping on `Val_ES_accuracy`; saves
`weights_day{days}.pt`.

---

## 7. STREAK (`agents/robot_policy_streak.py`)

Wraps STREAK's `GraphTranslatorModule` — a tiny graph-translator MLP (~8.8K params) doing
**stateless single-step** (prev→next) graph prediction — as a closed-loop robot policy.
Continual-learning variants (L2 / EWC) come from the `GraphTranslatorCL` subclass
(`external/STREAK/graph_translator_cl.py`); the HOMER+ → STREAK data conversion lives in
`external/STREAK/helpers/coadaptation_adapter.py` (`prepare_streak_dataset`).

**Config** (`robot.streak` in `base.yaml`):


| Key                    | Default        | Use                                                       |
| ---------------------- | -------------- | --------------------------------------------------------- |
| `model_checkpoint`     | null           | benchmarks always train from scratch                      |
| `train_from_scratch`   | true           | closed-loop warmup + retrain; do not set false for cells  |
| `confidence_threshold` | 0.0            | Min relocation confidence                                 |
| `lookahead_steps`      | 3              | Autoregressive forward steps in `_run_forward`            |
| `time_encoder`         | time_external  | 4-dim time context (vs 14-dim)                            |
| `cl_mode`              | none           | none \| l2 \| ewc — each `train()` round = one CL task     |
| `ewc_lambda`           | null           | reg strength; null → variant default (l2=200, ewc=100)    |


### Init and weights

- `common_data` from the checkpoint dir if given, else the HOMER+ household
  (`processed_seqLM_coarse/common_data.json`); sets `n_nodes` + `model_configs`.
- **Two masks** built at init: `nonstatic_edges` (static objects never move) and
  `valid_locations` (`build_valid_location_mask`: the superset of nodes ever used as a parent —
  blocks object-on-object stacking, the prior ancestor-cycle crash).
- `train_from_scratch=true` (default): random init until the sim's `train()` / resume.
- `_new_model`: `cl_mode='none'` → `GraphTranslatorModule`; l2/ewc → `GraphTranslatorCL`.
  Inference (`forward`) is identical across modes — CL only adds a training-time penalty.

### Eval (`get_actions` / `_run_forward`)

Every finetune/test tick, autoregressive over `lookahead_steps`:

1. Build input `(edges, nodes, context)` from `world_state`.
2. Each step: `model(...)` → `_apply_dynamic_mask` (static rows → −inf; **invalid-location
   columns → −inf**; softmax; static rows restored to current) → `argmax` = `pred_locs`,
   `max` = `conf`.
3. Step 0 sets `best_dest` + `changes`; later steps lock only the **first** new change per object
   and feed the masked probs back as the next input (synthetic +Δt time context).
4. `valid = changed & conf ≥ confidence_threshold & dynamic`; emit every valid relocation.

### Train (CL task windowing)

PyTorch-Lightning `Trainer.fit` on the STREAK-adapted dataset; `MAX_TRAIN_EPOCHS` env.

- **`cl_mode=none`**: plain `GraphTranslatorModule`, retrained on the data each round.
- **`cl_mode=l2|ewc`**: each `train()` round (every `train_every_n_days`) is **one task**.
  The task trains on only the **new window** (routines since the last task; `_cl_consumed`
  bookkeeping persisted for resume). After training task *t*, `consolidate()` snapshots the
  trained weights as reference ("star") params and — for EWC — estimates the diagonal Fisher
  over the task's data; `cl_t` advances.
  - Penalty (active only from the 2nd task, `cl_t > 0`; first task unregularized):
    `L2: loss += (λ/2)·Σ(p−p_ref)²` · `EWC: loss += (λ/2)·Σ Fₚ(p−p_ref)²`, `Fₚ=E[(∂loss/∂p)²]`.
  - Clean reimplementation — the shipped `CGTML2`/`CGTM_F` are non-functional (Fisher never
    computed); CL state (`*_cl_mean`, `*_cl_fisher`, `cl_t`) lives in the state_dict and
    survives retrain rounds / resume.

L2 is the reported choice.

### Training-data noise (`ADDED_BAD_PERC`) and why AB100 ≫ AB0

`ADDED_BAD_PERC` (closed_loop_eval.py:1380–1416) gates which of the **robot's own** actions
get baked into the reconstructed training trajectory; the **human's** movements are always
included. `=0` ("skipped-only", the legacy default) drops the robot's non-skipped (corrected)
actions; `=100` keeps them. Counter-intuitively **AB100 works (+20–35% net scripted) and AB0
collapses** — not because AB0 is "cleaner". AB0 is **behavioral cloning on the expert path only**:
it erases the robot's mistakes from training, so STREAK is never shown the off-distribution
states its own errors create. The first error lands it in an unseen state with no learned
recovery → compounding-error covariate shift → the over-act/blowup/paralysis modes. AB100 is
**DAgger-shaped** (robot's own mistaken states + the human's correction `L→H` are both in the
trajectory) → it learns recovery → stays stable. So AB0 is a pathological ablation, not the
charitable baseline its name implies.

The three AB0 households land in three different failure modes, which is the clearest evidence
for the argument above: HH-A over-acts (380k phantom churn, recall 1.000, 91% perturbed), HH-B
sits in between (63k, recall 0.184), HH-C is effectively paralysed (17 churn events, 4 true
positives, recall 0.002). Averaging them is what produces the bland 6.0% net row. Artifacts:
`results_final/ab0_streak_scripted` (and `ab0_slatepro_{scripted,llm_human}`); the source runs
were deleted in the release cleanup.

### LLM-human collapse: the robot↔human churn loop

AB100 holds against the **scripted** human (deterministic, correction-only: 0 undo/redirect)
but collapses against the **LLM human** (`redirect` mode), where net craters to **−30% to
−1700%**. Two compounding causes: (1) the LLM human charges an **explicit undo cost** for every
mistake (scripted only "corrects" = free), so the robot is net-harmful immediately; (2) the LLM
human is **non-stationary**, so DAgger's recovery target is a moving target the Markovian policy
churns against and **amplifies in training** instead of damping. The result is a literal
per-tick fight — e.g. HH-A day39, the robot moves `cleaning_solution → headset` (a collapsed,
nonsense fixed prediction) **every 10-min tick from 06:00 on, and the human undoes it — 108×
that day** (one bounce per tick). The magnitude grows over training and each household ignites
on its own day (B and C around day 15, A around day 35) before pinning to the ceiling. A
per-object cooldown would cap the saturation without fixing the mis-prediction underneath it,
which is the argument for GAP learning per-placement acceptance instead (§10).

---

## 8. Plain LLM (`agents/robot_policy_plainllm.py`)

**Config** (`robot.plain_llm`): `model` (default `gpt-4o-mini`), `history_days`,
`max_predictions`, `max_usd`, `cache_dir`, `common_data_path`.

### Train

No gradient learning. Renders last `history_days` warmup `.pt` files into text via
`render_events`; saves `{history_text, n_days}` checkpoint.

### Eval

Every tick (each 10-min step from `closed_loop_eval`):

1. `_update_history(world_state)`
2. `_observe_human_moves` — append any new human relocations to `_day_events`
3. LLM call every tick with past warmup days, today's moves (or a morning-idle
  message when `_day_events` is empty), and current time (`HH:MM`)
4. Structured schema `predict_relocations` (dest enum excludes `home`)
5. **Filters**: drop unmapped classes, unplaceable `home`, already-at-destination
6. Map class names → slate indices; emit up to `max_predictions` actions

---

## 9. Oracle (`agents/robot_policy_oracle.py`)

Measurement instrument reading **ground-truth** relocation streams from processed `.pt`
files — not a learnable method.

**GT source** (`_load_day_script`):

- `closed_loop_eval` passes the human's `script_path` each day via
`reset(script_path=...)`.
- GT file: `homers_script_path_to_processed_pt(script_path)` — same pairing as
offline edges / routines (`scripts_train/005.txt` → `processed_seqLM_coarse/train/005.pt`).

**Emit rule** (`TICK_MIN = 10`):

- `get_actions` is invoked every sim tick; emits only when `t_now == t_reach - 10`
(exact equality on the GT schedule — not a free-form poll model)
- Skip if `emit_t < t_prev + 10` (no free tick — structurally unclaimable)
- Skip if already at destination
- One move per object per tick (earliest reach wins)

No policy-style guards — pure physics ceiling only.

`train()` writes noop stub `weights_day{N}.pt`.

---

## 10. GAP, our method (`agents/robot_policy_hybrid.py`)

Grounding by an LLM curator at train time, Anticipation by task graphs, a particle filter and
a transformer, and Planning by a heuristic planner -- the three stages the paper names. The
LLM is used only during training: an evaluation tick makes zero LLM calls.

### 10.1 Training pipeline (`train`, ~208–262)

1. `MemoryCurator.curate_dataset(warmup_pt_dir)` — LLM segments days → `tagged_days/*.json`
2. `build_embeddings()` → `embeddings.json`
3. If `continual_learning.enabled`:
  - `select_gamma(tagged)` — prequential backtest on training tags only
  - `_attribute_robot_events(tagged)` — join `attributed_days/*.json` from prior eval
  - Rebuild graphs with γ-decayed day weights
4. `train_activity_transformer(...)` → `weights_day{N}.pt`
5. `load_artifacts(weights_path)` — transformer + graphs + PF

### 10.2 Memory curator (`agents/llm_memory_curator.py`)

**Train-time only.**

- `extract_events_from_pt`: net relocations per 10-min tick from `.pt` edges (no GT activity)
- `curate_day`: LLM assigns activity segments + open-vocabulary names; deterministic
`_sanitize_segments` repair; vocab merges propagate to all tagged days
- `build_task_graphs`: when CL enabled, mutates **in-memory** `tagged` with robot/
reverted/superseded flags before `TaskGraphSet.build_from_tagged_days`; saves
`task_graphs.json`. Does **not** write robot flags back to `tagged_days/*.json` on disk.

Robot join: match human events for offset detection; robot keys accept `t` and `t+10`;
crashes if robot keys exist but match rate `< 0.3`.

### 10.3 Task graphs (`agents/task_graph.py`)

Per-activity directed graph over nodes `(obj_idx, dest_idx)` in **slate index space**.

`**add_segment(events, weight, drop_reverted=True)`**:

- Skips `reverted` / `superseded` robot events when dropping
- Collapses **consecutive identical** moves; keeps non-adjacent repeats
- Accumulates `count`, `count_robot` (if `actor=='robot'`), `offset_sum_min`, edges

`**transition_probs`**: Laplace-smoothed successor distribution.

`**match(observed)**`: longest-suffix localization → `{node_id: weight}`.

`**predict_next(position, lookahead, beam_width, decay)**`: beam search over transitions.

`**prereq_nodes(node_id)**` (hybrid policy gate): same-object strict ancestors with earlier
mean offset; returns `(nid, p_on_path)` where `p_on_path = min(1, count/n_segments)`.

`**select_gamma(tagged_days)**`: prequential grid search; tie-break toward largest γ.

### 10.4 Activity transformer (`model/activity_transformer.py`)

~250k-param model; frozen OpenAI name embeddings → learned projection → causal transformer.

**Modes** (`transformer.mode`):

- `event` (default): one token per segment `[proj(name), τ(start), τ(end), dur/1440]`;
predicts next activity + start-time via clock CE over 10-min ticks
- `interval`: one token per tick; predicts next-tick activity

`**predict_next(segments, now_min)`** (eval): `(probs, gaps)` — used by **prep branch only**,
not PF recognition.

Training: temporal val split (`val_days`); optional γ day weights + warm-start state dict
when CL enabled.

### 10.5 Graph particle filter (`agents/activity_particle_filter.py`)

**Full algorithm flowcharts:** [docs/particle_filter.md](docs/particle_filter.md) — lifecycle,
`observe` / `observe_robot` / `observe_reaction`, `_path_credit`, resampling, readouts, and
open-activity state machine (mermaid).

**Particles**: N=1000; each has `open` (≤ `MAX_OPEN=3` hypotheses), `history` (closed segments),
`last_assign` (roulette label).

**Priors** (`derive_priors(tagged_days)` from disk tagged days):


| Key             | Source                                       |
| --------------- | -------------------------------------------- |
| `clock[act]`    | Gaussian on segment start times (σ ≥ 20 min) |
| `p_bg`          | unassigned event fraction / pair space       |
| `stale_gap_min` | p95 within-segment gap                       |
| `cont_tau_min`  | max(p95, mean gap, 1) — idle decay scale     |


**Path credit** (`_path_credit`, `CONT_HOPS=4`): best path to `(obj,dest)`; exact=1.0,
same-room wrong dest = `ROOM_MISS_DISCOUNT=0.1`.

`**observe(obj, dest, t)`** — human move:

1. Idle hazard in weights + stochastic stale-close proposals between human moves
2. Reweight: continue open / open new (from START or robot entry anchors) / background `p_bg`
3. Roulette-commit one interpretation; advance markers on human continue; renormalize;
  resample if ESS low

`**observe_robot(obj, dest, t)**`:

- Marks node in `done` + `robot_done` for matching open activities
- Stochastic marker advance (`ADVANCE_P=0.5`) if forward-reachable

`**observe_reaction(obj, robot_dest)**` — human undo:

- Re-arms `done`/`robot_done`; weight × `p_bg/(p_bg + p_node(nid))`; renormalize;
resample if ESS low

**Readouts:** `posterior_last_assign()`, `actuation_view(pace=True)` (always pace in
production), `map_segments()` (transformer prep input).

**Particle-filter core:** idle hazard in weights between observations, stochastic END
close (`_maybe_close`), undo penalty in `observe_reaction`, pace-anchored
`actuation_view`. No `pf_version` knob — older PF variants removed.

**Hybrid policy layer (not in PF):** graph-precedence gate (`_gate_precedence`),
learned per-key lead times (`_lead_time`, pinball-updated each day in `reset`), and the
**acceptance gate** (below).

The reported configuration is this filter plus continual learning (`HYBRID_CL=1` /
`--hybrid_cl`) and the acceptance gate at c_thresh = 0.1.

#### Acceptance gate (`_accept_gate`, opt-in via `policy.accept_gate` / `--hybrid_accept_gate`)

The PF/graph stack proposes where objects *tend* to go; the acceptance gate learns where the
**human actually tolerates the robot putting them**, and suppresses placements the human keeps
undoing. It is the mechanism that makes the hybrid viable under the LLM human (without it, churn
drives net saved negative).

- **Per-`(obj_idx, dest_idx)` acceptance probability** `_p_accept`: Laplace-smoothed,
  γ-decayed accept/reject weights with an **optimistic prior** (`ACCEPT_PRIOR=2.0` → an unseen
  placement starts at `P_accept=1.0`, never pre-suppressed). Weights persist across days like
  `_lead_time`, decayed by `accept_gamma` (default 0.9) each day boundary.
- **Soft multiplier, not a hard veto:** in `get_actions` the candidate score is scaled
  `score *= P_accept` *before* the `prob_threshold` guard — a placement the human reliably
  reverts decays out of contention; one left alone stays at the prior.
- **Within-day reject backoff:** a fresh reject arms `_reject_until[(obj,dest)]` with exponential
  backoff (`ACCEPT_BACKOFF_BASE_MIN=20` → doubling → `ACCEPT_BACKOFF_MAX_MIN=60`); that key is
  hard-skipped (`outcome=reject_backoff`) until the wait elapses.
- **Day-boundary credit** (`_update_accept_counts`): a move OFF a robot placement within
  `REACTION_WINDOW_MIN=20` is a reject; later motion is a continuation. reject = `reverted` /
  `redirected` / `rejected`; `superseded` (robot self-churn) is always excluded.
- **Build-upon credit (`accept_buildon` / `--hybrid_accept_buildon`), ablated in the paper:**
  - **Without it**, the gate credits an accept for both `continued` (human built upon the placement) **and**
    `active` (object merely stayed put). For an object the human is indifferent to, "stayed put"
    inflates `P_accept`, pinning it at a ~0.5 plateau so the placement never craters.
  - **With it** (reported), an accept is credited **only** for `continued`; `active`-stuck is **neutral** (excluded,
    like `superseded`). Correctly-placed-and-left objects stay at the optimistic prior (no recall
    hit), while genuinely-unwanted placements crater via their real reverts.

**Every tick:** `_observe_human_moves` updates the PF only when a new human
relocation appears in the history diff (tick 1 has no diff yet). **Actuation
always runs** — `_candidate_actions_pf` scores and may emit every tick, including
before the first human move (transformer prep from empty `map_segments()` =
morning idle). The old early-return guards (“wait for first tick / first human
move before acting”) are removed.

**Prep specifically:** `_prep_branch` runs on **every** eval tick (any time of
day, human moves or not). A prep move actually emits only when the transformer
predicts a next activity with `p ≥ next_act_conf` and start within
`prep_horizon_min`, the candidate passes precedence (prereq-free openers only),
`prob_threshold`, learned lead / timing, and is not already at destination —
same cap as all robot actions (`max_actions_per_tick`).

### 10.6 Eval tick flow (`get_actions`)

Each tick after history update:

```
_observe_human_moves:
  diff history edges → human events (skip robot last-tick emissions)
  for each new human event:
    revert? (_find_revert_placement: undo to from_idx)
      → observe_reaction; mark placement reverted; _lead_reverted
    else:
      → observe; _consume_continuation (move onward from robot dest)

_candidate_actions_pf:
  actuation_view(pace=True) → 'current' candidates (PF open activities)
  precedence gate (_gate_precedence)
  _prep_branch: transformer.predict_next on map_segments() → 'prep' candidates
    (empty segments at day start = BOS / morning idle)

Guards (hybrid.policy):
  accept gate (if on): reject_backoff skip, then score *= P_accept
  prob_threshold, already_at_dest, max_actions_per_tick,
  act_window_min fallback / learned per-key lead

Emit (all gate-approved candidates; LLM ranker removed):
  supersede prior placements on same object
  observe_robot; append attributed day event; track lead keys
```

**Placement stack** (per object): statuses `active | superseded | reverted | continued`.
Restaging marks prior active entries `superseded`. Success % for the learned lead excludes both
reverted and superseded.

**Day boundary** (`reset`): pinball-update the learned lead times (always, not CL-gated);
persist `attributed_days/{day:03d}.json` if CL enabled; clear PF and day state.

**Hybrid policy defaults** (`robot.hybrid.policy` in `base.yaml`): `prob_threshold`
0.1, `max_actions_per_tick` 8, `prep_horizon_min` 60, `next_act_conf` 0.5,
`prep_discount` 1.0, `act_window_min` 30 (learned lead overrides per action),
`lookahead_actions` 3; acceptance gate `accept_gate` off, `accept_buildon` off,
`accept_gamma` 0.9. CLI mirrors via `--hybrid_`* flags (`--hybrid_accept_gate`,
`--hybrid_accept_buildon`); CL via `--hybrid_cl`.

**`prob_threshold` is the headline tunable parameter** — the post-gate score cutoff that
sets the precision/recall operating point. The paper sweeps it (Fig. 4, `figures/cthresh_sweep.png`): raising it
trades recall for precision (pt0.5 → precision ≈0.81–0.92 but recall collapses to ~0.10–0.14).
`pt0.1` is the reported operating point. With the acceptance gate on, the effective cutoff is
`prob_threshold` applied to the `P_accept`-scaled score, so the two knobs compose.

### 10.7 Continual learning (`robot.hybrid.continual_learning`)

When `enabled` (CLI: `HYBRID_CL=1`):

- Each retrain: `select_gamma` on tagged days; γ-decayed graph counts; robot events joined
from eval `attributed_days/`; reverted/superseded dropped from graphs
- Transformer training mirrors γ day weights (`transformer_recency`, default True)
- `count_robot` accumulated per graph node during build; consumed by
`scripts/plot_task_graphs.py` (human vs robot node fill). Not read by PF,
hybrid policy, or transformer at eval time.

**CL-knob ablations (scripted human, retrain).** Two single-knob removals of the
continual-learning recency machinery, each on the reported configuration (build-upon
gate, pt0.1, pbg0.001) with exactly one knob flipped; 3 households (A+B+C mean),
scripted human, curator cache reused (zero curator spend).

- **no-recency** (`--hybrid_no_transformer_recency` →
  `continual_learning.transformer_recency=False`): transformer training stops
  mirroring the γ day-weights (uniform day weights); graph γ-decay unaffected.
- **no-CL-forget** (`--hybrid_no_cl_forget` →
  `continual_learning.cl_forget=False`): the prequential `select_gamma` is bypassed
  and γ pinned to **1.0** (no recency discount on accumulated graph evidence); CL
  still retrains each cycle and still attributes robot events — only forgetting is
  removed. Verified live in `gamma_log.jsonl` (`gamma: 1.0, scores: null`).

```
cond            net%   hard%  recall   prec     f1   unseen
baseline       30.83   10.13   0.320  0.775  0.452     868
no-recency     31.15    9.92   0.322  0.780  0.456     839   (Δnet +0.32, Δf1 +0.004)
no-CL-forget   31.21   10.01   0.325  0.783  0.459    1006   (Δnet +0.38, Δf1 +0.007)
```

**Reading it:** both knobs are **inert on the scripted human** — every delta is within
run-to-run noise (≤0.4 net%, ≤0.007 f1). This is expected: the scripted human is
**stationary**, so there is no temporal drift for the recency weighting (transformer
day-weights) or the γ-forgetting (graph evidence discount) to exploit — pinning γ=1 or
flattening day-weights costs nothing because recent and old days carry the same signal.
The CL recency machinery is built for the **non-stationary** human; its value is not
expected to (and does not) appear under a stationary schedule. Result dirs:
`results/hybrid_v2/hybrid_retrain_..._gate_pbg0.001{_norecency,_nocl}`.

### 10.8 Branch attribution: `current` (PF/task-graph) vs `prep` (transformer)

Every emitted action is tagged `source` in `policy_diag.jsonl` — `current` (PF/task-graph
assistance for the activity in progress) or `prep` (transformer-predicted **next**-activity
pre-staging). Joining that tag (by day, emit-tick, object) to the harness per-action
`human_action_outcome` in the benchmark json's `actionwise_evals_daily` attributes every
outcome to the branch that produced it. `scripts/branch_attribution.py` does the join and
prints this table for any run directory, and writes it as CSV to
`results_final/source_tables/`. One catch it handles: `policy_diag` counts days from the
policy's first evaluation day while the benchmark json is indexed by absolute day, so the two
differ by a constant per run (0 for the LLM-user cell, 4 for the scripted one). At the wrong
offset roughly 15% of actions still join by coincidence, so the script recovers the offset and
refuses to report unless the best one beats the runner-up 3:1.

Reported cells, pooled over households A/B/C:

```
                    % of robot actions   % of true positives   precision
LLM user   current                75.7                  83.4       0.639
           future (prep)          24.3                  16.6       0.471
scripted   current                66.9                  80.1       0.808
           future (prep)          33.1                  19.9       0.707
```

TP = skipped + accepted; FP = corrected + corrected-through-vigilance + redirected + undone;
precision = TP/(TP+FP). `unseen` (the robot moving an object again before the user ever saw
the first placement) is robot self-churn, not a user-tagged error, so it is excluded from
precision. 3.0% of emitted actions under the LLM user and 11.1% under the scripted user never
receive an outcome; they count in the action share and in neither TP nor FP.

**Reading it:** current-goal assistance is the high-precision workhorse and future-goal
preparation extends reach at a worse rate -- precision 0.81 vs 0.71 scripted, 0.64 vs 0.47
under the LLM user. Preparation is over-represented in cost relative to its share: it emits a
quarter to a third of all actions but earns only 17-20% of the true positives. Anticipating a
future goal is the harder problem, since the robot must predict which goal comes next before
it can predict its actions.

Reproduce: join `policy_diag.jsonl` (`candidates[].source` where `outcome=='emitted'`) to
`actionwise_evals_daily[*][ts][].human_action_outcome` on `(day, round(t_min), object_name)`.

---

## 10a. NOPA baseline (`agents/robot_policy_nopa.py`)

Puig, Shu, Tenenbaum & Torralba, *"NOPA: Neurally-guided Online Probabilistic
Assistance for Building Socially Intelligent Home Assistants"*, ICRA 2023.

The only prior-work baseline here that reasons about **goal uncertainty** — the others
(SLaTe-PRO, STREAK, plain-LLM) forecast the next state and act on the forecast. NOPA
maintains a distribution over what the human is trying to accomplish and decides whether
acting is worth it. Deliberately **self-contained**: it does not import GAP's
particle filter, task graphs, or LLM curator. It shares only `extract_events_from_pt`
(a pure `.pt` reader, also used by the oracle).

### 10a.1 Three components

| Paper | Implementation |
| --- | --- |
| Goal Proposal Network `q(g|s0,st)` (Fig. 2b) | `model/nopa_gpn.py::GoalProposalNetwork` — 4-layer FC trunk to `h=128`, 2-layer MLP head, per-predicate softmax over count bins (~279k params) |
| Inverse-planning particle filter (Alg. 2) | `_resample` / `_consume` + the rejection loop in `get_actions` |
| Uncertainty-aware helping planner (Alg. 3) | `_candidate_edges` + `_value` + `_emit` |

**Predicate space.** A predicate is an `(obj_idx, dest_idx)` pair in slate index space;
the vocabulary is every pair observed in the GPN's training days. Measured `|P|` =
106 / 109 / 84 (HH A/B/C, 65 days) and 93 / 87 / 73 over 5 days — close to the paper's
`p = 136`. Counts use **4 bins** `{0,1,2,3+}` (paper `c = 9`); the measured maximum
repeats of one predicate in a day is 3, and 87-93% of realizations are count 1.

**Goal / episode.** One simulated day = one NOPA episode; the goal is the multiset of
predicates the human realizes that day. `s0` is the world state at the day's first tick
(captured in `get_actions`, so under `--no_reset_env` it is the drifted 06:00 state).
Goals are **dense** here: 48-58 distinct pairs/day out of a ~100-pair vocabulary, i.e.
the human realizes roughly half the vocabulary every day, versus a handful of predicates
in O-WAH. Goal *identity* is therefore weakly informative and almost all the
discriminative signal is *timing*.

### 10a.2 Training (`train`)

The GPN is fit on **main-agent-alone days only** — the paper's premise (3-C: "GPN trained
on episodes with only the main agent performing the tasks"), which is what keeps it robust
once the helper starts perturbing the state. `robot.nopa.solo_days` (wired to
`--warmup_days` by the CLI) selects `sorted(glob(<dataset>/train/*.pt))[:solo_days]`;
unset raises rather than silently defaulting. One training row per `(day, tick)` — 109
ticks/day — with the day's **full** goal-count vector as a constant label. Temporal split,
last `val_days` days held out; `MAX_TRAIN_EPOCHS` env, early stop on val loss.

Reported val metrics include **goal-set F1** (`count>=1` vs true `count>=1`), which is the
number that predicts helping quality; per-predicate count accuracy is inflated by the many
always-zero predicates.

Each retrain round rebuilds the vocabulary and re-initialises from scratch. Consequence:
in the **retrain shape** (`WARMUP_DAYS=5`, `TRAIN_EVERY_N_DAYS=5`) every round re-fits the
same 5 solo days, so NOPA's retrain cells add no learning across rounds. That is faithful
— NOPA has no continual-learning story, and inventing one would make it a variant of our
method. `include_finetune_days: true` is the explicit deviation ablation.

The checkpoint `weights_day{N}.pt` is **fully self-contained** (state dict, config,
predicate vocabulary, arrival-clock prior, val metrics), so `load_artifacts` restores
everything in one read and the three resume paths in `closed_loop_eval.py` need no extra
state — unlike the hybrid, which needs `task_graphs.json` on disk.

### 10a.3 Eval tick flow (`get_actions`)

```
_update_history(world_state); _last_fwd = None
first tick of the day -> capture s0, _resample(reason='day_start')

_observe_human_moves:            diff last two history frames
  _is_undo?      -> skip (a revert of OUR placement is a reaction, not evidence)
  OOV predicate? -> n_oov += 1, skip (a closed vocabulary cannot represent it)
  else           -> counts[pid] += 1; reject particles without pid in their
                    T_prop prefix; drop pid from survivors' plans

Q empty -> _resample('empty');  l >= T_prop -> _resample('horizon')

_candidate_edges:  goal edges (p(e), L_M) + restore edges (p=1, L_M=0)
                   plan-order gate: one candidate per object
_value:            V = w_r*p(e)*|L_M - L_H|+ - w_c*L_H - w_m*dD
emit argmax V over min_value, cap max_actions (paper Alg. 3: 1)
```

`_is_undo` matches an active placement of the same object within
`REACTION_WINDOW_MIN = 20` whose `prev` equals the new destination. An LLM-human
**`redirect`** moves the object somewhere new (`dest != prev`) and correctly survives as
real goal evidence.

`D(s0, .)` is the count of movable objects off their `s0` location, so a single teleport
changes it by exactly one of `{+1, -1, 0}` — a closed form replacing the paper's state
distance. `min_value = 0.0` implements the paper's idle behaviour (Fig. 1 step 1,
`Help: No action`).

### 10a.4 Deviations from the paper

1. **MCTS -> sampled arrival-clock plan.** The simulator charges no navigation or
   manipulation cost (`environment/world.py`: a teleport is one tick), so search over
   motion plans would be uninformative — a plan here is a *schedule*, not a cost-to-go.
   Each remaining predicate draws `tau ~ TruncNormal(mu_p, sigma_p, lower=t_now)` from a
   per-predicate Gaussian fitted on the training days (`sigma` floored at 20 min, the
   `derive_priors` convention). Per-particle sampling supplies the bounded rationality the
   paper gets from limited rollouts: two particles holding the same goal get different
   trajectories, as Alg. 2 requires.
2. **`L_H(e) == 1` for every edge**, so `-w_c*L_H` is a constant that can never change
   `argmax V`. `w_c` is an **inert hyperparameter** in this domain.
3. **Predicted-intermediate-state edges collapse into predicted-goal edges** — a plan
   prefix here *is* a subset of the goal. Their only residual effect is the `L_M` vs
   `L_max` split at the `T_prop` horizon.
4. **Count bins 9 -> 4** (measured max repeats per day = 3).
5. **Plan-order gate** (`policy.plan_order_gate`, default on). NOPA's goals are consistent
   *final states* — in O-WAH each object has one target location, so Alg. 3 scores every
   goal edge independently. A household day is a *sequence*: **21 of 29 moved objects
   visit two or more destinations in the same day** (chessboard -> table in the morning,
   chessboard -> bookshelf at night). Scoring both at once makes the helper ping-pong the
   object, because achieving the later predicate destroys the earlier one. The gate keeps
   each object's **earliest remaining** predicate, ordered by the particles' own sampled
   plans — read off `Gamma_hat_k`, not bolted on. `--nopa_no_plan_order_gate` restores the
   literal reading (measured cost in 10a.6).
6. **Restore edges are limited to the robot's own placements.** Read literally, Alg. 3
   admits every initial-state edge, which here scores `V = -w_c + w_m = +4` for *any*
   object off `s0` — including ones only the human moved, so the robot would spend the
   afternoon undoing the human's work. Fig. 6b is explicit that the helper puts back "the
   fork **it grabbed**". Default: restore only objects the robot placed and that no
   particle still wants. `--nopa_restore_all_initial` is the literal reading.
7. **Delta s comes from human-attributed events, not the raw state diff** — unlike NOPA's
   helper, ours knows which state changes it caused.
8. **OOV observations are excluded from Alg-2 rejection**; rate logged as `n_oov`
   (measured 3.2-6.8% of events against a 5-day vocabulary).
9. **Clock features are appended to `Delta s_t`.** O-WAH episodes are short and
   clock-free; a household routine is not.
10. **Speedup -> net saved %** (metric definitions in §13). `AFGPN` and `Empowerment` are dropped:
    both are navigation/affordance-cost baselines with no analog in a teleport world.

No destination masking (`valid_locations` / `containment_ban`) is applied — every
vocabulary edge is a relocation a human was observed to perform, so ancestor cycles are
impossible by construction.

### 10a.5 Config and ablations

`robot.nopa` in `base.yaml`: `solo_days` (null -> raise), `include_finetune_days`,
`gpn.{n_counts 4, hidden 256, h 128, head_hidden 64, dropout 0.1, lr 1e-3, batch_size 256,
val_days 5, early_stop_patience 50}`, `policy.{K 20, T_prop 15, L_max 100, w_r 1, w_c 1,
w_m 5, max_actions 1, lm_units ticks, min_value 0, sigma_floor_min 20, goal_source gpn,
inverse_planning true, restore true, restore_all_initial false, plan_order_gate true,
argmax_goal false, seed 0}`. CLI mirrors via `--nopa_*`; wrapper
`scripts/run_benchmark_nopa.sh` (env knobs `NOPA_*`, `.done_<HH>` marker, USR1
self-resubmit). Each ablation gets its own out-dir tag so it never averages into the
headline row (`NOPA_ABLATION_TAGS` in `scripts/consolidated_table.py`).

| Paper | Flag | Mechanism |
| --- | --- | --- |
| `OursRG` | `--nopa_goal_source uniform` | uniform count bins instead of GPN probs |
| `Ours-InvPlan` | `--nopa_no_inverse_planning` | no Alg-2 rejection (still resamples every `T_prop`) |
| `Ours-Return` | `--nopa_no_restore` | drop restore edges and pin `w_m = 0` |
| `HPGPN` | `--nopa_argmax_goal` | `K=1`, per-predicate argmax count |
| `HPRG` | + `--nopa_goal_source uniform` | random goal, no particles |
| `HPGT` | — | **the existing oracle row** (9); not re-implemented |
| (ours) literal Alg-3 | `--nopa_no_plan_order_gate` | score every goal edge, several destinations per object |
| (ours) literal restore | `--nopa_restore_all_initial` | restore any object off `s0` |

`<memory>/nopa_diag.jsonl` logs per emission (`kind`, `V`, `p(e)`, `L_M`, particle count,
candidate count) and a per-day summary (`n_oov`, `n_resample`, `resample_reasons`,
`n_robot_actions`, `max_repeats_one_edge`). `resample_reasons` is the diagnostic that
matters: if `empty` dominates, Alg-2 rejection is killing the whole particle set every few
observations and NOPA is behaving as `Ours-InvPlan`.

### 10a.6 Results

Reported cells, macro mean ± sd over households A/B/C (`results_final/nopa_*`):

```
condition                        net saved %   perturbed %        F1
scripted user                     11.8 ± 1.9    8.0 ± 0.8   0.198 ± 0.034
LLM user                         -12.5 ± 7.5    5.2 ± 0.1   0.154 ± 0.029
scripted, test-time drift only     8.8 ± 1.1    3.4 ± 0.7   0.156 ± 0.015
```

**Reading it:**
- **NOPA perturbs less than any other method under both users** (5.2% LLM, 8.0% scripted,
  against GAP's 12.3 / 10.5) and saves the least of the non-oracle baselines on the scripted
  user. That profile follows from Alg. 3: one subgoal per tick plus the `w_m` disturbance
  penalty buy restraint, and the restraint is not enough to overcome a goal model that is weak
  in this domain (10a.1).
- **It goes net-negative under the LLM user** (11.8 -> -12.5), consistently across households
  (-11.7, -20.3, -5.4), so it is not a single-runaway-household artifact.
- **The mechanism is a churn loop the method cannot damp.** Against 1,563 actions the user
  credits as saved trips, it undoes 3,319 placements and redirects 675. When the user reverts
  a placement, the predicate is still in every particle's goal, so Alg. 3 re-proposes it on the
  next tick. `w_m` cannot stop this: it is static and goal-agnostic, penalising deviation from
  `s0` rather than learning *which* placements this user tolerates. That is exactly what GAP's
  acceptance estimate does learn, per (object, destination), from observed reverts.
- Test-time drift alone costs NOPA little (8.8 vs 11.8 net): it re-fits the same solo days
  every retrain round, so training-time drift adds exposure but no new learning.

## 11. Data flow (GAP)

```
Warm-start days:
  world ticks → gen-dataset .pt files

Train boundary:
  curator → tagged_days, embeddings, task_graphs
  transformer train → weights_dayN.pt
  load_artifacts → PF + graphs + transformer

Finetune/test ticks:
  history → human diff → PF observe
  PF posterior × graphs + transformer prep → candidates → guards → RobotMoveAction
  attributed_days/*.json (for next retrain join)
```

---

## 12. Key file map


| Component           | Path                                                           |
| ------------------- | -------------------------------------------------------------- |
| Closed-loop harness | `scripts/closed_loop_eval.py`                                  |
| Benchmark CLI       | `scripts/benchmark_comparison.py`                              |
| World / physics     | `environment/world.py`                                         |
| Scripted human      | `environment/human_agent.py`                                   |
| LLM human           | `environment/llm_human_agent.py`, `agents/llm_human_prompt.py` |
| SLaTe-PRO policy    | `agents/robot_policy_slatepro.py`, `agents/slatepro_utils.py`  |
| Plain LLM policy    | `agents/robot_policy_plainllm.py`                              |
| Oracle policy       | `agents/robot_policy_oracle.py`                                |
| Hybrid policy       | `agents/robot_policy_hybrid.py`                                |
| NOPA policy         | `agents/robot_policy_nopa.py`, `agents/nopa_utils.py`, `model/nopa_gpn.py` |
| Particle filter     | `agents/activity_particle_filter.py`                           |
| Task graphs         | `agents/task_graph.py`                                         |
| Curator             | `agents/llm_memory_curator.py`                                 |
| LLM prompt examples | `docs/llm_prompts.md` (one rendered example of every prompt)    |
| Transformer         | `model/activity_transformer.py`                                |
| Metrics analysis    | `scripts/analyze_llm_human_conditions.py`                      |
| Config defaults     | `config/base.yaml`                                             |


---

## 13. Results

Two live artifacts over all methods and settings, regenerated with
`scripts/final_results_tables.py` (writes `results/FINAL_RESULTS_TABLES.txt`, regenerate
it -- the stale copy is not shipped). Both average
over the households (A,B,C) with a `.done_<HH>` marker and read only each benchmark json's
leading `totals` block (avoids loading the ~1GB files). The older
`scripts/consolidated_table.py` (regenerates `results/CONSOLIDATED_TABLE.txt`) is retained as the
PRIOR/CURRENT effort-ledger view; the tables below supersede it for reporting.

**The exact runs behind every reported number live in `results_final/`**, one directory per
table cell, rebuilt by `scripts/export_results_final.py` from the sources it names. A cell is
exported only when all three households carry a `.done_<HH>` marker, so a partial run cannot
reach a table. The full run directories are archived outside the repo.

| table row | scripted | LLM human |
| --------- | -------- | --------- |
| SLaTe-PRO (AB100) | `results_final/slatepro_scripted` | `results_final/slatepro_llm_human` |
| STREAK (AB100; LLM = max2/capped) | `results_final/streak_scripted` | `results_final/streak_llm_human` |
| Plain LLM | `results_final/plainllm_scripted` | `results_final/plainllm_llm_human` |
| **our method** (AB0) | `results_final/hybrid_scripted` | `results_final/hybrid_llm_human` |
| Oracle | `results_final/oracle_scripted` | `results_final/oracle_llm_human` |

The two Plain LLM runs that shared an observed-dataset directory are deleted; both cells now
come from the corrected reruns (one observed-dataset directory per household), which finished
2026-11-20. The correction costs the baseline on both humans: scripted 19.5 -> 22.3 % net saved,
LLM human -174.5 -> -224.7 % net saved and 62.1 -> 88.6 % perturbed.

STREAK's LLM cell is the **capped** (`--streak_max_actions 2`) run; the uncapped LLM run
(collapse, −578%) is not shipped: its run directory was deleted in the release cleanup, and
only the number above survives.

**Metric definitions** (canonical, from `closed_loop_eval.py:_metrics`):
TP = skipped+accepted; FP = corrected+corrected_despite_correct+undo(vigilance)+redirected;
FN = done_as_is; precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2·TP/(2·TP+FP+FN);
**net** = savings−(undo+redirected) and **net%** = net/solo (net workload removed);
**hard** = corrected+corrected_repeated+cdc+redirected and **hard%** = hard/solo (actions
made harder); **unseen** = phantom/never-observed placements. Legend: `*` = <3 HH done · `-`
= variant not run.

### A. Reported results

Every number below is recomputed from `results_final/` by `scripts/final_results_tables.py`:
macro mean ± sample sd over households A/B/C, which is what the paper's intervals are.

```
MAIN RESULTS -- drift during training and evaluation (paper Table III)
                           LLM user                              scripted user             
method        net saved %   perturbed %            F1     net saved %   perturbed %            F1
SLaTe-PRO       2.1 ± 7.5     18.1 ± 3.1    0.216 ± 0.034      16.6 ± 3.0     11.8 ± 1.4    0.264 ± 0.043
STREAK       -159.9 ± 50.3    10.6 ± 10.8   0.002 ± 0.001      20.3 ± 7.3     48.3 ± 19.0   0.296 ± 0.105
Plain LLM    -224.7 ± 10.7    88.6 ± 8.3    0.078 ± 0.004      22.3 ± 2.4     26.0 ± 0.4    0.356 ± 0.026
NOPA          -12.5 ± 7.5      5.2 ± 0.1    0.154 ± 0.029      11.8 ± 1.9      8.0 ± 0.8    0.198 ± 0.034
GAP (ours)     20.3 ± 2.9     12.3 ± 1.9    0.412 ± 0.042      33.6 ± 3.3     10.5 ± 1.0    0.485 ± 0.042
Oracle         44.4 ± 1.6      8.2 ± 1.1    0.617 ± 0.015      51.3 ± 0.0      8.8 ± 1.9    0.651 ± 0.006

DRIFT AT EVALUATION ONLY -- scripted user (paper Table VII, left half)
method        net saved %   perturbed %            F1
SLaTe-PRO      24.2 ± 5.4     10.8 ± 3.0    0.345 ± 0.071
STREAK          0.4 ± 0.6     66.3 ± 57.4   0.009 ± 0.015
Plain LLM      31.2 ± 2.6     25.8 ± 3.5    0.440 ± 0.025
NOPA            8.8 ± 1.1      3.4 ± 0.7    0.156 ± 0.015
GAP (ours)     35.1 ± 2.3     17.3 ± 0.6    0.481 ± 0.032
Oracle         54.7 ± 1.3     12.4 ± 2.1    0.686 ± 0.017
```

Net saved % and perturbed % are both shares of the same denominator -- the effort the user
would have spent alone -- so they are read together, not traded off against each other.

### A′. Training-data noise: why baselines keep the robot's rejected actions

`ADDED_BAD_PERC` decides which of the robot's **own** actions enter the reconstructed
training trajectory; the user's moves always enter. At 100 (reported) the robot's rejected
actions are kept; at 0 they are dropped, so the policy trains only on the path the user
endorsed. The paper reports the cost of dropping them: 14.3 points of net saved and 19.1
points of perturbed actions on the scripted user. Measured on STREAK, the baseline most
exposed to it:

```
condition                 net saved %   perturbed %      F1
keep rejected (reported)     20.3          48.3        0.296
drop rejected                 6.0          67.4        0.106
```

Artifacts: `results_final/streak_scripted` and `results_final/ab0_streak_scripted`.

### A″. Why STREAK is capped under the LLM user

STREAK is Markovian: it re-initialises from the current world every tick. When the LLM user
undoes a placement, the next tick re-proposes it, and the two trade the same object back and
forth -- one bounce per tick (HH-A day 39: the same relocation re-issued and undone 108 times
in a day). Uncapped, this reaches roughly 5,000 robot actions a day and -578% net saved, a
number too degenerate to report. The paper therefore caps STREAK at two actions per tick
(footnote 4), which holds the action rate flat for the full 74 days:

```
STREAK, LLM user          net saved %   perturbed %      F1
uncapped                      -577.9        112.5      0.048
capped at 2/tick (reported)   -159.9         10.6      0.002
```

The cap bounds the collapse without fixing it: undos still dominate and F1 falls further,
because keeping the top two relocations by confidence keeps the confidently wrong ones. A rate
limiter cannot make wrong predictions right, which is the argument for GAP learning *which*
placements to suppress instead. Run-to-run spread is large for this baseline -- the same
configuration lands in one of several degenerate modes by seed, with robot-action rates from
about 1 to 6,500 a day -- so its intervals in the tables are wide by nature.

### A‴. Offline replication of the baselines

Each baseline's own published pipeline, trained on the same 65-day split and scored with its
native one-step relocation metric. This is the offline setting the paper contrasts against
(VI-B): the numbers below are the ones these methods report, and they are not comparable with
the closed-loop table above.

```
model              HH-A (P/R/F1)       HH-B (P/R/F1)       HH-C (P/R/F1)       Avg (P/R/F1)
slatepro OFFLINE   0.450 0.490 0.469   0.703 0.482 0.572   0.581 0.519 0.548   0.578 0.497 0.530
  saved reference  0.619 0.454 0.524   0.712 0.356 0.475   0.572 0.535 0.553   0.634 0.448 0.517
  STREAK OFFLINE   0.719 0.557 0.628   0.614 0.506 0.555   0.699 0.471 0.563   0.677 0.511 0.582
```

- `slatepro OFFLINE` ≈ reproduces the `saved reference` (HH-C near-exact; HH-B higher recall;
  HH-A lower) — same ballpark, our from-scratch-on-65-days run vs the reference checkpoint.


