# Graph particle filter — algorithm flowcharts

Reference for `agents/activity_particle_filter.py` (`GraphParticleFilter`), the goal
inference in GAP's Anticipation stage.
Called from `agents/robot_policy_hybrid.py` on human relocations, robot
placements, and human undos.

---

## 1. Lifecycle overview

```mermaid
flowchart TB
  subgraph init ["Construction (once)"]
    TD["tagged_days from memory"]
    GS["TaskGraphSet"]
    TD --> DP["derive_priors()"]
    DP --> PRI["clock, p_bg, stale_gap_min, cont_tau_min"]
    GS --> PF["GraphParticleFilter.__init__"]
    PRI --> PF
    PF --> RD["reset() → N=1000 particles, uniform weights"]
  end

  subgraph day ["Each simulation day"]
    RD --> TICK["Hybrid get_actions() every tick"]
    TICK --> HM["_observe_human_moves: diff history"]
    HM -->|new human relocation| OBS["pf.observe()"]
    HM -->|revert to robot from_idx| REA["pf.observe_reaction()"]
    TICK -->|robot emitted move| ORB["pf.observe_robot()"]
    OBS --> READ["Readouts: posterior_*, actuation_view, map_segments"]
    ORB --> READ
    REA --> READ
    READ --> ACT["Hybrid candidate scoring (not PF)"]
    ACT --> EOD["Day boundary: hybrid reset()"]
    EOD --> RD2["pf.reset() — wipe particles, history, robot_keys"]
    RD2 --> TICK
  end
```

**Important:** New activities **only** open inside `observe()` on a human
relocation. Robot moves and time alone never call `open.append`.

---

## 2. Particle state

Each of **N = 1000** particles carries:

| Field | Meaning |
|-------|---------|
| `open` | ≤ **MAX_OPEN = 3** hypotheses (`_Open` each) |
| `history` | Closed segments `(activity, t_open, t_close)` |
| `last_assign` | Label after last human `observe` roulette (`__background__` or activity name) |

Each **`_Open`** hypothesis:

| Field | Meaning |
|-------|---------|
| `act`, `pos` | Activity name and graph marker node |
| `opened_t`, `last_t` | Segment start; last human advance time |
| `done` | Nodes satisfied (human and/or robot) |
| `robot_done` | Subset satisfied by robot only (undo target) |
| `anchor_t`, `anchor_pos` | Human pace anchor (robot advance does **not** update) |

Filter-level state: `weights[N]`, `_t_last_obs`, `_robot_keys_today`, `_path_cache`.

---

## 3. `derive_priors(tagged_days)` — training-derived scalars

```mermaid
flowchart LR
  A["For each tagged day"] --> B["Collect event pairs, segment starts, within-segment gaps"]
  B --> C["clock[act] = Gaussian μ, σ on segment start (σ ≥ 20 min)"]
  B --> D["p_bg = unassigned_frac / #distinct pairs"]
  B --> E["stale_gap_min = p95(gaps)"]
  B --> F["cont_tau_min = max(stale_gap, mean_gap, 1)"]
```

---

## 4. `observe(obj_idx, dest_idx, t)` — human relocation

**Trigger:** hybrid detects a new human move; passes `t = t_min = t_now`.

**Δt:** `t − _t_last_obs` (0 on first human move of the day; robot-only ticks do
not update `_t_last_obs`).

```mermaid
flowchart TD
  START(["observe(key, t)"]) --> DT["dt ← t − t_last_obs; t_last_obs ← t"]
  DT --> LOOP["For each particle i with weight w"]

  LOOP --> IDLE{"dt > 0 AND |open| > 0?"}
  IDLE -->|yes| DECAY["w_pre ← w × exp(−dt × |open| / cont_tau_min)"]
  DECAY --> STALE["For each open j (reverse order):<br/>if last_t < t AND rand < 1−exp(−dt/stale_gap):<br/>history.append(act, opened_t, last_t); pop open"]
  IDLE -->|no| OPTS
  STALE --> OPTS["Build options = list of (credit, kind, payload)"]

  OPTS --> CONT["For each open j"]
  CONT --> REOBS{"exact node for key<br/>already in done?"}
  REOBS -->|yes| C1["credit ← max(reobs_credit, p_bg)<br/>kind ← cont"]
  REOBS -->|no| C2["credit, nid ← path_credit(act, pos, key)<br/>if credit > 0: kind ← cont"]
  C1 --> OPENNEW
  C2 --> OPENNEW

  OPENNEW{"|open| < MAX_OPEN?"}
  OPENNEW -->|yes| ON["For each activity not already open:<br/>credit ← path_credit(act, START, key)<br/>boost from best robot_keys_today anchor if better<br/>if credit > 0: append (credit × clock(act,t), open, (act,nid))"]
  OPENNEW -->|no| BG
  ON --> BG["Always append (p_bg, bg, None)"]

  BG --> LIKE["total ← Σ credits<br/>new_w[i] ← w_pre × total"]
  LIKE --> ROUL["Sample interpretation ∝ credit"]

  ROUL --> K{"kind?"}
  K -->|cont| UC["pos←nid; last_t←t; done∪{nid};<br/>anchor_t←t; anchor_pos←nid;<br/>last_assign←act; maybe_close END"]
  K -->|open| UO["open.append(_Open(act,nid,t));<br/>last_assign←act; maybe_close END"]
  K -->|bg| UB["last_assign ← __background__<br/>(open unchanged)"]

  UC --> NORM
  UO --> NORM
  UB --> NORM["Renormalize weights"]

  NORM --> ESS{"ESS < N/2?"}
  ESS -->|yes| RS["_resample(): systematic resample, clone particles"]
  ESS -->|no| END(["return"])
  RS --> END
```

### 4.1 Interpretation credits (likelihood mixture)

| Kind | When added | Credit formula |
|------|------------|----------------|
| **continue (re-obs)** | Exact graph node for key already in `done` | `max(reobs_credit(act,nid), p_bg)` |
| **continue (forward)** | Path from current `pos` explains key | `path_credit(act, pos, key)` if > 0 |
| **open new** | Activity not in `open`, room under cap | `path_credit(act, START or robot anchor, key) × clock(act,t)` if > 0 |
| **background** | Always | `p_bg` |

Particle weight multiplies by **sum of all credits** (marginal likelihood).
One interpretation is **roulette-sampled** for state update.

### 4.2 Activity closing inside `observe`

| Mechanism | When | Close probability | Recorded close time |
|-----------|------|-------------------|---------------------|
| **Idle stale close** | Before scoring, if `dt > 0` | `1 − exp(−dt / stale_gap_min)` per open | `last_t` |
| **Graph END close** | After continue or open-new (`_maybe_close`) | `transition_probs(pos)[END]` | current `t` |

Background assignment does **not** close opens.

---

## 5. `_path_credit(act, src, key)` — shared subroutine

Used by `observe` (continue / open) and `observe_robot` (advance check).

```mermaid
flowchart TD
  P(["path_credit(act, src, key)"]) --> CACHE{"cached?"}
  CACHE -->|hit| RET["return cached (credit, nid)"]
  CACHE -->|miss| TGT["Build targets:<br/>exact (obj,dest) → factor 1.0<br/>same obj + same room, wrong dest → ROOM_MISS_DISCOUNT=0.1"]
  TGT --> EMPTY{"any target?"}
  EMPTY -->|no| Z["return (0, None)"]
  EMPTY -->|yes| BEAM["Beam search ≤ CONT_HOPS=4:<br/>frontier ← {src: 1.0}<br/>each hop: q ← p × transition_probs(s→dst), skip END<br/>track best q × target_factor<br/>prune to top 16 states"]
  BEAM --> OUT["cache and return (best, best_nid)"]
```

**Transition probs** (from `TaskGraph.transition_probs`):

```
P(dst | src) = (count(src→dst) + 0.5) / (Σ_j count(src→j) + 0.5 × #successors)
```

Successors = observed edges only (including **END**). END is never extended
during path search.

---

## 6. `observe_robot(obj_idx, dest_idx, t)` — robot placement

```mermaid
flowchart TD
  R(["observe_robot(key, t)"]) --> RK["robot_keys_today.add(key)"]
  RK --> LP["For each particle, each open oa"]
  LP --> MATCH{"key maps to node nid<br/>in oa.act graph?"}
  MATCH -->|no| LP
  MATCH -->|yes| MARK["done ∪ {nid}; robot_done ∪ {nid}"]
  MARK --> ADV{"rand < ADVANCE_P (=0.5)?"}
  ADV -->|no| LP
  ADV -->|yes| PC["path_credit(act, pos, key) > 0?"]
  PC -->|yes| MOVE["pos ← nid; last_t ← t<br/>(anchor NOT updated)"]
  PC -->|no| LP
  MOVE --> LP
```

- **No weight change**, no resampling.
- **Never** opens new activities or closes existing ones.

---

## 7. `observe_reaction(obj_idx, robot_dest_idx)` — human undo

```mermaid
flowchart TD
  U(["observe_reaction(key)"]) --> LP["For each particle i, each open oa"]
  LP --> UNDO{"nid in robot_done?"}
  UNDO -->|yes| REARM["done − {nid}; robot_done − {nid}<br/>w[i] × p_bg / (p_bg + p_node(nid))"]
  UNDO -->|no| LP
  REARM --> LP
  LP --> CHG{"any weight changed?"}
  CHG -->|no| END(["return"])
  CHG -->|yes| NORM["Renormalize weights"]
  NORM --> ESS{"ESS < N/2?"}
  ESS -->|yes| RS["_resample()"]
  ESS -->|no| END
  RS --> END
```

- **No** new opens, **no** closes.

---

## 8. `_resample()` — systematic resampling

```mermaid
flowchart LR
  A["ESS = 1 / Σ w²"] --> B{"ESS < N/2?"}
  B -->|yes| C["Systematic resample N indices from weights"]
  C --> D["particles ← clone selected; weights ← 1/N"]
  B -->|no| E["keep"]
```

Called after `observe` and `observe_reaction` when ESS is low.

---

## 9. Day reset

```mermaid
flowchart LR
  A["hybrid reset() at day boundary"] --> B["pf.reset()"]
  B --> C["Fresh empty particles + uniform weights"]
  B --> D["Clear path cache, robot_keys_today, t_last_obs"]
```

Still-open hypotheses are **discarded** (not appended to `history`).

---

## 10. Readouts (downstream, no state change)

```mermaid
flowchart TD
  subgraph readouts ["Query API"]
    PLA["posterior_last_assign()<br/>Σ w over last_assign labels"]
    PO["posterior_open()<br/>Σ w over open activity names"]
    AV["actuation_view(pace=True)<br/>per activity: P(open), position dist,<br/>base_t from anchor, done mass"]
    MS["map_segments()<br/>MAP particle → history + open segments"]
  end
  AV --> HYB["Hybrid _candidate_actions_pf:<br/>timing, precedence, transformer prep"]
  MS --> HYB
  PO --> HYB
```

---

## 11. Hybrid integration (when each PF method runs)

```mermaid
flowchart TD
  TICK["get_actions() each tick"] --> DIFF["_observe_human_moves(t_now)"]
  DIFF --> EV{"For each new human event"}
  EV --> REV{"Human moved object back<br/>to robot from_idx?"}
  REV -->|yes| REA["observe_reaction"]
  REV -->|no| OBS["observe"]
  TICK --> ROB{"Robot action emitted<br/>this tick?"}
  ROB -->|yes| ORB["observe_robot"]
  OBS --> CAND["_candidate_actions_pf"]
  REA --> CAND
  ORB --> CAND
```

Multiple human events at the **same** `t_now`: first gets full `dt`; subsequent
calls in the same loop get `dt = 0` (no idle decay between them).

---

## 12. Constants (structural, not derived)

| Symbol | Value | Role |
|--------|-------|------|
| `N_PARTICLES` | 1000 | Ensemble size |
| `MAX_OPEN` | 3 | Concurrent open activities per particle |
| `CONT_HOPS` | 4 | Max path length in `_path_credit` |
| `ROOM_MISS_DISCOUNT` | 0.1 | Same-room wrong-destination factor |
| `STD_FLOOR_MIN` | 20 min | Clock prior σ floor |
| `ADVANCE_P` | 0.5 | Robot marker advance probability |

---

## 13. Open-activity state machine (per particle)

```mermaid
stateDiagram-v2
  [*] --> NoOpen: day start / reset

  NoOpen --> Open: observe → open-new wins roulette
  Open --> Open: observe → continue wins roulette
  Open --> Open: observe → background (unchanged)
  Open --> Open: observe_robot (mark done, maybe advance pos)
  Open --> Open: observe_reaction (re-arm robot_done node)

  Open --> Closed: idle stale close (silence gap)
  Open --> Closed: graph END close after continue/open
  Open --> [*]: day reset (discarded, not in history)

  Closed --> [*]: in history list
  NoOpen --> NoOpen: observe → background only
```
