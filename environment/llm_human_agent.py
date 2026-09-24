"""LLM-driven human agent.

Subclasses ``HumanAgent`` and replaces the two rule-based accommodation
branches in ``_execute_current_action`` plus the vigilance hook
``review_robot_action`` with calls to an OpenAI-backed LLM.

Configuration block (under ``config['human']['llm']``):
    provider             : 'openai'
    model                : 'gpt-4o-mini'
    temperature          : 0.0
    cache_dir            : 'cache/llm_human/'
    max_history_actions  : 20
    few_shot_path        : 'config/llm_human_examples.json'
    max_llm_calls_per_day: 5000   # live API calls only; cache replays are free
    max_usd_per_day      : 1.0    # code default; the benchmark passes
                                  # --llm_max_usd_per_day explicitly
"""

from __future__ import annotations

import os

from environment.human_agent import HumanAgent
from evolving_graph.scripts import parse_script_line

from agents.llm_human_prompt import (
    DecisionRecord,
    LLMHumanClient,
    build_at_dest_prompt,
    build_observe_prompt,
    build_unexpected_prompt,
    load_examples,
)


class LLMHumanAgent(HumanAgent):
    """LLM-driven human: replaces scripted accommodation + vigilance with GPT calls.

    Three prompt types: ``at_dest``, ``unexpected``, ``observe`` (undo/allow/redirect).
    Disabled during warm-start via ``llm_active=False`` (set by closed_loop_eval).
    """

    def __init__(self, config: dict, node_map, script_file: str,
                 accommodate_for_robot: bool = True):
        super().__init__(config, node_map, script_file, accommodate_for_robot)
        llm_cfg = config['human'].get('llm', {})
        self.llm_model = llm_cfg.get('model', 'gpt-4o-mini')
        self.llm_cache_dir = llm_cfg.get('cache_dir', 'cache/llm_human/')
        self.llm_max_history_actions = int(llm_cfg.get('max_history_actions', 20))
        self.llm_few_shot_path = llm_cfg.get(
            'few_shot_path', 'config/llm_human_examples.json'
        )

        # Dollar budget is the preferred cap. The per-day call cap below is a
        # hard backstop on **live** API calls (cache hits are free). If both
        # caps are set, whichever trips first stops further LLM calls for the
        # day. Dollar spend is ``LLMHumanClient.cost_usd`` (reset by
        # ``reset_day``). API failures retry with backoff then raise — no
        # scripted fallback.
        self.llm_max_usd_per_day = float(llm_cfg.get('max_usd_per_day', 1.0))
        self.llm_max_calls_per_day = int(llm_cfg.get('max_llm_calls_per_day', 5000))

        # Observe-hook decision mode:
        #   'undo_allow' (v1, default) -> binary undo/allow
        #   'redirect'   (v2)          -> undo/allow/redirect (work-minimizing)
        self.llm_observe_mode = llm_cfg.get('observe_mode', 'undo_allow')
        # Redirect lookahead window (minutes): an object is only redirected to a
        # planned destination that falls within this horizon. The plan shown to
        # the LLM covers the same window. Floor of 90 min so the human always
        # sees at least as far ahead as the hybrid robot can act
        # (prep_horizon_min 60 + act_window_min 30) and judges preps fairly.
        self.llm_redirect_window_min = max(
            int(llm_cfg.get('redirect_window_min', 120)), 90)
        # (obj_id, dest_id) the human moved EARLY via redirect; consumed when the
        # matching script line is later reached (-> 'done_early', ignored).
        self._moved_early: set[tuple[int, int]] = set()

        self._examples = load_examples(self.llm_few_shot_path)
        self._client = LLMHumanClient(
            model=self.llm_model,
            cache_dir=self.llm_cache_dir,
            provider=llm_cfg.get('provider', 'openai'),
        )

        # Per-day mutable state used by the prompt builders
        self._day_history: list[dict] = []
        self.llm_decisions: list[dict] = []
        self._calls_today = 0

        # When False, the agent behaves identically to the scripted base
        # (no LLM calls). Toggled off during warm-start by the caller — see
        # ``closed_loop_eval.py`` where it sets ``human.llm_active = False``.
        self.llm_active: bool = True

        # Human-readable audit trail (robot action, scripted response, LLM
        # prompt, LLM response). Set via ``set_audit_log``; None disables it.
        self._audit_log_path: str | None = None

        # Mid-run-modifiable budget. The effective per-day USD cap is re-read
        # from a small text file on every budget check, so the cap can be
        # raised/lowered while the run is live (edit the file, no restart).
        # Precedence: per-run file > global cache-dir file > config value.
        self._budget_file: str | None = None
        self._global_budget_file = os.path.join(
            self.llm_cache_dir, 'llm_budget_usd.txt')

        # VH node-id -> class_name, refreshed from world_state each call so
        # prompts read object/location names, not raw numeric ids.
        self._id2name: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_audit_log(self, path: str | None):
        """Point the per-decision audit log at *path* (append mode)."""
        self._audit_log_path = path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def set_budget_file(self, path: str | None):
        """Point the per-run budget-override file at *path*.

        Seeds the file with the configured per-day cap if it doesn't exist, so
        there's always a discoverable knob to edit mid-run. Write a single
        float (USD) into it to change the per-day cap live.
        """
        self._budget_file = path
        if path and not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(f"{self.llm_max_usd_per_day}\n")

    def _read_budget_file(self, path: str | None) -> float | None:
        """Return the float in *path*, or None if absent. Malformed -> raise.

        Malformed content raises (rather than silently ignoring) because the
        file is an intentional control knob — garbage means a mistake the user
        should see immediately.
        """
        if not path or not os.path.isfile(path):
            return None
        with open(path) as f:
            raw = f.read().strip()
        if not raw:
            return None
        try:
            val = float(raw)
        except ValueError as e:
            raise RuntimeError(
                f"Budget override file {path} contains non-numeric value "
                f"{raw!r}; write a single USD float."
            ) from e
        if val <= 0:
            raise RuntimeError(
                f"Budget override file {path} has non-positive value {val}.")
        return val

    def _effective_budget(self) -> float:
        """Per-day USD cap, re-read live. Per-run file > global file > config."""
        for path in (self._budget_file, self._global_budget_file):
            val = self._read_budget_file(path)
            if val is not None:
                return val
        return self.llm_max_usd_per_day

    def _audit(self, *, day_tag: str, timestep: int, prompt_type: str,
               robot_action: str, scripted_response: str,
               system_prompt: str, user_prompt: str,
               llm_decision: str, llm_rationale: str, cache_hit: bool):
        """Append one human-readable decision block to the audit log."""
        if not self._audit_log_path:
            return
        hh = (timestep // 60) % 24
        mm = timestep % 60
        block = (
            f"\n{'=' * 78}\n"
            f"{day_tag}  t={hh:02d}:{mm:02d} ({timestep})  type={prompt_type}  "
            f"activity={self.current_activity}  cache_hit={cache_hit}\n"
            f"ROBOT ACTION    : {robot_action}\n"
            f"SCRIPTED RESPONSE: {scripted_response}\n"
            f"--- LLM SYSTEM PROMPT ---\n{system_prompt}\n"
            f"--- LLM USER PROMPT ---\n{user_prompt}\n"
            f"--- LLM RESPONSE ---\n"
            f"decision={llm_decision}  rationale={llm_rationale}\n"
        )
        with open(self._audit_log_path, 'a') as f:
            f.write(block)

    def _require_budget(self):
        """Raise (no silent fallback) when the per-day budget is exhausted."""
        if self._calls_today >= self.llm_max_calls_per_day:
            raise RuntimeError(
                f"LLM per-day live-call cap ({self.llm_max_calls_per_day}) reached. "
                "Raise max_llm_calls_per_day / max_usd_per_day and resume "
                "(cached decisions replay for free)."
            )
        cap = self._effective_budget()
        if self._client.cost_usd >= cap:
            raise RuntimeError(
                f"LLM per-day USD budget (${cap:.2f}) reached "
                f"(spent ${self._client.cost_usd:.4f}). Raise the cap — edit "
                f"the budget file {self._budget_file} (or {self._global_budget_file}) "
                "live, or set max_usd_per_day and resume (cache replays for free)."
            )

    def _charge_llm_call(self, cache_hit: bool) -> None:
        """Increment the live-call cap; cache replays do not count."""
        if not cache_hit:
            self._calls_today += 1

    def _day_tag(self) -> str:
        return getattr(self, '_current_day_tag', 'day?')

    def reset_day(self, day_tag: str | None = None):
        """Reset per-day history, LLM counters, and redirect bookkeeping."""
        self._day_history = []
        self.llm_decisions = []
        self._calls_today = 0
        self._current_day_tag = day_tag or 'day?'
        self._moved_early = set()
        self._client.reset_counters()

    def stats(self) -> dict:
        """LLM call/cache stats plus count of decisions logged today."""
        return {
            **self._client.stats(),
            'n_llm_decisions': len(self.llm_decisions),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record(
        self,
        timestep: int,
        prompt_type: str,
        decision: str,
        rationale: str,
        cache_hit: bool,
        fallback: bool,
        elapsed_ms: float,
    ):
        """Append one structured decision to ``llm_decisions``."""
        self.llm_decisions.append(
            DecisionRecord(
                timestep=timestep,
                prompt_type=prompt_type,
                decision=decision,
                rationale=rationale,
                cache_hit=cache_hit,
                fallback=fallback,
                elapsed_ms=elapsed_ms,
            ).__dict__
        )

    def _push_history(self, *, actor: str, timestep: int, obj: str,
                       loc: str, outcome: str = ''):
        """Append one human/robot event to the in-memory day history for prompts."""
        self._day_history.append({
            'actor': actor,
            'timestep': timestep,
            'object': obj,
            'location': loc,
            'outcome': outcome,
        })

    def _budget_left(self) -> bool:
        """True if both call-count and USD caps have headroom."""
        if self._calls_today >= self.llm_max_calls_per_day:
            return False
        if self._client.cost_usd >= self._effective_budget():
            return False
        return True

    def _refresh_id2name(self, world_state: dict | None):
        """Build VH node-id -> class_name from the live world_state nodes.

        ``self.node_map`` (GraphReader) is keyed by '<class_name>' strings, not
        ids, so it can't resolve a numeric id. world_state carries id+class_name.
        """
        if not world_state:
            return
        for n in world_state.get('nodes', []):
            nid = n.get('id')
            cn = n.get('class_name')
            if nid is not None and cn:
                self._id2name[int(nid)] = cn

    def _name_for_id(self, vh_id: int | None) -> str:
        if vh_id is None:
            return '?'
        try:
            vh_id = int(vh_id)
        except (TypeError, ValueError):
            return str(vh_id)
        if vh_id in self._id2name:
            return self._id2name[vh_id]
        return str(vh_id)

    # ------------------------------------------------------------------
    # Branch A / B replacement
    # ------------------------------------------------------------------

    def _execute_current_action(self, current_time: int,
                                 world_state: dict | None,
                                 remember_actions: bool = True,
                                 robot_actions: list[dict] | None = None,
                                 pending_teleports: dict[int, int] | None = None):
        """Override to inject LLM decisions at the two accommodation branches."""
        if not self.llm_active:
            return super()._execute_current_action(
                current_time, world_state, remember_actions,
                robot_actions, pending_teleports)
        self._refresh_id2name(world_state)
        if pending_teleports is None:
            pending_teleports = {}
        if robot_actions is None:
            robot_actions = []

        self.current_activity = self._activity_from_schedule(current_time)
        actions = self.script.lines[self.next_action_idx]
        if not isinstance(actions, list):
            actions = [actions]

        script_actions: list = []
        robot_action_locs = [(a['obj_idx'], a['loc_idx']) for a in robot_actions]
        robot_action_objs = [a['obj_idx'] for a in robot_actions]

        for action_str in actions:
            action_script = parse_script_line(action_str, 0)
            outcome = 'done_as_is'
            if 'TELEPORT' not in action_script.action._name_:
                outcome = 'no_movement'
                self.script_line_treatments.append((action_script, 'no_movement'))

            if action_script.action._name_ == 'TELEPORT':
                obj_id = action_script.object().instance
                dest_id = action_script.subject().instance
                obj_name = action_script.object().name
                dest_name = action_script.subject().name

                if obj_id in pending_teleports:
                    already_at_dest = (pending_teleports[obj_id] == dest_id)
                    found_loc_id = pending_teleports[obj_id]
                else:
                    already_at_dest = world_state is not None and any(
                        e['from_id'] == obj_id and e['to_id'] == dest_id
                        and e['relation_type'] in ('INSIDE', 'ON')
                        for e in world_state.get('edges', [])
                    )
                    found_loc_id = self._lookup_object_location(world_state, obj_id)

                robot_for_obj = None
                if robot_action_objs and obj_id in robot_action_objs:
                    robot_for_obj = robot_actions[robot_action_objs.index(obj_id)]

                if already_at_dest:
                    outcome, treatment = self._decide_at_dest(
                        current_time, obj_name, dest_name,
                        robot_for_obj, robot_action_locs,
                        obj_id, dest_id, robot_actions)
                    self.script_line_treatments.append((action_script, treatment))
                else:
                    decision_outcome, append_script_action = self._decide_unexpected(
                        current_time, obj_name, dest_name,
                        self._name_for_id(found_loc_id),
                        robot_for_obj, robot_action_locs,
                        obj_id, dest_id, robot_actions)
                    outcome = decision_outcome
                    self.script_line_treatments.append((action_script, decision_outcome))
                    if append_script_action:
                        script_actions.append(action_script)
                        pending_teleports[obj_id] = dest_id

                # Push to day history, but skip non-informative no-ops
                # (object already in place, nothing the human/robot did).
                if outcome not in ('no_need_to_move', 'no_movement'):
                    self._push_history(
                        actor='H', timestep=current_time,
                        obj=obj_name, loc=dest_name, outcome=outcome,
                    )

                if remember_actions:
                    self._record_action_outcome(
                        current_time, action_script, outcome, obj_id, obj_name, dest_name)
            # Match the scripted agent: non-TELEPORT actions do not produce
            # action_outcomes entries (the base class only records inside the
            # TELEPORT branch).

        self.next_action_idx += 1
        return script_actions

    # ------------------------------------------------------------------
    # Decision helpers — each calls the LLM (with scripted fallback) and
    # returns the outcome string + side effects on robot_actions.
    # ------------------------------------------------------------------

    def _decide_at_dest(self, current_time, obj_name, dest_name,
                         robot_for_obj, robot_action_locs,
                         obj_id, dest_id, robot_actions) -> tuple[str, str]:
        """Branch A: script wants obj -> dest, object is already at dest.

        Pre-compute the scripted outcome but defer any side effects on
        ``robot_actions`` until we know the LLM decision.
        """
        # The human already moved this object to this destination EARLY via a
        # redirect (v2). The object is here because of the human, not the robot
        # — so this script line is a no-op that must NOT be credited as a robot
        # 'skipped' (TP). Tag 'done_early' (excluded from metrics) and consume.
        if (int(obj_id), int(dest_id)) in self._moved_early:
            self._moved_early.discard((int(obj_id), int(dest_id)))
            return 'done_early', 'done_early'

        match_idx = None
        if robot_action_locs and (obj_id, dest_id) in robot_action_locs:
            match_idx = robot_action_locs.index((obj_id, dest_id))
            scripted_outcome = (
                'skipped'
                if robot_actions[match_idx]['human_action_outcome'] is None
                else 'skipped_repeated'
            )
        else:
            scripted_outcome = 'no_need_to_move'

        def _commit_scripted():
            if match_idx is not None and scripted_outcome == 'skipped':
                robot_actions[match_idx]['human_action_outcome'] = 'skipped'
                robot_actions[match_idx]['eval_time'] = current_time
            return scripted_outcome, scripted_outcome

        # Genuine no-op: object already at dest with no robot involvement at
        # all today → nothing to decide, no LLM call (this is not a fallback).
        if robot_for_obj is None and match_idx is None:
            return _commit_scripted()

        # A real decision is needed → the LLM must run. No silent fallback:
        # if the budget is exhausted, crash (the user wants to know).
        self._require_budget()

        system, user, schema = build_at_dest_prompt(
            activity=self.current_activity,
            planned_object=obj_name,
            planned_destination=dest_name,
            robot_recent_action=(
                {'object': obj_name, 'location': dest_name}
                if robot_for_obj else None
            ),
            day_history=self._day_history,
            examples=self._examples,
            max_history_actions=self.llm_max_history_actions,
            upcoming_plan=self._next_planned_actions(current_time, n=12),
        )
        payload, hit, ms = self._client.call(system, user, schema)
        self._charge_llm_call(hit)
        decision = payload['decision']   # strict schema guarantees presence
        rationale = payload.get('rationale', '')
        self._audit(
            day_tag=self._day_tag(), timestep=current_time, prompt_type='at_dest',
            robot_action=f"{obj_name} -> {dest_name} (matched plan)",
            scripted_response=scripted_outcome,
            system_prompt=system, user_prompt=user,
            llm_decision=decision, llm_rationale=rationale, cache_hit=hit)

        if decision == 'correct':
            # Human moves object away despite the robot getting it right —
            # unusual; tag with a distinct 'corrected_despite_correct' outcome.
            if match_idx is not None and robot_actions[match_idx]['human_action_outcome'] is None:
                robot_actions[match_idx]['human_action_outcome'] = 'corrected_despite_correct'
                robot_actions[match_idx]['eval_time'] = current_time
            self._record(current_time, 'at_dest', 'corrected_despite_correct',
                         rationale, hit, False, ms)
            return 'corrected_despite_correct', 'corrected_despite_correct'
        self._record(current_time, 'at_dest', scripted_outcome, rationale, hit, False, ms)
        return _commit_scripted()

    def _decide_unexpected(self, current_time, obj_name, dest_name,
                            found_location, robot_for_obj, robot_action_locs,
                            obj_id, dest_id, robot_actions) -> tuple[str, bool]:
        """Branch B: script wants obj -> dest, object is elsewhere.

        Returns (outcome, append_script_action).
        """
        # Compute what the scripted rule WOULD do — for the audit log and for
        # the genuine no-robot-involvement no-op case. We re-run it read-only
        # below; here we only peek without committing side effects.
        scripted_preview = self._scripted_unexpected_preview(
            obj_id, dest_id, robot_action_locs, robot_actions)

        # No robot involvement on this object → genuine 'done_as_is', no LLM.
        if robot_for_obj is None and scripted_preview == 'done_as_is':
            return self._scripted_unexpected_outcome(
                current_time, obj_id, dest_id, robot_action_locs, robot_actions)

        # Real decision needed → LLM must run; crash if budget exhausted.
        self._require_budget()

        system, user, schema = build_unexpected_prompt(
            activity=self.current_activity,
            planned_object=obj_name,
            planned_destination=dest_name,
            found_location=found_location,
            robot_recent_action=(
                {'object': obj_name,
                 'location': self._name_for_id(robot_for_obj.get('loc_idx'))}
                if robot_for_obj else None
            ),
            day_history=self._day_history,
            examples=self._examples,
            max_history_actions=self.llm_max_history_actions,
            upcoming_plan=self._next_planned_actions(current_time, n=12),
        )
        payload, hit, ms = self._client.call(system, user, schema)
        self._charge_llm_call(hit)
        decision = payload['decision']   # strict schema guarantees presence
        rationale = payload.get('rationale', '')
        self._record(current_time, 'unexpected', decision, rationale, hit, False, ms)
        self._audit(
            day_tag=self._day_tag(), timestep=current_time, prompt_type='unexpected',
            robot_action=(f"{obj_name} found at {found_location} "
                          f"(planned dest {dest_name})"),
            scripted_response=scripted_preview,
            system_prompt=system, user_prompt=user,
            llm_decision=decision, llm_rationale=rationale, cache_hit=hit)

        if decision == 'correct':
            # Tag robot action as corrected (if it matched on obj only) and
            # execute the script action.
            if robot_for_obj is not None and robot_for_obj['human_action_outcome'] is None:
                robot_for_obj['human_action_outcome'] = 'corrected'
                robot_for_obj['eval_time'] = current_time
            return 'corrected', True
        if decision == 'accept':
            # Human accepts the robot's mistake; do not execute the script
            # action. Tag the robot action with a distinct 'accepted' outcome
            # so we can later count how many "wrong" robot actions the LLM
            # chose to go with instead of fight.
            if robot_for_obj is not None and robot_for_obj['human_action_outcome'] is None:
                robot_for_obj['human_action_outcome'] = 'accepted'
                robot_for_obj['eval_time'] = current_time
            return 'accepted', False
        # decision == 'skip' — commit the scripted side effects (skipped /
        # skipped_repeated tagging on the matching robot action).
        return self._scripted_unexpected_outcome(
            current_time, obj_id, dest_id, robot_action_locs, robot_actions)

    def _scripted_unexpected_preview(self, obj_id, dest_id,
                                      robot_action_locs, robot_actions) -> str:
        """Read-only: what the scripted rule's outcome WOULD be (no side effects).

        Used for the audit log and to detect the genuine no-op case.
        """
        if not robot_actions:
            return 'done_as_is'
        robot_action_objs = [a['obj_idx'] for a in robot_actions]
        if (obj_id, dest_id) in robot_action_locs:
            idx = robot_action_locs.index((obj_id, dest_id))
            return ('skipped' if robot_actions[idx]['human_action_outcome'] is None
                    else 'skipped_repeated')
        if obj_id in robot_action_objs:
            idx = robot_action_objs.index(obj_id)
            return ('corrected' if robot_actions[idx]['human_action_outcome'] is None
                    else 'corrected_repeated')
        return 'done_as_is'

    def _scripted_unexpected_outcome(self, current_time, obj_id, dest_id,
                                      robot_action_locs, robot_actions
                                      ) -> tuple[str, bool]:
        """Reproduce the original rule's outcome for Branch B (commits side effects)."""
        if not robot_actions:
            return 'done_as_is', True
        robot_action_objs = [a['obj_idx'] for a in robot_actions]
        if (obj_id, dest_id) in robot_action_locs:
            idx = robot_action_locs.index((obj_id, dest_id))
            if robot_actions[idx]['human_action_outcome'] is None:
                robot_actions[idx]['human_action_outcome'] = 'skipped'
                robot_actions[idx]['eval_time'] = current_time
                return 'skipped', not self.accommodate_for_robot
            return 'skipped_repeated', not self.accommodate_for_robot
        if obj_id in robot_action_objs:
            idx = robot_action_objs.index(obj_id)
            if robot_actions[idx]['human_action_outcome'] is None:
                robot_actions[idx]['human_action_outcome'] = 'corrected'
                robot_actions[idx]['eval_time'] = current_time
                return 'corrected', True
            return 'corrected_repeated', True
        return 'done_as_is', True

    def _lookup_object_location(self, world_state: dict | None,
                                  obj_id: int) -> int | None:
        """Return parent node id (INSIDE/ON ``to_id``) for *obj_id*, or None."""
        if world_state is None:
            return None
        for e in world_state.get('edges', []):
            if e.get('from_id') == obj_id and e.get('relation_type') in ('INSIDE', 'ON'):
                return e.get('to_id')
        return None

    def _record_action_outcome(self, current_time, action_script, outcome,
                                 obj_id, obj_name, location_name):
        """Mirror scripted agent: append one TELEPORT outcome with tensor-eval tags."""
        is_teleport = (action_script.action._name_ == 'TELEPORT')
        teleport_obj_id = int(obj_id) if (is_teleport and obj_id is not None) else None
        slate_idx = (
            self._vh_id_to_slate_idx.get(teleport_obj_id)
            if teleport_obj_id is not None else None
        )
        if slate_idx is not None:
            cats = self._per_obj_tensor_category.get(int(slate_idx)) or {
                '1step': 'no_event', 'anyFuture': 'no_event'}
            cat_1s = cats['1step']
            cat_af = cats['anyFuture']
        else:
            cat_1s = cat_af = 'unknown'
        self.action_outcomes.append({
            'timestep': current_time,
            'action': str(action_script),
            'object': obj_name,
            'object_id': teleport_obj_id,
            'slate_obj_idx': int(slate_idx) if slate_idx is not None else None,
            'location': location_name,
            'activity': self.current_activity,
            'outcome': outcome,
            'tensor_eval_category_1step': cat_1s,
            'tensor_eval_category_anyFuture': cat_af,
        })
        self.action_results[outcome] = self.action_results.get(outcome, 0) + 1

    # ------------------------------------------------------------------
    # Vigilance replacement
    # ------------------------------------------------------------------

    def review_robot_action(self, robot_action, world_state: dict, current_time: int,
                             day_history: list[dict] | None = None
                             ) -> tuple[str, dict | None]:
        if not self.llm_active:
            return super().review_robot_action(
                robot_action, world_state, current_time, day_history)

        self._refresh_id2name(world_state)
        # Push the robot action onto day history before deciding (the LLM sees it)
        obj_name = getattr(robot_action, 'object_name', None) or '?'
        obj_id = getattr(robot_action, 'object_id', None)
        loc_id = getattr(robot_action, 'dest_id', None)
        # RobotMoveAction carries a human-readable dest_name; prefer it over a
        # node-id lookup so the prompt reads "wine -> wine_rack", not "wine -> 273".
        loc_name = getattr(robot_action, 'dest_name', None) or self._name_for_id(loc_id)
        self._push_history(
            actor='R', timestep=current_time, obj=obj_name, loc=loc_name,
        )

        # No silent fallback — crash if the budget is exhausted.
        self._require_budget()

        # Scripted baseline for the audit: what legacy vigilance WOULD decide
        # (read-only; vigilance is disabled in LLM mode).
        scripted_baseline = 'allow'
        if obj_id is not None and loc_id is not None:
            if not self.has_matching_future_action(current_time, obj_id, loc_id):
                scripted_baseline = 'undo'

        # In redirect (v2) mode, find where the human plans to use this object
        # next within the lookahead window — that's the redirect target.
        redirect_target = None
        if self.llm_observe_mode == 'redirect' and obj_id is not None:
            redirect_target = self._next_planned_dest_for(int(obj_id), current_time)

        upcoming = self._next_planned_actions(
            current_time, n=10,
            window_min=(self.llm_redirect_window_min
                        if self.llm_observe_mode == 'redirect' else None))
        system, user, schema = build_observe_prompt(
            activity=self.current_activity,
            robot_object=obj_name,
            robot_destination=loc_name,
            upcoming_plan=upcoming,
            day_history=self._day_history,
            examples=self._examples,
            max_history_actions=self.llm_max_history_actions,
            mode=self.llm_observe_mode,
            redirect_destination=(redirect_target[0] if redirect_target else None),
            plan_window_min=(self.llm_redirect_window_min
                             if self.llm_observe_mode == 'redirect' else None),
        )
        payload, hit, ms = self._client.call(system, user, schema)
        self._charge_llm_call(hit)
        decision = payload['decision']   # strict schema guarantees presence
        rationale = payload.get('rationale', '')

        # A 'redirect' with no concrete planned destination in-window can't be
        # executed — treat it as 'undo' (reverse), and note it.
        if decision == 'redirect' and redirect_target is None:
            rationale = '[redirect chosen but no planned dest in window -> undo] ' + rationale
            decision = 'undo'
        # If the robot already placed the object exactly at its planned
        # destination, "redirecting" there is a no-op — the robot got it right.
        # Treat as 'allow' so it scores as a skip (TP), not a redirect (FP).
        elif (decision == 'redirect' and redirect_target is not None
              and loc_id is not None and int(loc_id) == int(redirect_target[1])):
            rationale = '[robot already at planned dest -> allow] ' + rationale
            decision = 'allow'

        record = DecisionRecord(
            timestep=current_time,
            prompt_type='observe',
            decision=decision,
            rationale=rationale,
            cache_hit=hit,
            fallback=False,
            elapsed_ms=ms,
        ).__dict__
        if decision == 'redirect':
            tgt_name, tgt_id = redirect_target
            self._moved_early.add((int(obj_id), int(tgt_id)))
            record['redirect_target'] = {'name': tgt_name, 'id': int(tgt_id)}
        self.llm_decisions.append(record)
        self._audit(
            day_tag=self._day_tag(), timestep=current_time, prompt_type='observe',
            robot_action=f"{obj_name} -> {loc_name}",
            scripted_response=f"vigilance-would={scripted_baseline}",
            system_prompt=system, user_prompt=user,
            llm_decision=(f"redirect->{redirect_target[0]}" if decision == 'redirect' else decision),
            llm_rationale=rationale, cache_hit=hit)
        return decision, record

    def _next_planned_actions(self, current_time: int, n: int = 10,
                               window_min: int | None = None) -> list[dict]:
        """Upcoming TELEPORTs. If *window_min* is set, only those whose
        timestamp falls within current_time + window_min are returned."""
        out: list[dict] = []
        t_max = current_time + window_min if window_min is not None else None
        for idx in range(self.next_action_idx, len(self.script.timestamps)):
            t = self.script.timestamps[idx]
            if t_max is not None and t > t_max:
                break
            for action_str in self.script.lines[idx]:
                a = parse_script_line(action_str, 0)
                if a.action._name_ == 'TELEPORT':
                    out.append({
                        'time': int(t),
                        'object': a.object().name,
                        'location': a.subject().name,
                    })
                if window_min is None and len(out) >= n:
                    return out
        return out

    def _next_planned_dest_for(self, obj_id: int, current_time: int):
        """Return (dest_name, dest_id) of the next planned TELEPORT of *obj_id*
        within the redirect window, or None. Used as the redirect target."""
        t_max = current_time + self.llm_redirect_window_min
        for idx in range(self.next_action_idx, len(self.script.timestamps)):
            t = self.script.timestamps[idx]
            if t > t_max:
                break
            for action_str in self.script.lines[idx]:
                a = parse_script_line(action_str, 0)
                if a.action._name_ == 'TELEPORT' and a.object().instance == obj_id:
                    return a.subject().name, a.subject().instance
        return None
