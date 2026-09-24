"""LLM memory curator for the hybrid robot policy.

Training time: processes warmup days (gen-dataset .pt files) day-by-day. The
LLM does only the semantic work — segmenting each day's object-relocation
stream, naming activities in an open vocabulary it curates (adding/merging
names across days). Deterministic code then builds the branching task graphs
(agents/task_graph.py) and activity-name embeddings from the tagged output.

Eval time: tags a partial day against the FROZEN vocabulary (no curation).

GT-contamination guard: the .pt files carry an 'activity' field derived from
ground-truth labels. This module never reads it — the policy must discover
activities from the relocation stream alone.

Memory dir layout:
    activities.json        curated vocabulary + per-activity aggregates
    tagged_days/NNN.json   per-day events + tagged segments
    embeddings.json        activity-name embeddings (plain JSON, not diskcache)
    task_graphs.json       serialized TaskGraphSet
    curation_state.json    resume bookkeeping
"""

import glob
import hashlib
import json
import os

import torch

from agents.llm_human_prompt import LLMHumanClient
from agents.task_graph import TaskGraphSet


# ---------------------------------------------------------------------------
# Event extraction from gen-dataset .pt files
# ---------------------------------------------------------------------------

def extract_events_from_pt(pt_path: str) -> dict:
    """Net object moves per 10-min tick, in slate index space.

    Reads only edges/times/active_edges/class_names/node_ids. The 'activity'
    key is ground truth and is deliberately never touched.
    """
    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    edges = d['edges']            # (T, N, N) one-hot rows
    times = d['times']            # (T,)
    movable = d['active_edges'].sum(-1) > 0   # (N,)
    class_names = list(d['class_names'])
    locs = edges.argmax(dim=-1)   # (T, N)
    events = []
    for t in range(1, edges.shape[0]):
        changed = ((locs[t] != locs[t - 1]) & movable).nonzero(as_tuple=False)
        for i in changed.squeeze(-1).tolist():
            events.append({
                't_min': float(times[t]),
                'obj_idx': int(i),
                'dest_idx': int(locs[t, i]),
                'prev_idx': int(locs[t - 1, i]),
                'obj_class': class_names[i],
                'dest_class': class_names[int(locs[t, i])],
                'prev_class': class_names[int(locs[t - 1, i])],
            })
    return {'events': events,
            'class_names': class_names,
            'node_ids': list(d['node_ids'])}


def _hhmm(minutes: float) -> str:
    """Format minutes-since-midnight as ``HH:MM`` (wraps at 24h)."""
    m = int(round(minutes)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def render_events(events: list) -> str:
    """One line per event: ``idx. HH:MM obj: prev -> dest``."""
    return "\n".join(
        f"{i}. {_hhmm(ev['t_min'])} {ev['obj_class']}: "
        f"{ev['prev_class']} -> {ev['dest_class']}"
        for i, ev in enumerate(events))


# ---------------------------------------------------------------------------
# Prompts + schemas
# ---------------------------------------------------------------------------

CURATE_SYSTEM_PROMPT = (
    "You annotate a single person's household routine from object-relocation "
    "logs (the only sensor available — no vision, no audio). Each line is one "
    "observed move: time, object, from-location -> to-location. The day runs "
    "06:00-24:00. Your job: partition the day's numbered events into "
    "contiguous segments, each corresponding to one activity episode (e.g. a "
    "meal, dishwashing, getting ready), and name each segment with a short "
    "snake_case activity name. REUSE an existing vocabulary name whenever the "
    "episode is the same kind of activity; create a new name only when "
    "nothing in the vocabulary fits; propose a merge when you realize two "
    "vocabulary entries denote the same underlying activity; propose a rename "
    "when an existing name no longer describes what you have observed (e.g. "
    "cereal_for_breakfast -> cereal_or_oatmeal_breakfast after seeing oatmeal "
    "days). Names are encoded by a language model downstream, so make them "
    "semantically meaningful and self-describing — say what the person is "
    "doing and with what (prefer 'wash_dinner_dishes' over 'cleanup_2')."
)

def _curate_schema() -> dict:
    """OpenAI strict JSON schema for one-day curation response."""
    return {
        "name": "curate_day",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["segments", "new_activities", "merges", "renames"],
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["first_event_idx", "last_event_idx",
                                     "activity_name", "is_new_activity"],
                        "properties": {
                            "first_event_idx": {"type": "integer"},
                            "last_event_idx": {"type": "integer"},
                            "activity_name": {"type": "string"},
                            "is_new_activity": {"type": "boolean"},
                        },
                    },
                },
                "new_activities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "description"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "merges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["keep", "remove", "reason"],
                        "properties": {
                            "keep": {"type": "string"},
                            "remove": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                    },
                },
                "renames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["old", "new", "new_description", "reason"],
                        "properties": {
                            "old": {"type": "string"},
                            "new": {"type": "string"},
                            "new_description": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        },
    }

## Currently supports only Non-Overlapping segments. TODO: Support Overlapping segments.
# ---------------------------------------------------------------------------
# Validation (crash over fallback: one repair re-prompt, then raise)
# ---------------------------------------------------------------------------

def _sanitize_segments(segments: list, n_events: int) -> list:
    """Deterministically repair common LLM index sloppiness on partial days:
    sort by first_event_idx, clamp to [0, n_events-1], clip overlaps to the
    previous segment's end, drop segments emptied by clipping. Prints when
    anything was changed."""
    segs = sorted((dict(s) for s in segments),
                  key=lambda s: (s['first_event_idx'], s['last_event_idx']))
    out, prev_last, changed = [], -1, False
    for s in segs:
        if s['first_event_idx'] <= prev_last:
            s['first_event_idx'] = prev_last + 1
            changed = True
        if s['first_event_idx'] < 0 or s['last_event_idx'] > n_events - 1:
            s['first_event_idx'] = max(0, s['first_event_idx'])
            s['last_event_idx'] = min(n_events - 1, s['last_event_idx'])
            changed = True
        if s['last_event_idx'] < s['first_event_idx']:
            changed = True
            continue
        out.append(s)
        prev_last = s['last_event_idx']
    if changed:
        print("  [tagger] sanitized overlapping/empty/out-of-range segments "
              "in LLM output")
    return out


# ---------------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------------

class MemoryCurator:
    """Train-time LLM curator: segment days, maintain vocab, build graphs.

    Eval-time recognition is handled by ``GraphParticleFilter``, not this class.
    """

    def __init__(self, llm_cfg: dict, embedding_cfg: dict, memory_dir: str,
                 cl_cfg: dict = None):
        self.memory_dir = memory_dir
        self.embedding_model = embedding_cfg['model']
        self.cl_cfg = cl_cfg or {}      # v2-PF continual learning config
        self.last_gamma = None
        os.makedirs(os.path.join(memory_dir, 'tagged_days'), exist_ok=True)
        cache_dir = llm_cfg['cache_dir']
        self.curate_client = LLMHumanClient(
            model=llm_cfg['curate_model'],
            cache_dir=cache_dir,
            provider=llm_cfg.get('provider', 'openai'),
        )
        self.max_usd = float(llm_cfg.get('max_usd', 30.0))

    # ----------------------------------------------------------- persistence

    def _path(self, *parts) -> str:
        """Join path under ``memory_dir``."""
        return os.path.join(self.memory_dir, *parts)

    def load_activities(self) -> dict:
        """Load ``activities.json`` vocab or return empty scaffold."""
        p = self._path('activities.json')
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
        return {'version': 1, 'activities': [], 'aliases': {}}

    def _save_activities(self, acts: dict) -> None:
        """Write ``activities.json``."""
        with open(self._path('activities.json'), 'w') as f:
            json.dump(acts, f, indent=2)

    def vocab_names(self) -> list:
        """Sorted list of curated activity name strings."""
        return [a['name'] for a in self.load_activities()['activities']]

    def load_tagged_days(self) -> list:
        """Load all ``tagged_days/*.json`` in day order."""
        days = []
        for p in sorted(glob.glob(self._path('tagged_days', '*.json'))):
            with open(p) as f:
                days.append(json.load(f))
        return days

    def _check_budget(self):
        """Raise if curator LLM spend exceeds ``max_usd``."""
        spent = self.curate_client.cost_usd
        if spent > self.max_usd:
            raise RuntimeError(
                f"LLM curator exceeded max_usd={self.max_usd:.2f} "
                f"(spent ~${spent:.2f}); raise robot.hybrid.llm.max_usd to continue")

    # ------------------------------------------------------- prompt building

    def _vocab_block(self, acts: dict, tagged_days: list) -> str:
        if not acts['activities']:
            return "(empty — this is the first day; every activity will be new)"
        # one compact example trace per entry: first 5 actions of its most
        # recent tagged segment
        example = {}
        start_sum, start_n = {}, {}
        for day in tagged_days:
            for seg in day['segments']:
                name = seg['activity_name']
                evs = [day['events'][i] for i in seg['event_idxs']]
                if not evs:
                    continue
                example[name] = "; ".join(
                    f"{_hhmm(e['t_min'])} {e['obj_class']} -> {e['dest_class']}"
                    for e in evs[:5])
                start_sum[name] = start_sum.get(name, 0.0) + evs[0]['t_min']
                start_n[name] = start_n.get(name, 0) + 1
        lines = []
        for a in acts['activities']:
            name = a['name']
            extra = ""
            if start_n.get(name):
                extra = f" (typical start ~{_hhmm(start_sum[name] / start_n[name])}, seen {start_n[name]}x)"
            lines.append(f"- {name}: {a['description']}{extra}")
            if name in example:
                lines.append(f"    example: {example[name]}")
        return "\n".join(lines)

    def _curate_user_prompt(self, day_idx: int, events: list, acts: dict,
                            tagged_days: list) -> str:
        return (
            "## Current activity vocabulary (reuse these names whenever possible)\n"
            f"{self._vocab_block(acts, tagged_days)}\n\n"
            f"## Today's observed object relocations (day {day_idx})\n"
            f"{render_events(events)}\n\n"
            "## Instructions\n"
            "- Group events 0..{last} into contiguous, non-overlapping "
            "segments (use first_event_idx/last_event_idx). Label every event "
            "that belongs to an activity; an isolated event that clearly "
            "interrupts an activity it does not belong to (e.g. a stray "
            "put-away in the middle of TV time) may be left out of all "
            "segments rather than absorbed into the surrounding one.\n"
            "- Consecutive events that serve the same purpose belong to one "
            "segment, even with small time gaps; a long gap or a change of "
            "purpose starts a new segment.\n"
            "- Name each segment with an existing vocabulary name when the "
            "episode is the same kind of activity; otherwise create a new "
            "snake_case name (declare it in new_activities with a one-line "
            "description, and set is_new_activity on those segments). Make "
            "names semantically meaningful and self-describing — they are "
            "encoded by a language model downstream.\n"
            "- If two existing vocabulary entries are really the same "
            "activity, propose a merge (keep, remove, reason).\n"
            "- If an existing name no longer fits what you have observed, "
            "propose a rename (old, new, new_description, reason) — e.g. "
            "cereal_for_breakfast -> cereal_or_oatmeal_breakfast. Tag "
            "today's segments with the NEW name.\n"
        ).replace("{last}", str(len(events) - 1))

    # ------------------------------------------------------------- curation

    def curate_day(self, day_idx: int, day_data: dict, source_pt: str) -> dict:
        """Curate one day; writes tagged_days/NNN.json and updates the vocab.
        Returns the tagged-day dict."""
        events = day_data['events']
        if not events:
            raise RuntimeError(f"day {day_idx}: no events extracted from {source_pt}")
        acts = self.load_activities()
        tagged_days = self.load_tagged_days()
        vocab = {a['name'] for a in acts['activities']}
        system = CURATE_SYSTEM_PROMPT
        user = self._curate_user_prompt(day_idx, events, acts, tagged_days)
        schema = _curate_schema()
        payload, _, _ = self.curate_client.call(system, user, schema)
        # LLM index sloppiness is sanitized deterministically (and loudly),
        # never crashed on: over 65 days x 3 households some outputs WILL be
        # imperfect. Crashes are reserved for our own bugs (empty output,
        # empty vocabulary, budget).
        payload['segments'] = _sanitize_segments(payload['segments'], len(events))
        if not payload['segments']:
            repair_user = (user + "\n\n## Your previous answer was invalid\n"
                           f"Previous answer: {json.dumps(payload)}\n"
                           "Validator error: no usable segments\n"
                           "Return a corrected answer.")
            payload, _, _ = self.curate_client.call(system, repair_user, schema)
            payload['segments'] = _sanitize_segments(payload['segments'], len(events))
            if not payload['segments']:
                raise RuntimeError(
                    f"day {day_idx}: curation returned no usable segments "
                    f"after repair")
        self._check_budget()
        n_covered = sum(s['last_event_idx'] - s['first_event_idx'] + 1
                        for s in payload['segments'])
        if n_covered < len(events):
            print(f"  [curator] day {day_idx}: {len(events) - n_covered} of "
                  f"{len(events)} events left unassigned by the LLM")

        # The LLM sometimes references names that no longer exist (already
        # merged/renamed on an earlier day). Resolve through the alias
        # history; drop edits that still don't resolve, loudly.
        aliases = acts['aliases']

        def _resolve(name: str) -> str:
            seen = set()
            while name not in vocab and name in aliases and name not in seen:
                seen.add(name)
                name = aliases[name]
            return name

        # vocabulary: add new activities
        for a in payload['new_activities']:
            if a['name'] not in vocab:
                acts['activities'].append({
                    'name': a['name'], 'description': a['description'],
                    'created_day': day_idx})
                vocab.add(a['name'])

        # renames: old -> new (entry kept, name + description updated);
        # a rename onto an existing name is really a merge
        rename = {}
        for r in payload['renames']:
            old = _resolve(r['old'])
            if old not in vocab:
                print(f"  [curator] day {day_idx} dropping rename of unknown "
                      f"activity '{r['old']}'")
                continue
            if r['new'] == old:
                continue
            if r['new'] in vocab:
                print(f"  [curator] day {day_idx} rename {old} -> {r['new']} "
                      f"targets an existing name; treating as merge "
                      f"({r['reason']})")
                acts['activities'] = [a for a in acts['activities']
                                      if a['name'] != old]
                vocab.discard(old)
                rename[old] = r['new']
                continue
            print(f"  [curator] day {day_idx} rename: {old} -> "
                  f"{r['new']} ({r['reason']})")
            for a in acts['activities']:
                if a['name'] == old:
                    a['name'] = r['new']
                    a['description'] = r['new_description']
            vocab.discard(old)
            vocab.add(r['new'])
            rename[old] = r['new']

        # merges: remove -> keep (removed entry deleted)
        for m in payload['merges']:
            remove = _resolve(m['remove'])
            keep = _resolve(m['keep'])
            while remove in rename:
                remove = rename[remove]
            while keep in rename:
                keep = rename[keep]
            if remove == keep or remove not in vocab or keep not in vocab:
                print(f"  [curator] day {day_idx} dropping invalid merge "
                      f"'{m['remove']}' -> '{m['keep']}'")
                continue
            if len(vocab) <= 1:
                print(f"  [curator] day {day_idx} dropping merge that would "
                      f"empty the vocabulary")
                continue
            print(f"  [curator] day {day_idx} merge: {remove} -> "
                  f"{keep} ({m['reason']})")
            acts['activities'] = [a for a in acts['activities']
                                  if a['name'] != remove]
            vocab.discard(remove)
            rename[remove] = keep
        if rename:
            acts['aliases'].update(rename)
            for p in sorted(glob.glob(self._path('tagged_days', '*.json'))):
                with open(p) as f:
                    td = json.load(f)
                changed = False
                for seg in td['segments']:
                    if seg['activity_name'] in rename:
                        seg['activity_name'] = rename[seg['activity_name']]
                        changed = True
                if changed:
                    with open(p, 'w') as f:
                        json.dump(td, f, indent=2)

        # tagged day (after renames so today's segments use final names)
        segments = []
        for s in sorted(payload['segments'], key=lambda s: s['first_event_idx']):
            name = s['activity_name']
            while name in rename:
                name = rename[name]
            name = _resolve(name)
            if name not in vocab:
                # used in a segment but never declared: auto-add, loudly
                print(f"  [curator] day {day_idx} auto-adding undeclared "
                      f"activity '{name}'")
                acts['activities'].append({
                    'name': name,
                    'description': f"auto-added from an undeclared segment "
                                   f"label on day {day_idx}",
                    'created_day': day_idx})
                vocab.add(name)
            idxs = list(range(s['first_event_idx'], s['last_event_idx'] + 1))
            segments.append({
                'activity_name': name,
                'start_min': events[idxs[0]]['t_min'],
                'end_min': events[idxs[-1]]['t_min'],
                'event_idxs': idxs,
            })
        tagged = {'day': day_idx, 'source_pt': source_pt,
                  'events': events, 'segments': segments}
        with open(self._path('tagged_days', f'{day_idx:03d}.json'), 'w') as f:
            json.dump(tagged, f, indent=2)

        # refresh per-activity segment counts
        counts = {}
        for td in self.load_tagged_days():
            for seg in td['segments']:
                counts[seg['activity_name']] = counts.get(seg['activity_name'], 0) + 1
        for a in acts['activities']:
            a['n_segments'] = counts.get(a['name'], 0)
        self._save_activities(acts)
        return tagged

    def curate_dataset(self, pt_dir: str, max_days: int = None) -> None:
        """Curate all days in <pt_dir>/*.pt (sorted), with resume; then build
        embeddings + task graphs. max_days limits to the first N days (smokes)."""
        pt_files = sorted(glob.glob(os.path.join(pt_dir, '*.pt')))
        if max_days is not None:
            pt_files = pt_files[:max_days]
        if not pt_files:
            raise RuntimeError(f"no .pt files found in {pt_dir}")
        state_p = self._path('curation_state.json')
        state = {'last_day_curated': -1, 'source_dir': pt_dir}
        if os.path.isfile(state_p):
            with open(state_p) as f:
                state = json.load(f)
            if state['source_dir'] != pt_dir:
                raise RuntimeError(
                    f"memory dir {self.memory_dir} was curated from "
                    f"{state['source_dir']}, not {pt_dir}; use a fresh memory dir")
        for day_idx, pt in enumerate(pt_files):
            tagged_p = self._path('tagged_days', f'{day_idx:03d}.json')
            if day_idx <= state['last_day_curated'] and os.path.isfile(tagged_p):
                continue
            day_data = extract_events_from_pt(pt)
            tagged = self.curate_day(day_idx, day_data, pt)
            state['last_day_curated'] = day_idx
            with open(state_p, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"  [curator] day {day_idx}: {len(day_data['events'])} events, "
                  f"{len(tagged['segments'])} segments, vocab={len(self.vocab_names())}, "
                  f"spent=${self.curate_client.cost_usd:.2f}")
        # Prune vocabulary entries with zero tagged segments (e.g. orphaned by
        # merges): they would be offered to the eval-time tagger but have no
        # task graph and no transformer class.
        acts = self.load_activities()
        orphans = [a['name'] for a in acts['activities']
                   if a.get('n_segments', 0) == 0]
        if orphans:
            print(f"  [curator] pruning zero-segment activities: {orphans}")
            acts['activities'] = [a for a in acts['activities']
                                  if a['name'] not in orphans]
            self._save_activities(acts)
        self.build_embeddings()
        self.build_task_graphs(cl_cfg=self.cl_cfg)

    # ------------------------------------------------- embeddings + graphs

    def build_embeddings(self) -> dict:
        """Fetch OpenAI embeddings for all vocab names; plain-JSON cache."""
        names = self.vocab_names()
        p = self._path('embeddings.json')
        data = {'model': self.embedding_model, 'dim': None, 'vectors': {}}
        if os.path.isfile(p):
            with open(p) as f:
                data = json.load(f)
            if data['model'] != self.embedding_model:
                raise RuntimeError(
                    f"embeddings.json was built with {data['model']}, "
                    f"config wants {self.embedding_model}")
        missing = [n for n in names if n not in data['vectors']]
        if missing:
            from openai import OpenAI
            resp = OpenAI().embeddings.create(model=self.embedding_model,
                                              input=missing)
            for name, item in zip(missing, resp.data):
                data['vectors'][name] = item.embedding
            data['dim'] = len(resp.data[0].embedding)
            with open(p, 'w') as f:
                json.dump(data, f)
            print(f"  [curator] embedded {len(missing)} activity names "
                  f"({self.embedding_model}, dim={data['dim']})")
        return data

    def build_task_graphs(self, cl_cfg: dict = None) -> TaskGraphSet:
        """Rebuild graphs from scratch from tagged_days (merges propagate).

        cl_cfg (v2-PF continual learning): {'enabled': bool, 'gamma_grid':
        [...]} — prequentially select a per-day decay gamma, build with
        recency weights gamma^age, attribute robot events from the policy's
        attributed_days/ log, and append the chosen gamma to
        gamma_log.jsonl (plot with scripts/plot_gamma.py)."""
        tagged = self.load_tagged_days()
        weights = None
        if cl_cfg and cl_cfg.get('enabled'):
            from agents.task_graph import select_gamma
            n = len(tagged)
            # Ablation cl_forget=False: pin gamma to 1.0 (no recency discount on
            # accumulated evidence). CL still retrains/rebuilds each cycle and
            # attributes robot events; only the forgetting decay is removed.
            if cl_cfg.get('cl_forget', True) is False:
                gamma = 1.0
                sel = {'gamma': 1.0, 'scores': None}
            else:
                sel = select_gamma(tagged, cl_cfg.get('gamma_grid'))
                gamma = sel['gamma']
            # graph_recency=False (ablation): build task graphs with UNIFORM day
            # weights (the optimized gamma is NOT applied to the graphs), while
            # last_gamma still carries gamma so the transformer can use it. This
            # decouples where recency is applied (graphs vs transformer).
            if cl_cfg.get('graph_recency', True):
                weights = [gamma ** (n - 1 - d) for d in range(n)]
            else:
                weights = None
            n_attr = self._attribute_robot_events(tagged)
            with open(self._path('gamma_log.jsonl'), 'a') as f:
                f.write(json.dumps({'n_days': n, 'gamma': gamma,
                                    'scores': sel['scores'],
                                    'robot_events_attributed': n_attr}) + '\n')
            self.last_gamma = gamma
            print(f"  [curator] v2-PF CL: gamma={gamma} over {n} days "
                  f"(half-life {0.693/max(1e-9, 1-gamma):.0f}d), "
                  f"{n_attr} robot events attributed")
        gs = TaskGraphSet.build_from_tagged_days(tagged, day_weights=weights)
        gs.save(self._path('task_graphs.json'))
        print(f"  [curator] task graphs: "
              f"{ {n: round(g.n_segments, 2) for n, g in gs.graphs.items()} }")
        return gs

    def _attribute_robot_events(self, tagged: list) -> int:
        """v2-PF: mark tagged-day events as actor='robot' by joining against
        the policy's attributed_days/<counter>.json files (written at each
        day boundary). Offset between dataset day index and the policy's day
        counter is detected by match-rate; unmatched events stay human."""
        import glob as _glob
        attr_files = sorted(_glob.glob(self._path('attributed_days', '*.json')))
        if not attr_files:
            return 0
        attr = [json.load(open(p)) for p in attr_files]
        # robot moves surface in the world stream one tick AFTER emission —
        # accept both t and t+10 when marking. Offset detection uses HUMAN
        # events only (those match exactly).
        robot_keys = [{(round(e['t_min']) + dt, e['obj_idx'], e['dest_idx'])
                       for e in a if e.get('actor') == 'robot'
                       for dt in (0, 10)} for a in attr]
        # reverted / superseded robot keys (clean re-curation): flagged on the
        # policy's attributed_days log at day boundary.
        rev_keys = [{(round(e['t_min']) + dt, e['obj_idx'], e['dest_idx'])
                     for e in a if e.get('actor') == 'robot' and e.get('reverted')
                     for dt in (0, 10)} for a in attr]
        sup_keys = [{(round(e['t_min']) + dt, e['obj_idx'], e['dest_idx'])
                     for e in a if e.get('actor') == 'robot' and e.get('superseded')
                     for dt in (0, 10)} for a in attr]
        all_keys = [{(round(e['t_min']), e['obj_idx'], e['dest_idx'])
                     for e in a if e.get('actor') != 'robot'} for a in attr]

        def match_rate(off):
            hits = tot = 0
            for ci in range(len(attr)):
                di = off + ci
                if di < 0 or di >= len(tagged):
                    continue
                ev = tagged[di]['events']
                keys = {(round(e['t_min']), e['obj_idx'], e['dest_idx'])
                        for e in ev}
                tot += max(len(all_keys[ci]), 1)
                hits += len(keys & all_keys[ci])
            return hits / max(tot, 1)

        # attributed files accumulate DAILY while curation runs per round, so
        # n_attr can exceed n_tagged — search the full plausible offset range
        # (attr day 0 = first eval day = tagged index warmup_days).
        best_off, best_r = 0, -1.0
        for off in range(-len(attr), len(tagged)):
            r = match_rate(off)
            if r > best_r:
                best_off, best_r = off, r
        if best_r < 0.3:
            # No robot events in the attributed stream → nothing to attribute;
            # a low match rate is meaningless here, so return quietly.
            n_robot_keys = sum(len(rk) for rk in robot_keys)
            if n_robot_keys == 0:
                return 0
            # Robot events DO exist but the attributed_days↔tagged-days join
            # failed. Silently leaving every event 'human' would disable clean
            # re-curation / actor-split without any signal — crash instead so
            # the broken join is fixed rather than masked (crashes over fallbacks).
            raise RuntimeError(
                f"[curator] v2-PF robot-event attribution failed: best offset "
                f"match rate {best_r:.2f} < 0.3 across {len(attr)} attributed "
                f"day(s) carrying {n_robot_keys} robot keys. The "
                f"attributed_days <-> tagged-days alignment is broken; clean "
                f"re-curation / actor-split counts would silently fall back to "
                f"human-only. Fix the day alignment instead of degrading silently."
            )
        n_attr = 0
        for ci in range(len(attr)):
            di = best_off + ci
            if di < 0 or di >= len(tagged):
                continue
            for e in tagged[di]['events']:
                k = (round(e['t_min']), e['obj_idx'], e['dest_idx'])
                if k in robot_keys[ci]:
                    e['actor'] = 'robot'
                    if k in rev_keys[ci]:
                        e['reverted'] = True
                    if k in sup_keys[ci]:
                        e['superseded'] = True
                    n_attr += 1
        return n_attr

    # NOTE: eval-time recognition (LLM tagger / ranker) was removed — the
    # particle filter (agents/activity_particle_filter.py) replaced it. This
    # class is now a TRAIN-TIME component only (curation -> graphs). See
    # docs/failed_experiments.md "LLM+transformer Hybrid model".

    def stats(self) -> dict:
        """Curator LLM client stats wrapper."""
        return {'curate': self.curate_client.stats()}
