"""LLM-driven human agent: prompt builders, OpenAI client wrapper, disk cache.

Three decision points are exposed; each returns a structured JSON enum:

  * ``decide_at_unexpected_location`` — script wants obj X -> L, but X is
    somewhere else. LLM returns ``"correct"`` | ``"accept"`` | ``"skip"``.
  * ``decide_at_already_at_dest`` — script wants obj X -> L, X already at L
    (robot got there first). LLM returns ``"skip"`` | ``"correct"``.
  * ``decide_observe_robot_action`` — vigilance replacement. Robot just acted
    on obj X -> L. LLM returns ``"undo"`` | ``"allow"``.

Each call is keyed on the full prompt text plus (provider, model) and cached
to disk so resumed runs and re-runs reuse decisions verbatim.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field

try:
    import diskcache  # type: ignore
except ImportError:  # pragma: no cover
    diskcache = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


# Rough USD pricing per 1M tokens, model -> (input, output).
# Updated 2026-06-08. Used only for an approximate per-day budget cap; not for
# billing accuracy.
_MODEL_PRICING_USD_PER_1M = {
    "gpt-4o-mini":   (0.150, 0.600),
    "gpt-4o":        (2.500, 10.000),
    "gpt-4-turbo":   (10.000, 30.000),
}
# Approx chars-per-token (English text).
_CHARS_PER_TOKEN = 4


def estimate_cost_usd(model: str, n_input_chars: int, n_output_chars: int) -> float:
    """Rough USD cost from char counts (or token counts × ``_CHARS_PER_TOKEN``)."""
    in_per_1m, out_per_1m = _MODEL_PRICING_USD_PER_1M.get(
        model, _MODEL_PRICING_USD_PER_1M["gpt-4o-mini"])
    n_in = n_input_chars / _CHARS_PER_TOKEN
    n_out = n_output_chars / _CHARS_PER_TOKEN
    return (n_in / 1_000_000) * in_per_1m + (n_out / 1_000_000) * out_per_1m


class LLMInsufficientFundsError(RuntimeError):
    """Raised when the OpenAI account is out of quota/credits.

    Surfaced immediately (no retries) so the user knows to top up rather
    than waiting through pointless backoff.
    """


def _is_insufficient_funds(exc: Exception) -> bool:
    """Best-effort detection of an out-of-credit / quota error."""
    code = getattr(exc, "code", None)
    if code in ("insufficient_quota", "billing_hard_limit_reached"):
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if err.get("code") in ("insufficient_quota", "billing_hard_limit_reached"):
            return True
    text = str(exc).lower()
    return "insufficient_quota" in text or "exceeded your current quota" in text


SYSTEM_PROMPT = (
    "You are simulating a human in a household where a robot assistant "
    "sometimes acts on its own. When the robot's action diverges from your "
    "plan, decide what the human would actually do. Adapt naturally: "
    "sometimes correct the robot, sometimes go along with what it did, "
    "sometimes skip the planned step entirely. Output one decision and a "
    "one-sentence rationale."
)

# JSON schemas per decision point
_SCHEMA_UNEXPECTED = {
    "name": "human_decision_unexpected",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "rationale"],
        "properties": {
            "decision": {"type": "string", "enum": ["correct", "accept", "skip"]},
            "rationale": {"type": "string"},
        },
    },
}
_SCHEMA_AT_DEST = {
    "name": "human_decision_at_dest",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "rationale"],
        "properties": {
            "decision": {"type": "string", "enum": ["skip", "correct"]},
            "rationale": {"type": "string"},
        },
    },
}
_SCHEMA_OBSERVE = {
    "name": "human_decision_observe",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "rationale"],
        "properties": {
            "decision": {"type": "string", "enum": ["undo", "allow"]},
            "rationale": {"type": "string"},
        },
    },
}
# v2 (observe_mode='redirect'): adds a third option — move the object straight
# to where it's planned to go next, in one trip, instead of undo-then-redo.
_SCHEMA_OBSERVE_REDIRECT = {
    "name": "human_decision_observe_redirect",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "rationale"],
        "properties": {
            "decision": {"type": "string", "enum": ["undo", "allow", "redirect"]},
            "rationale": {"type": "string"},
        },
    },
}


@dataclass
class DecisionRecord:
    """One LLM decision; persisted in per_day records."""

    timestep: int
    prompt_type: str          # 'unexpected' | 'at_dest' | 'observe'
    decision: str
    rationale: str
    cache_hit: bool
    fallback: bool
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Day-history rendering
# ---------------------------------------------------------------------------


# Human-readable phrasing for outcome tags in the day-history prompt.
_ACTOR_PHRASE = {"H": "human", "R": "robot"}
_OUTCOME_PHRASE = {
    "done_as_is": "did it as usual",
    "corrected": "corrected the robot's action",
    "corrected_repeated": "corrected the robot's action again",
    "corrected_despite_correct": "moved it elsewhere even though the robot was right",
    "accepted": "went along with the robot's choice",
    "skipped": "nothing to do, robot already handled it",
    "skipped_repeated": "nothing to do, robot already handled it",
}


def _format_event(ev: dict) -> str:
    """Render one robot/human event from the day history as a plain-English line."""
    t = ev.get("timestep") or ev.get("time") or 0
    actor = _ACTOR_PHRASE.get(ev.get("actor", "?"), ev.get("actor", "?"))
    obj = ev.get("object", "?")
    loc = ev.get("location", "?")
    outcome = ev.get("outcome", "")
    h = (t // 60) % 24
    m = t % 60
    line = f"  {h:02d}:{m:02d} {actor} moved {obj} -> {loc}"
    phrase = _OUTCOME_PHRASE.get(outcome)
    if phrase:
        line += f" ({phrase})"
    return line


def _format_day_history(history: list[dict], max_actions: int) -> str:
    if not history:
        return "  (no prior events today)"
    tail = history[-max_actions:]
    n_dropped = max(0, len(history) - len(tail))
    body = "\n".join(_format_event(e) for e in tail)
    if n_dropped:
        body = f"  ... ({n_dropped} earlier events omitted)\n" + body
    return body


def _format_examples(examples: list[dict], prompt_type: str) -> str:
    relevant = [e for e in examples if e.get("prompt_type") == prompt_type]
    if not relevant:
        return ""
    pieces = ["Examples of how this human acted in similar situations:"]
    for i, ex in enumerate(relevant, 1):
        pieces.append(f"\nExample {i}:")
        pieces.append(ex.get("situation", "").rstrip())
        pieces.append(
            f"Decision: {ex['decision']}. Rationale: {ex.get('rationale', '')}"
        )
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Prompt builders — return (system_msg, user_msg, schema)
# ---------------------------------------------------------------------------


def _format_upcoming_plan(upcoming_plan: list[dict], focus_object: str | None = None,
                          max_items: int = 12) -> str:
    """Render the human's planned future actions as clock-time lines.

    If *focus_object* is given, flag the lines that involve it so the model can
    tell at a glance whether it will still need that object soon.
    """
    if not upcoming_plan:
        return ("  (nothing else planned for the rest of the day — you are done "
                "with your scheduled tasks)")
    lines = []
    for p in upcoming_plan[:max_items]:
        t = int(p.get('time', 0))
        hh, mm = (t // 60) % 24, t % 60
        obj = p.get('object', '?')
        loc = p.get('location', '?')
        mark = "  <-- this object" if (focus_object and obj == focus_object) else ""
        lines.append(f"  {hh:02d}:{mm:02d} move {obj} -> {loc}{mark}")
    return "\n".join(lines)


_PLAN_GUIDANCE = (
    "Use your upcoming plan to judge whether you still need this object soon. "
    "If you have no remaining planned use for it, its current spot is usually "
    "fine even if it's not where you'd normally keep it. Only intervene if its "
    "placement actually gets in the way of something you still plan to do."
)


def build_unexpected_prompt(
    activity: str,
    planned_object: str,
    planned_destination: str,
    found_location: str,
    robot_recent_action: dict | None,
    day_history: list[dict],
    examples: list[dict],
    max_history_actions: int,
    upcoming_plan: list[dict] | None = None,
) -> tuple[str, str, dict]:
    """Prompt when script dest differs from object's current location. Returns (system, user, schema)."""
    hist = _format_day_history(day_history, max_history_actions)
    ex = _format_examples(examples, "unexpected")
    plan = _format_upcoming_plan(upcoming_plan or [], focus_object=planned_object)
    robot_note = ""
    if robot_recent_action:
        robot_note = (
            f"\nThe robot just moved {robot_recent_action.get('object', '?')} "
            f"to {robot_recent_action.get('location', '?')}.\n"
        )
    user = (
        f"Current activity: {activity}\n"
        f"You planned to move the {planned_object} to {planned_destination}.\n"
        f"But you find the {planned_object} is currently at {found_location}.\n"
        f"{robot_note}"
        f"Your upcoming planned actions:\n{plan}\n\n"
        f"Day history so far:\n{hist}\n\n"
        f"{_PLAN_GUIDANCE}\n\n"
        f"{ex}\n\n"
        "Decide: 'correct' (put it back where you planned), "
        "'accept' (adapt your activity to use it where the robot put it), "
        "or 'skip' (this step is no longer needed)."
    )
    return SYSTEM_PROMPT, user, _SCHEMA_UNEXPECTED


def build_at_dest_prompt(
    activity: str,
    planned_object: str,
    planned_destination: str,
    robot_recent_action: dict | None,
    day_history: list[dict],
    examples: list[dict],
    max_history_actions: int,
    upcoming_plan: list[dict] | None = None,
) -> tuple[str, str, dict]:
    """Prompt when object is already at the script destination. Returns (system, user, schema)."""
    hist = _format_day_history(day_history, max_history_actions)
    ex = _format_examples(examples, "at_dest")
    plan = _format_upcoming_plan(upcoming_plan or [], focus_object=planned_object)
    robot_note = ""
    if robot_recent_action:
        robot_note = (
            f"\nThe robot moved {robot_recent_action.get('object', '?')} "
            f"to {robot_recent_action.get('location', '?')} (matches your plan).\n"
        )
    user = (
        f"Current activity: {activity}\n"
        f"You planned to move the {planned_object} to {planned_destination}.\n"
        f"The {planned_object} is already at {planned_destination}.\n"
        f"{robot_note}"
        f"Your upcoming planned actions:\n{plan}\n\n"
        f"Day history so far:\n{hist}\n\n"
        f"{_PLAN_GUIDANCE}\n\n"
        f"{ex}\n\n"
        "Decide: 'skip' (nothing to do — the robot got there first) "
        "or 'correct' (move it somewhere else anyway)."
    )
    return SYSTEM_PROMPT, user, _SCHEMA_AT_DEST


def build_observe_prompt(
    activity: str,
    robot_object: str,
    robot_destination: str,
    upcoming_plan: list[dict],
    day_history: list[dict],
    examples: list[dict],
    max_history_actions: int,
    mode: str = "undo_allow",
    redirect_destination: str | None = None,
    plan_window_min: int | None = None,
) -> tuple[str, str, dict]:
    """Vigilance-replacement prompt after a robot action. Returns (system, user, schema)."""
    hist = _format_day_history(day_history, max_history_actions)
    ex = _format_examples(examples, "observe")
    plan = _format_upcoming_plan(upcoming_plan or [], focus_object=robot_object,
                                 max_items=40)
    window_txt = (f"next ~{plan_window_min / 60:.0f} hours"
                  if plan_window_min else "next ~2 hours")
    head = (
        f"Current activity: {activity}\n"
        f"You just noticed the robot moved the {robot_object} to {robot_destination}.\n"
        f"Your upcoming planned actions ({window_txt}; this list is COMPLETE "
        f"for that window):\n{plan}\n\n"
        f"Day history so far:\n{hist}\n\n"
    )
    if mode == "redirect":
        dest_txt = (f"to {redirect_destination}" if redirect_destination
                    else "to wherever you plan to use it next")
        guidance = (
            "You move objects to minimize your own effort. If you'll need this "
            "object soon at a known place, it's wasteful to undo the robot and "
            "then move it again later — instead move it straight there now, in "
            "one trip. Only undo if leaving the object where it is would be "
            "actively harmful (e.g. a perishable food left out of the fridge "
            "will spoil), or if you have no foreseeable use for it. "
            "IMPORTANT: the plan above lists EVERY move you will make in that "
            "window. If this object does not appear in it, you will not need "
            "it in that window — do not assume a need based on the current "
            "activity or your habits; judge only from the listed actions."
        )
        decide = (
            f"Decide: 'redirect' (move it now {dest_txt} — the place you'll need "
            "it next, saving a second trip), 'undo' (move it back where it "
            "belongs — only if leaving it is harmful, e.g. it will spoil), or "
            "'allow' (leave it where the robot put it)."
        )
        schema = _SCHEMA_OBSERVE_REDIRECT
    else:
        guidance = _PLAN_GUIDANCE
        decide = ("Decide: 'undo' (move the object back — the robot got it wrong) "
                  "or 'allow' (leave it where the robot put it).")
        schema = _SCHEMA_OBSERVE
    user = head + f"{guidance}\n\n{ex}\n\n" + decide
    return SYSTEM_PROMPT, user, schema


# ---------------------------------------------------------------------------
# LLM client with disk cache
# ---------------------------------------------------------------------------


@dataclass
class LLMHumanClient:
    """Wrapper around OpenAI's chat completion with structured outputs + cache.

    Set ``OPENAI_API_KEY`` in the environment before instantiating.
    """

    model: str = "gpt-4o-mini"
    cache_dir: str = "cache/llm_human/"
    provider: str = "openai"
    request_timeout: float = 30.0
    max_retries: int = 6           # exponential backoff: 1, 2, 4, 8, 16, 32s
    backoff_base: float = 1.0      # seconds for the first retry
    n_calls: int = field(default=0, init=False)
    n_cache_hits: int = field(default=0, init=False)
    n_retries: int = field(default=0, init=False)
    cost_usd: float = field(default=0.0, init=False)

    def __post_init__(self):
        if OpenAI is None:
            raise RuntimeError(
                "openai package not installed. `pip install openai>=1.30` first."
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your shell before "
                "launching the run."
            )
        self._client = OpenAI()
        if diskcache is None:
            raise RuntimeError(
                "diskcache package not installed. `pip install diskcache>=5` first."
            )
        os.makedirs(self.cache_dir, exist_ok=True)
        self._cache = diskcache.Cache(self.cache_dir)

    # ------------------------------------------------------------------
    def _cache_io(self, op, *args):
        """Run a diskcache op (get/set), retrying transient SQLite/OS disk-I/O
        errors with exponential backoff. These happen when many parallel jobs
        touch the same cache on a network filesystem (diskcache writes on read
        for LRU stats, so even reads contend). Re-raises on exhaustion — it
        NEVER swallows the error and returns None, because a swallowed *read*
        error would look like a cache miss and trigger an OpenAI call."""
        delay = 0.25
        for attempt in range(self.max_retries + 1):
            try:
                return op(*args)
            except (sqlite3.OperationalError, OSError):
                if attempt >= self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

    # ------------------------------------------------------------------
    def _cache_key(self, system: str, user: str, schema_name: str) -> str:
        """SHA-256 of provider/model/prompts (disk-cache key). Temperature is
        hardcoded 0.0 everywhere, so it's pinned in the key (keeps existing
        temperature-0 caches valid; runs that were cached at 0.7 miss and re-call)."""
        blob = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "temperature": 0.0,
                "schema": schema_name,
                "system": system,
                "user": user,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def call(
        self, system: str, user: str, schema: dict
    ) -> tuple[dict, bool, float]:
        """Return (parsed_response, cache_hit, elapsed_ms).

        On API failure, retry with exponential backoff up to ``max_retries``.
        If every retry fails, raise the final exception — there is no silent
        fallback (the user wants the run to surface the error instead).
        """
        key = self._cache_key(system, user, schema["name"])
        cached = self._cache_io(self._cache.get, key)
        if cached is not None:
            self.n_cache_hits += 1
            return cached, True, 0.0

        start = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,   # hardcoded: deterministic, reproducible benchmark
                    timeout=self.request_timeout,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": schema,
                    },
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                payload = json.loads(resp.choices[0].message.content)
                self.n_calls += 1
                # Approximate cost: prefer the API's token counts when
                # available; otherwise estimate from char counts.
                in_chars = len(system) + len(user)
                out_chars = len(resp.choices[0].message.content)
                usage = getattr(resp, 'usage', None)
                if usage and getattr(usage, 'prompt_tokens', None) is not None:
                    in_chars = usage.prompt_tokens * _CHARS_PER_TOKEN
                    out_chars = (usage.completion_tokens or 0) * _CHARS_PER_TOKEN
                self.cost_usd += estimate_cost_usd(self.model, in_chars, out_chars)
                self._cache_io(self._cache.set, key, payload)
                return payload, False, elapsed_ms
            except Exception as e:  # noqa: BLE001
                # Billing/quota errors won't resolve with backoff — fail loud.
                if _is_insufficient_funds(e):
                    raise LLMInsufficientFundsError(
                        "OpenAI request rejected for insufficient quota/credits. "
                        "Top up the account or lower usage, then resume "
                        "(cached decisions will be reused)."
                    ) from e
                last_exc = e
                if attempt >= self.max_retries:
                    break
                self.n_retries += 1
                wait = self.backoff_base * (2 ** attempt)
                print(
                    f"  [LLM] call failed (attempt {attempt + 1}/"
                    f"{self.max_retries + 1}: {type(e).__name__}: {e}); "
                    f"retrying in {wait:.1f}s"
                )
                time.sleep(wait)
        raise RuntimeError(
            f"LLM call failed after {self.max_retries + 1} attempts"
        ) from last_exc

    # ------------------------------------------------------------------
    def reset_counters(self):
        """Zero per-day call/cache/retry/cost counters."""
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_retries = 0
        self.cost_usd = 0.0

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return call, cache-hit, retry, and approximate USD totals."""
        return {
            "n_llm_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "n_retries": self.n_retries,
            "cost_usd": round(self.cost_usd, 4),
        }


# ---------------------------------------------------------------------------
# Few-shot examples loader
# ---------------------------------------------------------------------------


def load_examples(path: str | None) -> list[dict]:
    """Load few-shot examples JSON (``{"examples": [...]}`` or bare list)."""
    if not path or not os.path.isfile(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("examples", [])
    return data
