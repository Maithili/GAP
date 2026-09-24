# Model components

Train-time models for the hybrid particle-filter policy.

## `activity_transformer.py`

Small transformer (~250k params) that predicts the next activity label and
expected start time from LLM-curated day segments. Activity names are an open
vocabulary; tokens use frozen language embeddings (OpenAI `text-embedding-3-small`,
stored by the curator in `embeddings.json`) passed through a learned projection.

Two input modes share the same trunk (`config/base.yaml` → `robot.hybrid.transformer`):

- **event** (primary) — one token per tagged segment; targets = next activity + gap to its start.
- **interval** — one token per 10-min tick; target = activity at the next tick.

Trained by `train_activity_transformer()` inside the hybrid policy's `train()` path.
Used at eval time for prep/next-activity timing; online recognition is handled by
`agents/activity_particle_filter.py`, not this model.

## `vocab.py`

Vocabulary helpers for the activity transformer.

## `nopa_gpn.py`

Goal Proposal Network for the NOPA baseline (Puig et al., ICRA 2023): learns
`q(g | s_0, s_t)` over per-predicate goal counts, which the NOPA policy samples
its K goal proposals from. Trained on warm-start (human-only) days.
