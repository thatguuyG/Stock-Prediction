# ADR 0003 — VADER for Phase 1 sentiment

**Status:** Accepted (2026-05-15)

## Context

Phase 1 scores headline sentiment to feed the eventual model. Options range from lexicon-based scorers to transformer models fine-tuned on financial text:

- VADER — lexicon + rules, no GPU, ~5ms per headline.
- FinBERT — BERT fine-tuned on financial sentiment; needs the model weights loaded; orders of magnitude slower without a GPU.
- OpenAI / Anthropic API — high quality but per-call cost and rate limits.

We don't yet have a model that consumes these scores, so we don't yet know how much quality matters versus throughput.

## Decision

Use **VADER 3.3.2** in Phase 1. Tag every score with `model="vader-1.0"` so future scorers can coexist without overwriting.

## Consequences

**Positive**
- Zero infra cost: no GPU, no API key, no model weights to download.
- Fast enough to score the entire backlog on every CI run.
- Deterministic — same input always produces the same compound score, so tests are stable.

**Negative**
- VADER's lexicon is general-purpose. It misses financial vocabulary: "soars" is neutral, "beat estimates" is neutral, "guidance lowered" can register positive on the word "guidance." Phase 1 tests deliberately use lexicon-friendly vocabulary to avoid masking this.
- We will likely outgrow VADER in Phase 2 if it provides no signal to the model. The schema is ready for that — the `sentiments` table keys on `(news_item_id, model)` indirectly via uniqueness on `news_item_id` and a `model` tag, so swapping in FinBERT is additive.

## Alternatives considered

- **FinBERT now**: better quality but the engineering cost (model artifact handling, batching, GPU vs CPU inference) is misplaced before we know sentiment matters. Rejected for Phase 1.
- **LLM-based scoring**: most flexible but per-headline cost and rate-limit concerns. Worth piloting later for nuanced classification (event type, not just polarity), not for bulk scoring.

## Revisit if

- Phase 2 model evaluation shows sentiment features are uninformative — try FinBERT before concluding sentiment doesn't help.
- We start ingesting volumes that exceed NewsAPI free tier and need to be choosier about what we score.
