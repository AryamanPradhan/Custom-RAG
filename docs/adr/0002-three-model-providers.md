# Three model providers behind one gateway

The Guide routes work across OpenAI (answers), Anthropic (groundedness
verification, query rewriting, reranking, eval) and Google (embeddings). Three
vendors and three API keys for one small product looks like accidental sprawl,
so: it was chosen deliberately, per task.

The answer path is the high-volume, visitor-facing one and runs on GPT-4o mini
at roughly $0.15/$0.60 per million tokens. That is a small model, and the answer
path is exactly where the product's liability sits - it must never state a
property fact that is not in a retrieved chunk. So the groundedness verifier
stays on Claude Haiku and *blocks*: an answer that fails it is replaced by a
Deflection rather than annotated. Cheap model where the volume is, competent
model where the liability is. Gemini serves embeddings because Anthropic has no
embedding endpoint, and pairing it with the answer vendor was not a requirement.

## Consequences

- Every model call goes through `app/gateway/`, which is the only place that
  constructs a provider client. Cross-vendor cost is normalised there, which is
  what makes a per-property daily spend cap meaningful at all.
- Adding a model means adding it to `app/gateway/pricing.py`. An unknown model
  id raises rather than defaulting, because a silent fallback would make that
  model's spend invisible to the cap.
- **The embedding choice is the expensive one to revisit.** Dimension is fixed
  when the Qdrant collection is created and vector spaces are not comparable
  across models, so changing the embedding model means re-embedding every
  property's corpus. The chat providers can be swapped in a line.
