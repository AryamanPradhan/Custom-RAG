# Model providers behind one gateway

> **Amended 2026-09-05.** The Anthropic leg is retired: OpenAI serves every
> chat task (answer and rewrite on GPT-4o mini, verification on GPT-4o mini,
> eval on GPT-4o) and Google still serves embeddings. The reason is
> operational, not technical - one chat vendor, one billing relationship. The
> per-task routing argument below is unchanged and is what made the swap a
> config change: the Anthropic adapter and its prices stay in the gateway, so
> pointing a task at a `claude-*` id restores the original split.
>
> Reranking then left the chat providers entirely for **Cohere rerank-v3.5**, a
> dedicated relevance model - see the reranking section below.
>
> What changed in code is cost accounting, twice. Cached prompt tokens are now
> billed per vendor (OpenAI reads at 0.5x and never charges to write the cache,
> Anthropic reads at 0.1x and writes at 1.25x), because one shared multiplier
> under-reported OpenAI spend against the daily cap. And `ModelSpec` grew a
> per-search-unit price, because Cohere does not bill tokens at all and folding
> it into an invented token rate would have made the reranker's spend fiction.

## Original decision

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

## Reranking is not a chat task

Reranking was a listwise LLM call because there was already a chat vendor and
no reranker. That is a prompt, a JSON schema, ~40 truncated passages of input
tokens and several seconds, every turn, to produce a number.

A dedicated reranker does the same job as a cross-encoder over (query,
passage) pairs: no prompt, one call, about a tenth of the latency, and a price
that does not scale with how much context is fed to it. On this pipeline it
removes the single largest input-token cost in the request path.

The cost is a third vendor and a third key, immediately after consolidating on
one chat vendor - accepted because rerank is not the answer path. If Cohere is
down or unkeyed the Guide still answers: point `RERANK_MODEL` at a chat model
and the listwise path takes over, unchanged.

The subtler cost is that the two rerankers do not speak the same units. Cohere
returns 0-1 relevance, uncalibrated across queries; the listwise prompt returns
a 0-10 rubric score with defined bands. The deflect decision hangs on that
threshold, so they are kept as two settings - `MIN_RERANK_SCORE` and
`MIN_RERANK_RELEVANCE` - and only the one matching `RERANK_MODEL` is ever read.
Converting between them would make "nothing is relevant enough to answer" mean
something different without anyone changing a line.

## Consequences

- Every model call goes through `app/gateway/`, which is the only place that
  constructs a provider client. Cross-vendor cost is normalised there, which is
  what makes a per-property daily spend cap meaningful at all.
- Adding a model means adding it to `app/gateway/pricing.py`. An unknown model
  id raises rather than defaulting, because a silent fallback would make that
  model's spend invisible to the cap. That is also the guard on switching to a
  newer reranker: it fails loudly until its price is entered.
- `MIN_RERANK_RELEVANCE` is a starting value, not a validated one. Cohere's
  scores are not comparable across queries, so it has to be tuned against the
  eval set for a given corpus - too high deflects answerable questions, too low
  hands the answer model passages that do not answer anything.
- **The embedding choice is the expensive one to revisit.** Dimension is fixed
  when the Qdrant collection is created and vector spaces are not comparable
  across models, so changing the embedding model means re-embedding every
  property's corpus. The chat providers can be swapped in a line.
