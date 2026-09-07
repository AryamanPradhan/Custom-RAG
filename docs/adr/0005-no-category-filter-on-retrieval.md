# Retrieval searches the whole Corpus: no category filter

Hybrid search runs once, over every Chunk a Property has, with the Visitor's own
wording alongside the planner's rewrites. The planner no longer guesses content
categories, the vector store no longer filters on them, and the two retries that
existed to undo a bad guess are gone.

The filter was a guess made in the one place where guessing is least informed:
before anything has been retrieved. `plan_query` labelled the *question*,
ingestion labelled the *document*, and those two taxonomies do not agree —
because a Source is one kind of thing while the questions it answers are many.
"Is there parking?" is an `amenities` question; the parking paragraph lives in a
`location` document about getting to the property. Both labels are correct, and
their disagreement removed the only Chunk that held the answer.

Retrieval then returned something anyway — wrong-category Chunks, scored and
ranked — so the retry that fired on an empty result never fired, and the Guide
deflected on a Corpus that had the answer. Two eval cases failed this way,
`parking` and `breakfast-included`, and a third, `phrasing-mismatch`, passed only
because its wrong-category candidates happened to score *below* the rerank floor
and trip the fallback. Passing by luck of scoring and failing by luck of scoring
are the same defect.

There is nothing to trade against it. A Corpus is a few dozen to a few hundred
Chunks; `top_k` already bounds what reaches the reranker, and reranking is billed
per search unit of 100 documents, so the filter bought no latency and no money.
Deciding relevance is the reranker's job, and it does it having read the
candidates.

## Consequences

- `QueryPlan` carries `queries` and `unit_dependent` only. The planner's JSON
  schema lost `categories`, which makes the call marginally cheaper and removes
  a field the model could get wrong.
- `VectorStore._filter` enforces tenant isolation and nothing else.
- `Retriever.retrieve` is one pass. The Visitor's raw phrasing is now a query on
  every question rather than a consolation prize on the ones that already
  failed — a rewrite is a paraphrase, and a paraphrase can lose the one word the
  Source uses.
- An empty result after reranking now means what it says: the search saw the
  whole Corpus and nothing in it was relevant. A Deflection from that point is a
  real gap, so `retrieval.retry` and the `retrieve_retry` / `retrieve_widened`
  spans no longer exist.
- `DocCategory` survives on the Chunk payload. It is still worth knowing what a
  Source is for — it just does not get a veto over what a Visitor can be told.
- Recall rises and precision falls at the retrieval stage by design. The rerank
  relevance floor is now the only thing standing between a weak candidate and
  the answer model, which makes it the next thing to calibrate.
