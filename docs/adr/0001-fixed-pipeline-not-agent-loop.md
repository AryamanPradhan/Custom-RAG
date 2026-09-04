# A fixed pipeline, not an agent loop

The reference architecture this project was designed against is an *agentic* AI
stack with a planner and sub-agents, so a reader will reasonably expect an agent
loop and find a straight-line pipeline instead.

We deliberately did not build one. The Guide is read-only: it has no tools to
call, no transactions to perform and no actions to take, so there is nothing for
a planner to orchestrate. The work is always the same five steps - screen, plan
the query, retrieve, rerank, answer, verify - and a loop over that sequence buys
nothing while costing flat latency, straightforward evaluation and a single
traceable path per request.

## Considered options

- **Free-running agent loop.** Would buy multi-hop questions and self-correction
  on empty retrieval. We get the useful part of both from the query planner
  (which decomposes multi-part questions into several searches) and one
  conditional retrieval retry, without unbounded turns on a public endpoint
  where each turn costs money.
- **Fixed pipeline (chosen).** Flat latency, which matters on a website widget
  where visitors abandon; each stage independently scoreable by the eval
  harness, so "the answer was wrong" resolves to a retrieval, rerank or
  generation problem.

## Consequences

The tool-call seam is preserved in the gateway. If live availability or rates
are ever added - the one thing RAG must never answer from a cached crawl - that
becomes a tool call at a known insertion point rather than a rewrite.
