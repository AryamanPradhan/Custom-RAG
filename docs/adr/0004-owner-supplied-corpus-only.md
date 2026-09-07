# Owner-supplied corpus only: no crawling

A Corpus contains exactly what a property owner handed over — uploaded PDFs,
DOCX, Markdown, HTML, and structured feeds injected as text. The site crawler
and the drift detector that re-fetched crawled pages are both gone.

Crawling a hotel site is the fastest way to fill an index and the slowest way to
trust it. A marketing site carries stale rates, expired offers, a policy page
last edited two owners ago, and boilerplate that a chunker cannot tell from
substance — and every one of those becomes a citable Source that the grounding
check *passes*, because the answer really is supported by the retrieved chunk.
The check can only tell whether a claim traces to the Corpus, never whether the
Corpus should have contained it. Requiring someone to hand over each document
puts a human decision in front of everything the Guide can say.

It is also the cheaper posture. A 300-page crawl is 300 classification calls and
thousands of embeddings for content that is mostly not answer material; a
folder of documents is a few dozen.

## Consequences

- Ingestion has one entrance: `ingest_upload` in `app/ingestion/pipeline.py`,
  reachable through `POST /admin/properties/{id}/upload` and `guide upload`.
  There is no URL to point the system at.
- Freshness is a service, not a signal. Nothing polls a live page, so nothing
  can report that a corpus has fallen behind — that is why every citation
  carries the date its Source was ingested, and why `source_state` still
  records one row per indexed Source.
- Owner-supplied is **not** the same as trusted. Owners forward files they did
  not write, so retrieved chunks stay fenced as data and injection screening in
  `app/guardrails/` applies unchanged.
- The corpus is now bounded by what a person bothered to send. A property with a
  rich website and no documents starts empty, and the Guide deflects rather than
  guesses — which is the intended failure.
