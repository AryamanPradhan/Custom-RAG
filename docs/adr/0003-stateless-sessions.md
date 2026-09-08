# Stateless sessions: the widget carries the transcript

> **Amended 2026-09-08.** Turns are now recorded, in `chat_log`. Sessions stay
> stateless in the sense that decides the architecture: nothing is read back
> out of that table, no request depends on a previous one having been stored,
> and the widget still carries the history a model sees. What changed is that
> the record is written *after* the answer is served, for the operator rather
> than for the pipeline - a deflection nobody can see is a gap in the Corpus
> nobody can fix, and "what did the Guide tell my guest?" was unanswerable.
>
> The claim below that gets retracted is the last one: there *are* now visitor
> transcripts to expire and hand over. So the question is stored as the
> guardrails left it (PII already redacted, since the log holds exactly the
> text the provider saw), `CHAT_LOG_RETENTION_DAYS` bounds how long a row
> lives, `CHAT_LOG_ENABLED=false` turns the whole thing off for a client who
> wants nothing kept, and `DELETE /admin/properties/{id}/chats` erases on
> request. What has not changed: no visitor identifier is stored beyond the
> `session_id` the widget itself generates, and no IP address is.

## Original decision

There is no conversation store. The widget holds the history and sends it back
with each turn, and the backend keeps nothing about a visitor between requests.

Conversations here are short, anonymous and low-value to retain, and this
removes an entire piece of infrastructure (Redis or a conversations table) from
a product whose only other persistent state is six config fields per property.
It also sidesteps retention entirely: there are no visitor transcripts to
expire, export or hand over.

## Consequences

The transcript is client-controlled, so it is untrusted input. A forged
`assistant` turn is the cheapest way to plant an instruction a model will treat
as its own prior output, so `check_history` screens both roles for injection
before anything reaches a provider - see `app/guardrails/input_guard.py`.

The per-IP rate limiter is in-process for the same reason (no shared store), so
with N containers the effective limit is N times the configured rate. The daily
spend cap, which is in SQLite and therefore shared, is the real backstop.
