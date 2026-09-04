# Stateless sessions: the widget carries the transcript

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
