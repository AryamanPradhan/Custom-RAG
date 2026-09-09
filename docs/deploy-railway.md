# Deploying to Railway

The API is a long-running HTTP server that streams. It is not a serverless
function: `/chat/stream` holds a connection open for the length of an answer,
and the SQLite database has to survive a restart.

## 1. Create the service

Point Railway at this repository. `railway.json` selects the Dockerfile
builder and sets `/health` as the health check, so no build configuration is
needed in the dashboard.

## 2. Mount a volume at `/app/data` — before the first deploy

This is the step that cannot be skipped. `data/guide.db` holds nine tables,
including `properties`, `property_origins` and `spend_ledger`. On Railway's
ephemeral filesystem every restart and every redeploy wipes it, which means:

- every Property registration gone
- every origin allowlist gone, so every client's widget starts returning 403
- the day's spend ledger reset to zero

The Corpus itself is safe — it lives in Qdrant Cloud — but the metadata that
makes it reachable does not. Mount the volume, then set `DATABASE_PATH` to a
path inside it.

## 3. Environment variables

| variable | value | why |
| --- | --- | --- |
| `DATABASE_PATH` | `/app/data/guide.db` | on the mounted volume, not the container filesystem |
| `SESSION_SECRET` | `python -c "import secrets; print(secrets.token_urlsafe(32))"` | unset means a per-process key, so every restart cuts every conversation in flight |
| `ADMIN_API_KEY` | a generated key | guards ingestion, corpus deletion and the chat log |
| `DEV_PAGES_ENABLED` | `false` | `/demo` and `/console` are unauthenticated, and `/console` prompts for the admin key |
| `TRUSTED_PROXY_HOPS` | `1` | Railway terminates TLS in front of the app, so the socket peer is the proxy |
| `RUN_SPEND_CAP_USD` | `0` (unlimited) or a real ceiling | **see below** |
| `ACCOUNT_DAILY_SPEND_CAP_USD` | above the sum of every per-property cap | it is a shared fuse |
| `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY` | | |
| `QDRANT_URL`, `QDRANT_API_KEY` | the Qdrant Cloud cluster | |
| `LOGFIRE_TOKEN` | optional | |

### The two caps that will take the service down

`RUN_SPEND_CAP_USD` is scoped to one **process** and only resets when the
process restarts. At the default `0.15`, and roughly $0.003 an answer, the
server begins refusing every request after about fifty answers and stays that
way until someone redeploys. That is a sensible guard for `guide eval` on a
laptop and a time bomb on a server nobody is watching.

`ACCOUNT_DAILY_SPEND_CAP_USD` is shared by every Property. When it trips,
`resolve_property` returns 429 to all of them, so one busy client takes every
other client's widget down. It defaults to `1.00`, which is a fifth of a
single client's own `5.00` cap — meaning the per-property cap can never
actually be reached.

Set the account cap above the sum of the per-property caps, and let the
per-property cap be the real control. Then a runaway client is limited to
their own budget instead of everyone else's availability.

## 4. After the first deploy

Railway gives the service a public domain. With that URL:

```
guide origins glass-hotel --add https://the-client-site.com
```

and on the client's site:

```html
<script src="https://your-service.up.railway.app/guide.js"
        data-property-id="glass-hotel"
        data-endpoint="https://your-service.up.railway.app"></script>
```

Check `/health` returns 200 and that a question answers end to end before
pointing a client's domain at it.

## Notes

- **One replica.** The rate limiter and the usage limiter are both
  per-process, so a second replica silently doubles both ceilings. The
  Dockerfile runs a single uvicorn worker for the same reason.
- **Ingestion runs inline** on `/admin/.../upload`, so a large document set
  ties up the worker while it indexes. Fine for hand-operated onboarding;
  worth revisiting if it is ever automated.
- **The image excludes `.env`.** Railway supplies the environment; a key baked
  into a layer is a key in every copy of that image.
