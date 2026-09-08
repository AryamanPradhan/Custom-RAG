"""Layer 06 - routing, cost accounting and the ceiling under every call.

No network: the provider adapters are replaced with fakes, and what is asserted
is everything the gateway does *around* a provider call - which model served
it, at what temperature, what it cost, who was billed, and whether the call was
allowed to happen at all. That last one is the point of the layer. Every path
into a vendor goes through this class precisely so the spend cap has one place
to stand, and a call that reaches a provider without passing `check` is money
nothing can see.

`tests/test_storage.py` covers the ledger arithmetic and the boot-time
preflight; this covers the calls that spend against them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.config import Settings
from app.gateway.budget import UsageExceeded, UsageLimiter, set_usage_limiter
from app.gateway.llm_gateway import LLMGateway, Task
from app.gateway.pricing import (
    Provider,
    UnknownModelError,
    cost_usd,
    provider_for,
    spec_for,
)
from app.gateway.providers import (
    CohereReranker,
    LLMResult,
    ProviderRegistry,
    RerankOutcome,
)


@pytest.fixture(autouse=True)
def _own_limiter():
    """The usage limiter is a process-wide singleton, so a test that did not
    install its own would spend against whatever .env configures - and trip a
    cap belonging to another test."""
    set_usage_limiter(UsageLimiter(spend_cap_usd=0, call_cap=0))
    yield
    set_usage_limiter(None)


def _settings(**overrides) -> Settings:
    base = {
        "openai_api_key": "sk-test",
        "cohere_api_key": "co-test",
        "answer_model": "gpt-4o-mini",
        "rerank_model": "rerank-v3.5",
    }
    return Settings(**{**base, **overrides})


class FakeProvider:
    """Stands in for an SDK adapter. Records what the gateway asked it for."""

    def __init__(self, *, result: LLMResult | None = None, fail: Exception | None = None):
        self._result = result or LLMResult(text="ok", model="gpt-4o-mini")
        self._fail = fail
        self.calls: list[dict] = []
        self.stream_chunks = ["Check-in ", "is from 2pm."]
        self.report_usage = True

    async def complete(self, **kwargs) -> LLMResult:
        self.calls.append(kwargs)
        if self._fail:
            raise self._fail
        return self._result

    async def stream(self, *, out: LLMResult, **kwargs) -> AsyncIterator[str]:
        self.calls.append({**kwargs, "stream": True})
        for piece in self.stream_chunks:
            yield piece
        # Providers report usage in a final chunk. An abandoned stream never
        # reaches this line, which is the case the estimate exists for.
        if self.report_usage:
            out.input_tokens = 1_000_000
            out.output_tokens = 500_000
            out.stop_reason = "stop"


class FakeReranker:
    def __init__(self, *, search_units: int = 1) -> None:
        self.calls: list[dict] = []
        self._units = search_units

    async def rerank(self, *, model, query, documents, top_n) -> RerankOutcome:
        self.calls.append(
            {"model": model, "query": query, "documents": documents, "top_n": top_n}
        )
        scores = [(i, 1.0 - i / 10) for i in range(min(top_n, len(documents)))]
        return RerankOutcome(
            scores=scores,
            usage=LLMResult(text="", model=model, search_units=self._units),
        )


def _wire(gateway: LLMGateway, provider=None, reranker=None) -> LLMGateway:
    """Install fakes in the registry's cache, which is where the lazily built
    real clients would otherwise land."""
    if provider is not None:
        gateway._registry._cache[Provider.OPENAI] = provider
    if reranker is not None:
        gateway._registry._reranker = reranker
    return gateway


class TestPricing:
    def test_cache_discounts_are_priced_per_vendor(self) -> None:
        """One shared cache constant would under-report OpenAI spend fivefold
        against the cap: OpenAI reads cached tokens at half price and never
        bills the write, Anthropic reads at a tenth and charges 1.25x to write."""
        assert cost_usd("gpt-4o-mini", cache_read_tokens=1_000_000) == pytest.approx(0.075)
        assert cost_usd("gpt-4o-mini", cache_write_tokens=1_000_000) == 0.0

        assert cost_usd("claude-haiku-4-5", cache_read_tokens=1_000_000) == pytest.approx(0.10)
        assert cost_usd("claude-haiku-4-5", cache_write_tokens=1_000_000) == pytest.approx(1.25)

    def test_input_and_output_are_priced_apart(self) -> None:
        assert cost_usd("gpt-4o-mini", input_tokens=1_000_000) == pytest.approx(0.15)
        assert cost_usd("gpt-4o-mini", output_tokens=1_000_000) == pytest.approx(0.60)

    def test_reranking_is_billed_per_search_unit_not_per_token(self) -> None:
        """A rerank of 40 passages costs what a rerank of 4 does, so folding it
        into an invented token rate would price it wrong in both directions."""
        assert cost_usd("rerank-v3.5", search_units=3) == pytest.approx(0.006)
        assert cost_usd(
            "rerank-v3.5", input_tokens=50_000, search_units=1
        ) == pytest.approx(0.002)

    def test_an_unpriced_model_is_loud(self) -> None:
        """A silent fallback here is spend the daily cap cannot see."""
        with pytest.raises(UnknownModelError, match="pricing catalogue"):
            spec_for("gpt-9-imaginary")
        with pytest.raises(UnknownModelError):
            cost_usd("gpt-9-imaginary", input_tokens=10)

    def test_every_catalogued_model_resolves_to_a_provider(self) -> None:
        assert provider_for("gpt-4o-mini") is Provider.OPENAI
        assert provider_for("rerank-v3.5") is Provider.COHERE
        assert provider_for("gemini-embedding-2") is Provider.GOOGLE
        assert provider_for("claude-opus-5") is Provider.ANTHROPIC


class TestResultReading:
    @pytest.mark.parametrize("reason", ["max_tokens", "length"])
    def test_a_cut_off_answer_is_visible_to_the_audit(self, reason: str) -> None:
        """Two vendors, two words for the same event - and an answer that
        stopped mid-sentence is one an operator needs to see in the log."""
        assert LLMResult(text="...", model="gpt-4o-mini", stop_reason=reason).truncated

    def test_a_finished_answer_is_not_truncated(self) -> None:
        assert not LLMResult(text="done", model="gpt-4o-mini", stop_reason="stop").truncated


class TestRouting:
    def test_each_task_goes_to_its_configured_model(self) -> None:
        """Routing is by task, not by caller: moving the answer path to another
        model is one env change, not a grep across the repo."""
        gateway = LLMGateway(
            _settings(
                answer_model="gpt-4o-mini",
                rewrite_model="gpt-4o-mini",
                verifier_model="gpt-4o",
                eval_model="gpt-4o",
            )
        )
        assert gateway.model_for(Task.ANSWER) == "gpt-4o-mini"
        assert gateway.model_for(Task.VERIFY) == "gpt-4o"
        assert gateway.model_for(Task.EVAL) == "gpt-4o"
        assert gateway.model_for(Task.RERANK) == "rerank-v3.5"

    def test_only_the_answer_is_sampled(self) -> None:
        """The verifier blocks answers. Sampling it means blocking a different
        set of correct answers on every run."""
        gateway = LLMGateway(_settings(answer_temperature=0.2))
        assert gateway.temperature_for(Task.ANSWER) == 0.2
        for task in (Task.REWRITE, Task.RERANK, Task.VERIFY, Task.EVAL):
            assert gateway.temperature_for(task) == 0.0

    def test_the_rerank_path_is_chosen_by_the_model_id(self) -> None:
        """The two rerankers differ in more than vendor - one returns a 0-10
        rubric score, the other 0-1 relevance - so the stage above has to know
        which it is talking to before it reads a threshold."""
        assert LLMGateway(_settings(rerank_model="rerank-v3.5")).uses_dedicated_reranker
        assert not LLMGateway(_settings(rerank_model="gpt-4o-mini")).uses_dedicated_reranker


class TestProviderRegistry:
    def test_one_client_per_vendor(self) -> None:
        """Lazy and cached: a deployment routed entirely at OpenAI never builds
        the Anthropic client, so its SDK never has to be installed."""
        registry = ProviderRegistry(openai_api_key="sk-test")
        first = registry.for_model("gpt-4o-mini")
        assert registry.for_model("gpt-4o") is first

    def test_a_reranker_cannot_be_asked_to_answer(self) -> None:
        registry = ProviderRegistry(openai_api_key="sk-test", cohere_api_key="co-test")
        with pytest.raises(ValueError, match="no chat adapter"):
            registry.for_model("rerank-v3.5")

    def test_an_embedding_model_cannot_be_asked_to_answer(self) -> None:
        registry = ProviderRegistry(openai_api_key="sk-test")
        with pytest.raises(ValueError, match="no chat adapter"):
            registry.for_model("gemini-embedding-2")

    def test_a_chat_model_is_not_a_dedicated_reranker(self) -> None:
        registry = ProviderRegistry(openai_api_key="sk-test", cohere_api_key="co-test")
        with pytest.raises(ValueError, match="not a dedicated reranker"):
            registry.reranker_for("gpt-4o-mini")

    def test_a_missing_cohere_key_says_what_to_do_about_it(self) -> None:
        with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
            CohereReranker(api_key=None)


class TestCompletion:
    async def test_the_call_is_billed_to_the_property(self) -> None:
        billed: list[tuple[str, float]] = []

        async def on_spend(property_id: str, amount: float) -> None:
            billed.append((property_id, amount))

        gateway = LLMGateway(_settings(), on_spend=on_spend)
        _wire(
            gateway,
            provider=FakeProvider(
                result=LLMResult(
                    text="Check-in is from 2pm.",
                    model="gpt-4o-mini",
                    input_tokens=1_000_000,
                )
            ),
        )

        result = await gateway.complete(
            Task.ANSWER,
            messages=[{"role": "user", "content": "hi"}],
            property_id="casa-verde",
        )

        assert result.text == "Check-in is from 2pm."
        assert billed == [("casa-verde", pytest.approx(0.15))]

    async def test_a_call_that_bills_nobody_still_hits_the_account_ledger(self) -> None:
        """Ingestion and evals name no Property. They are exactly the calls
        that run away unattended, so the account counters must still move."""
        billed: list[tuple[str, float]] = []

        async def on_spend(property_id: str, amount: float) -> None:
            billed.append((property_id, amount))

        limiter = UsageLimiter(spend_cap_usd=0, call_cap=0)
        set_usage_limiter(limiter)

        gateway = LLMGateway(_settings(), on_spend=on_spend)
        _wire(
            gateway,
            provider=FakeProvider(
                result=LLMResult(text="{}", model="gpt-4o-mini", input_tokens=1_000_000)
            ),
        )
        await gateway.complete(Task.EVAL, messages=[{"role": "user", "content": "hi"}])

        assert billed == []
        assert limiter.snapshot()["spent_usd"] == pytest.approx(0.15)
        assert limiter.snapshot()["calls"] == 1

    async def test_the_task_temperature_reaches_the_provider(self) -> None:
        gateway = LLMGateway(_settings(answer_temperature=0.2))
        provider = FakeProvider()
        _wire(gateway, provider=provider)

        await gateway.complete(Task.ANSWER, messages=[{"role": "user", "content": "hi"}])
        await gateway.complete(Task.VERIFY, messages=[{"role": "user", "content": "hi"}])

        assert [call["temperature"] for call in provider.calls] == [0.2, 0.0]

    async def test_an_explicit_model_overrides_the_route(self) -> None:
        """The eval harness scores one model's output with another's."""
        gateway = LLMGateway(_settings(answer_model="gpt-4o-mini"))
        provider = FakeProvider()
        _wire(gateway, provider=provider)

        await gateway.complete(
            Task.ANSWER, messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
        )
        assert provider.calls[0]["model"] == "gpt-4o"

    async def test_the_cap_is_checked_before_the_provider_is_reached(self) -> None:
        """A cap enforced after the call is a report, not a cap."""
        limiter = UsageLimiter(spend_cap_usd=0.10, call_cap=0)
        await limiter.record(0.20)
        set_usage_limiter(limiter)

        gateway = LLMGateway(_settings())
        provider = FakeProvider()
        _wire(gateway, provider=provider)

        with pytest.raises(UsageExceeded):
            await gateway.complete(Task.ANSWER, messages=[{"role": "user", "content": "hi"}])
        assert provider.calls == []

    async def test_a_provider_failure_bills_nothing_and_propagates(self) -> None:
        """A call that never returned is not one anyone should pay for, and the
        pipeline above has its own answer for an outage."""
        billed: list[tuple[str, float]] = []

        async def on_spend(property_id: str, amount: float) -> None:
            billed.append((property_id, amount))

        limiter = UsageLimiter(spend_cap_usd=0, call_cap=0)
        set_usage_limiter(limiter)

        gateway = LLMGateway(_settings(), on_spend=on_spend)
        _wire(gateway, provider=FakeProvider(fail=RuntimeError("upstream 503")))

        with pytest.raises(RuntimeError, match="upstream 503"):
            await gateway.complete(
                Task.ANSWER, messages=[{"role": "user", "content": "hi"}], property_id="a"
            )
        assert billed == []
        assert limiter.snapshot()["calls"] == 0


class TestRerankCall:
    async def test_scores_come_back_indexed_to_the_documents_sent(self) -> None:
        gateway = LLMGateway(_settings())
        reranker = FakeReranker()
        _wire(gateway, reranker=reranker)

        scores = await gateway.rerank(
            question="pet policy",
            documents=["a", "b", "c"],
            top_n=2,
            property_id="casa-verde",
        )

        assert scores == [(0, 1.0), (1, 0.9)]
        assert reranker.calls[0]["query"] == "pet policy"

    async def test_the_reranker_is_billed_like_every_other_call(self) -> None:
        """A stage that skipped the ledger would be spend the daily cap cannot
        see - and it runs on every single question."""
        billed: list[tuple[str, float]] = []

        async def on_spend(property_id: str, amount: float) -> None:
            billed.append((property_id, amount))

        gateway = LLMGateway(_settings(), on_spend=on_spend)
        _wire(gateway, reranker=FakeReranker(search_units=2))

        await gateway.rerank(question="q", documents=["a"], top_n=1, property_id="casa-verde")
        assert billed == [("casa-verde", pytest.approx(0.004))]

    async def test_the_cap_stops_a_rerank_too(self) -> None:
        limiter = UsageLimiter(spend_cap_usd=0, call_cap=1)
        await limiter.record(0.0)
        set_usage_limiter(limiter)

        gateway = LLMGateway(_settings())
        reranker = FakeReranker()
        _wire(gateway, reranker=reranker)

        with pytest.raises(UsageExceeded):
            await gateway.rerank(question="q", documents=["a"], top_n=1)
        assert reranker.calls == []


class TestStreaming:
    async def test_a_full_stream_bills_what_the_provider_reported(self) -> None:
        limiter = UsageLimiter(spend_cap_usd=0, call_cap=0)
        set_usage_limiter(limiter)

        gateway = LLMGateway(_settings())
        _wire(gateway, provider=FakeProvider())

        text = "".join(
            [
                piece
                async for piece in gateway.stream_answer(
                    messages=[{"role": "user", "content": "when is check-in?"}]
                )
            ]
        )

        assert text == "Check-in is from 2pm."
        # 1M input + 0.5M output on gpt-4o-mini.
        assert limiter.snapshot()["spent_usd"] == pytest.approx(0.15 + 0.30)

    async def test_an_abandoned_stream_is_still_paid_for(self) -> None:
        """A Visitor closing the tab stops the iteration - but the tokens were
        spent. Auditing after the loop rather than in a finally would let
        anyone stream answers for free by disconnecting, past the daily cap."""
        limiter = UsageLimiter(spend_cap_usd=0, call_cap=0)
        set_usage_limiter(limiter)

        gateway = LLMGateway(_settings())
        provider = FakeProvider()
        provider.stream_chunks = ["one ", "two ", "three"]
        _wire(gateway, provider=provider)

        stream = gateway.stream_answer(
            messages=[{"role": "user", "content": "x" * 400}], property_id="casa-verde"
        )
        async for _ in stream:
            break
        await stream.aclose()  # type: ignore[attr-defined]

        assert limiter.snapshot()["calls"] == 1
        assert limiter.snapshot()["spent_usd"] > 0

    async def test_usage_is_estimated_when_the_final_chunk_never_arrives(self) -> None:
        """Billing zero for an interrupted stream is the same hole by another
        route, so the ~4-chars-per-token estimate stands in."""
        limiter = UsageLimiter(spend_cap_usd=0, call_cap=0)
        set_usage_limiter(limiter)

        gateway = LLMGateway(_settings())
        provider = FakeProvider()
        provider.report_usage = False
        _wire(gateway, provider=provider)

        async for _ in gateway.stream_answer(
            messages=[{"role": "user", "content": "x" * 4000}], system="y" * 4000
        ):
            pass

        # 8000 prompt characters -> ~2000 estimated input tokens at $0.15/Mtok,
        # plus the handful of output tokens the fake yields.
        assert limiter.snapshot()["spent_usd"] == pytest.approx(0.0003, abs=2e-5)
