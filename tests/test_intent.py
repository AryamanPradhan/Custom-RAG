"""Intent routing.

The routing decision is cheap to get right and expensive to get wrong in one
direction only: a question misread as smalltalk is never answered at all. So
most of what is tested here is the boundary - messages that look like
smalltalk and are not.
"""

from __future__ import annotations

import pytest

from app.pipeline.intent import Intent, classify_intent
from app.pipeline.prompts import build_smalltalk


class TestSmalltalk:
    @pytest.mark.parametrize(
        "message",
        [
            "hi",
            "Hi!",
            "hello",
            "hey there",
            "hi how are you?",
            "Hello, how are you doing today?",
            "hii",
            "namaste",
            "good morning",
            "how are you",
            "hey, hope you are well",
        ],
    )
    def test_greetings(self, message: str) -> None:
        assert classify_intent(message) is Intent.GREETING

    @pytest.mark.parametrize(
        "message",
        ["thanks", "thank you", "thanks a lot!", "ty", "perfect, thanks", "ok great"],
    )
    def test_thanks(self, message: str) -> None:
        assert classify_intent(message) is Intent.THANKS

    @pytest.mark.parametrize(
        "message", ["bye", "goodbye", "see you", "good night", "thanks, bye!"]
    )
    def test_farewell(self, message: str) -> None:
        assert classify_intent(message) is Intent.FAREWELL

    @pytest.mark.parametrize(
        "message",
        ["who are you", "what can you do", "are you a bot?", "help", "how do you work"],
    )
    def test_capability(self, message: str) -> None:
        assert classify_intent(message) is Intent.CAPABILITY


class TestQuestionsAreNeverSmalltalk:
    """The failure that actually costs a visitor something."""

    @pytest.mark.parametrize(
        "message",
        [
            # A greeting wrapped around a real question is a real question.
            "hi, what time is check-in?",
            "hello! do you allow dogs",
            "hey are you open on sundays",
            "good morning, is breakfast included?",
            "thanks - and what about parking?",
            # Built from ordinary words, but asking something.
            "are you well located for MG Marg",
            "is the pool good",
            "how are the rooms",
            "what are your rates",
            # Bare "help" is a capability question; help with something is not.
            "help me find a room for four people",
            # Long enough that it cannot be smalltalk whatever words it uses.
            "ok so i am not sure how all of this works and what you can do for me "
            "and my family",
        ],
    )
    def test_routed_to_retrieval(self, message: str) -> None:
        assert classify_intent(message) is Intent.INFORMATIONAL

    @pytest.mark.parametrize("message", ["", "   ", "\n"])
    def test_empty_is_not_smalltalk(self, message: str) -> None:
        # The input guard rejects these; intent must not claim them first.
        assert classify_intent(message) is Intent.INFORMATIONAL


class TestReplies:
    @pytest.mark.parametrize(
        "intent", ["greeting", "thanks", "farewell", "capability"]
    )
    def test_every_intent_has_a_reply_naming_the_property(self, intent: str) -> None:
        reply = build_smalltalk(intent, "The Glass Hotel")
        assert "The Glass Hotel" in reply
        assert "{" not in reply

    def test_greeting_invites_a_question(self) -> None:
        reply = build_smalltalk("greeting", "The Glass Hotel")
        # The turn has to be handed back, or the visitor is left holding a
        # pleasantry instead of an answer.
        assert reply.rstrip().endswith("?")
