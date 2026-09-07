"""Which kind of turn is this?

Not every message to a property's Guide is a question about the property. A
visitor opens the widget and says "hi"; they say "thanks" when they have what
they came for. Sent down the retrieval path, those cost a rewrite call, an
embedding, a vector search and a rerank - and end in a Deflection telling
someone who said hello to phone the hotel. It is the worst first impression
the system can make, and it is also the most expensive way to make it.

So intent is decided here, before anything is retrieved and before any model
is called. Two rules keep it safe:

  1. The whole message must be smalltalk. Matching is against the entire
     normalised message, never a substring, so "hi, what time is check-in?"
     is a question that happens to open with a greeting - and goes to
     retrieval like any other.
  2. Anything unrecognised is INFORMATIONAL. A greeting misread as a question
     costs a visitor one slightly formal answer; a question misread as a
     greeting costs them the answer entirely.

No model call, so a greeting is free and instant rather than $0.002 and six
seconds.
"""

from __future__ import annotations

import re
from enum import StrEnum


class Intent(StrEnum):
    INFORMATIONAL = "informational"  # a question about the property -> retrieval
    GREETING = "greeting"
    THANKS = "thanks"
    FAREWELL = "farewell"
    CAPABILITY = "capability"  # "who are you", "what can you do"


# Longer than this and it is not smalltalk, whatever words it uses. A safety
# net against a long message built entirely from common words.
_MAX_SMALLTALK_TOKENS = 8

# Asked about the Guide itself rather than the property. Matched whole, so
# "help me find the spa" is a question and only a bare "help" is not.
_CAPABILITY = re.compile(
    r"(who|what) (are|r) (you|u)"
    r"|what (can|do) (you|u) do"
    r"|what (are|is) (you|your) (for|purpose)"
    r"|how (do|does) (you|this) work"
    r"|are (you|u) (a |an )?(bot|robot|human|real|person|ai|chatgpt|gpt)"
    r"|help|menu|options|start"
)

# Smalltalk vocabulary. A message qualifies only when *every* token is here.
_GREETING = {
    "hi", "hii", "hiii", "hey", "heya", "hello", "helo", "hiya", "yo",
    "namaste", "namaskar", "greetings", "morning", "afternoon", "evening",
}
_WELLBEING = {
    "how", "are", "is", "you", "u", "r", "doing", "going", "hows", "whats",
    "up", "today", "hope", "well", "everything", "life", "it",
}
_THANKS = {
    "thanks", "thank", "thankyou", "thx", "thnx", "ty", "cheers", "appreciate",
    "appreciated", "great", "perfect", "awesome", "excellent", "brilliant",
    "nice", "cool", "ok", "okay", "okey", "k", "got", "understood", "sure",
    "alright", "right", "fine", "lovely", "helpful",
}
_FAREWELL = {
    "bye", "byee", "goodbye", "see", "ya", "later", "night", "gn", "goodnight",
    "take", "care", "farewell", "ciao", "adios", "tata",
}
_FILLER = {
    "good", "there", "please", "and", "a", "an", "the", "so", "much", "very",
    "lot", "lots", "my", "friend", "mate", "sir", "maam", "madam", "dear",
    "then", "just", "for", "your", "help", "that", "s", "all", "im", "i", "am",
    "to", "of", "me", "no", "yes", "yeah", "yep", "nope", "not", "now",
}

_SMALLTALK = _GREETING | _WELLBEING | _THANKS | _FAREWELL | _FILLER

# Apostrophes and emoji are noise for this decision; digits are not - "hi 2"
# is odd enough to send down the normal path.
_NORMALISE = re.compile(r"[^a-z\s]+")


def classify_intent(message: str) -> Intent:
    """Route one visitor turn. Never raises, never calls a model."""
    text = _NORMALISE.sub(" ", (message or "").lower())
    tokens = text.split()
    if not tokens or len(tokens) > _MAX_SMALLTALK_TOKENS:
        return Intent.INFORMATIONAL

    # Checked before the vocabulary, because "what can you do" is built from
    # words the bag would not otherwise recognise.
    if _CAPABILITY.fullmatch(" ".join(tokens)):
        return Intent.CAPABILITY

    if not all(token in _SMALLTALK for token in tokens):
        return Intent.INFORMATIONAL

    # Precedence, not first match: "thanks, bye" is a farewell, "hi thanks"
    # is a greeting - in both cases the later intent is the real one.
    present = set(tokens)
    if present & _FAREWELL:
        return Intent.FAREWELL
    if present & _GREETING:
        return Intent.GREETING
    if present & _THANKS:
        return Intent.THANKS
    if present & _WELLBEING:
        return Intent.GREETING  # a bare "how are you"
    return Intent.INFORMATIONAL
