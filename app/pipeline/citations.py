"""Citation markers: read them, then take them off (Layer 02).

`[1]` does two jobs that pull in opposite directions.

Inside the pipeline it is the provenance mechanism. The answer model is told to
put one after every claim about the Property, and `build_citations` reads them
back to decide which of the retrieved sources the answer actually leaned on -
listing all eight would imply it leaned on all eight. An answer carrying none
of them is a non-answer however warmly it is phrased, and is deflected without
being shown.

On screen it is clutter. A visitor reading "check-in is from 2pm [1]" is not
going to open the source and check, so the marker costs legibility and buys
nothing at the point where it is read.

So the markers are stripped, and *where* is the whole design: after
`build_citations` has read them, after the verifier has run against the text
that carries them, and after the turn has been filed in the chat log with them
intact. What reaches a Visitor is clean; what the system reasons about, and
what an operator reads back later to check an answer was legitimate, is not.

Streaming makes that harder than a regex. A marker arrives split across tokens
- `[`, then `1`, then `].` - so the filter has to hold back anything that
could still become one, and release it when it turns out not to be. It also
has to hold the whitespace *before* a `[`, because "2pm [1]." must not leave
"2pm ." behind.
"""

from __future__ import annotations

import re

from app.models.domain import Citation, ScoredChunk

# What sits between a claim and its marker, or between two markers: spaces, and
# the comma a model writes in "staffed 24 hours [1], [2]". Taking the separator
# with the marker is what stops "hours,." coming out the other side.
_SEPARATOR = r"[ \t]*,?[ \t]*"

# `[1]`, and the `[1, 2]` a model occasionally writes instead of `[1][2]`.
_MARKER = re.compile(rf"{_SEPARATOR}\[\d+(?:\s*,\s*\d+)*\]")
_LEADING_MARKER = re.compile(rf"^{_SEPARATOR}\[\d+(?:\s*,\s*\d+)*\]")
# Text that could still become a marker once more tokens arrive.
_UNFINISHED = re.compile(rf"^{_SEPARATOR}(?:\[[\d,\s]*)?$")
_TRAILING_SEPARATOR = re.compile(rf"{_SEPARATOR}$")
_UNFINISHED_AT_END = re.compile(rf"{_SEPARATOR}\[[\d,\s]*$")

# A '[' that never closes must not stall the stream forever.
_HOLD_LIMIT = 24


def strip_markers(text: str) -> str:
    """Remove every citation marker from a finished answer.

    Including a half-written one at the very end: an answer truncated on
    max_tokens can stop mid-marker, and "staffed 24 hours [1" is worse to read
    than either the marker or its absence.
    """
    text = _MARKER.sub("", text)
    text = _UNFINISHED_AT_END.sub("", text)
    return text.lstrip()


class MarkerFilter:
    """`strip_markers` for a token stream.

    Feed tokens in, get display-safe text out. The contract that matters is
    that it agrees with `strip_markers` no matter where the token boundaries
    fall - which is what `tests/test_citations.py` checks by splitting the same
    answer every possible way.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._started = False

    def feed(self, token: str) -> str:
        self._pending += token
        return self._drain(final=False)

    def flush(self) -> str:
        """Whatever is left, once no more tokens are coming."""
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        out: list[str] = []

        while self._pending:
            marker = _LEADING_MARKER.match(self._pending)
            if marker:
                self._pending = self._pending[marker.end() :]
                continue

            # Might still become one. Wait, unless nothing more is coming or
            # it has gone on too long to be a marker.
            if _UNFINISHED.match(self._pending) and len(self._pending) <= _HOLD_LIMIT:
                if not final:
                    break
                if "[" in self._pending:
                    # The stream stopped mid-marker. Drop it rather than
                    # leaving "[1" on screen.
                    self._pending = ""
                    break

            out.append(self._emit(final))
            if self._held_for_more(final):
                break

        text = "".join(out)
        if not self._started:
            # A marker opening the answer would otherwise leave it indented.
            text = text.lstrip()
            if text:
                self._started = True
        return text

    def _emit(self, final: bool) -> str:
        """Take the next run of text that cannot be part of a marker."""
        opening = self._pending.find("[")

        if opening == -1:
            chunk, self._pending = self._pending, ""
            if not final:
                # Trailing whitespace might yet turn out to precede a marker.
                held = _TRAILING_SEPARATOR.search(chunk)
                if held and held.start() > 0:
                    self._pending = chunk[held.start() :]
                    chunk = chunk[: held.start()]
            return chunk

        # Hold back the separator leading into the '['; it goes with the
        # marker if that is what follows.
        head = self._pending[:opening]
        held = _TRAILING_SEPARATOR.search(head)
        cut = held.start() if held else opening

        if cut == 0:
            # Nothing before the '[' but separator, and it is not a marker or
            # the caller would not be here - so the bracket is literal text.
            cut = opening + 1

        chunk, self._pending = self._pending[:cut], self._pending[cut:]
        return chunk

    def _held_for_more(self, final: bool) -> bool:
        return not final and bool(self._pending) and not self._pending.startswith("[")


def build_citations(chunks: list[ScoredChunk], answer: str) -> list[Citation]:
    """Return only the sources the answer actually cited.

    Reads the markers, so it must run on the answer *before* it is stripped.

    Listing all eight retrieved chunks would imply the answer leaned on all of
    them. Citations are a checkable claim about provenance, so they track the
    [n] markers the model actually wrote.
    """
    cited: list[Citation] = []
    for i, scored in enumerate(chunks, start=1):
        if f"[{i}]" not in answer:
            continue
        chunk = scored.chunk
        snippet = chunk.text.strip().replace("\n", " ")
        cited.append(
            Citation(
                index=i,
                uri=chunk.uri,
                label=scored.citation_label,
                snippet=snippet[:220] + ("..." if len(snippet) > 220 else ""),
                published_on=chunk.fetched_at,
                unit=chunk.unit,
            )
        )
    return cited
