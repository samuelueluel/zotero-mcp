"""Bounded, source-preserving context primitives; no inference or I/O.

Evidence IDs bind an indexed chunk to its library, text, and source metadata.
They are locators, not credentials: adapters must still enforce library scope.
"""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from typing import Any

MAX_CONTEXT_CHARS = 16000
DEFAULT_CONTEXT_CHARS = 8000
KEY_PATTERN = re.compile(r"[A-Z0-9]{8}\Z")
ID_PATTERN = re.compile(r"zr1:(0|[1-9][0-9]{0,15}):([A-Z0-9]{8}(?:#[0-9]{1,7})?):([a-f0-9]{64})\Z")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_fingerprint(text: str, metadata: dict[str, Any]) -> str:
    # Include location/source changes, even if the text at a reused #N is identical.
    payload = {"text": text, "metadata": metadata}
    return text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def evidence_id(chunk_id: str, text: str, metadata: dict[str, Any]) -> str | None:
    group = metadata.get("group_id")
    if isinstance(group, bool) or not isinstance(group, int) or group < 0:
        return None
    value = f"zr1:{group}:{chunk_id}:{chunk_fingerprint(text, metadata)}"
    return value if ID_PATTERN.fullmatch(value) else None


def parse_evidence_id(value: str) -> tuple[int, str, str]:
    match = ID_PATTERN.fullmatch(value)
    if not match:
        raise ValueError("Invalid evidence_id; copy the ID returned by semantic_search.")
    return int(match[1]), match[2], match[3]


def source_preview(query: str, text: str, width: int = 600) -> tuple[str, int]:
    """Choose a bounded verbatim sentence/paragraph window, ignoring DCR headers.

    This is still lexical selection, not a claim that the window answers the
    query. Scores operate on content rather than the injected paper breadcrumb.
    Offsets refer to the original string; no OCR/whitespace repairs are made.
    """
    if not text:
        return "", 0
    content_start = 0
    if text.startswith("[Paper:"):
        end = text.find("\n")
        if end >= 0:
            content_start = end + 1
    terms = set(re.findall(r"\w{3,}", query.lower()))
    spans = list(re.finditer(r"[^\n]+(?:\n|$)", text[content_start:]))
    # Paragraph/sentence starts give natural boundaries without rewriting text.
    starts: set[int] = set()
    for span in spans:
        start = content_start + span.start()
        if text[start:].startswith("#"):
            continue
        starts.add(start)
        for boundary in re.finditer(r"(?<=[.!?])\s+", span.group()):
            starts.add(start + boundary.end())
    natural_starts = starts.copy()
    # Long unbroken paragraphs/HTML rows may have no sentence boundary near
    # the match. Allow a clipped window there, but prefer natural boundaries
    # when they cover equally many query terms.
    for match in re.finditer(r"\w{3,}", text[content_start:]):
        if match.group().lower() in terms:
            starts.add(max(content_start, content_start + match.start() - width // 3))
    starts = starts or {content_start}

    def score(start: int) -> tuple[int, int, bool, int]:
        window = text[start : start + width]
        content = "\n".join(line for line in window.splitlines() if not line.lstrip().startswith(("#", "[Paper:")))
        words = set(re.findall(r"\w{3,}", content.lower()))
        leading_words = set(re.findall(r"\w{3,}", content[:150].lower()))
        return len(terms & words), len(terms & leading_words), start in natural_starts, -start

    start = max(starts, key=score)
    end = min(len(text), start + width)
    if end < len(text):
        boundaries = list(re.finditer(r"[.!?](?=\s)|\n", text[start:end]))
        # Don't shrink to a tiny heading/preamble just to force a boundary.
        if boundaries and boundaries[-1].end() >= width // 2:
            candidate_end = start + boundaries[-1].end()
            # A sentence trim must not throw away the query-bearing tail.
            tail_terms = set(re.findall(r"\w{3,}", text[candidate_end:end].lower())) & terms
            kept_terms = set(re.findall(r"\w{3,}", text[start:candidate_end].lower())) & terms
            if tail_terms <= kept_terms:
                end = candidate_end
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return text[start:end], start


def line_starts(text: str) -> list[int]:
    return [0] + [m.end() for m in re.finditer("\n", text) if m.end() < len(text)]


def source_window(text: str, start: int, end: int, starts: list[int]) -> dict[str, Any]:
    return {
        "text": text[start:end],
        "char_start": start,
        "char_end": end,
        "start_line": bisect_right(starts, start),
        "end_line": bisect_right(starts, max(start, end - 1)),
        "starts_mid_line": start not in starts,
        "ends_mid_line": end < len(text) and end not in starts,
    }


def find_windows(
    text: str,
    query: str | None,
    *,
    start_line: int = 1,
    end_line: int | None = None,
    context_lines: int = 3,
    max_matches: int = 5,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Literal search or line-range read with a total text budget and continuation.

    Literal matching is Unicode case-insensitive via escaped regex (no user
    regex execution). Match positions remain offsets into the original text.
    """
    if not 256 <= max_chars <= MAX_CONTEXT_CHARS:
        raise ValueError(f"max_chars must be between 256 and {MAX_CONTEXT_CHARS}.")
    if not 1 <= max_matches <= 10 or not 0 <= context_lines <= 20:
        raise ValueError("max_matches must be 1–10 and context_lines 0–20.")
    starts = line_starts(text)
    if start_line < 1 or start_line > len(starts):
        raise ValueError("start_line is outside the source.")
    if end_line is not None and (end_line < start_line or end_line > len(starts)):
        raise ValueError("end_line is outside the requested source range.")
    if query is not None and (not query.strip() or len(query) > 500):
        raise ValueError("query must contain 1–500 characters, or be null for a line read.")
    lo = starts[start_line - 1]
    last = end_line or (len(starts) if query is not None else min(start_line + 39, len(starts)))
    hi = starts[last] if last < len(starts) else len(text)
    windows: list[dict[str, Any]] = []
    truncated = False
    next_line = None
    if query is None:
        end = min(hi, lo + max_chars)
        window = source_window(text, lo, end, starts)
        window["truncated"] = end < hi
        windows.append(window)
        truncated = end < hi
        next_line = bisect_right(starts, end) if end < len(text) else None
    else:
        used = 0
        covered_until = lo
        for match in re.finditer(re.escape(query), text[lo:hi], flags=re.IGNORECASE):
            mstart, mend = lo + match.start(), lo + match.end()
            if mend <= covered_until:
                continue
            if len(windows) >= max_matches or max_chars - used < len(query):
                truncated, next_line = True, bisect_right(starts, mstart)
                break
            line = bisect_right(starts, mstart) - 1
            start = max(lo, covered_until, starts[max(0, line - context_lines)])
            last_line = min(len(starts), bisect_right(starts, mend - 1) + context_lines)
            desired_end = min(hi, starts[last_line] if last_line < len(starts) else len(text))
            budget = max_chars - used
            # A very long table row must not hide the actual matched text.
            if mend > start + budget:
                start = max(start, mstart - max(0, (budget - (mend - mstart)) // 2))
            end = min(desired_end, start + budget)
            window = source_window(text, start, end, starts)
            window.update(
                {
                    "match_char_start": mstart,
                    "match_char_end": mend,
                    "truncated": end < desired_end or start > starts[max(0, line - context_lines)],
                }
            )
            windows.append(window)
            used += end - start
            covered_until = end
            truncated = truncated or window["truncated"]
    return {
        "windows": windows,
        "truncated": truncated,
        "next_start_line": next_line,
        "total_lines": len(starts),
        "source_chars": len(text),
    }
