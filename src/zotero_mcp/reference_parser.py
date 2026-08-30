"""Conservative extraction of bibliography sections and entries from sidecars.

MinerU output is heterogeneous and often OCR-damaged. The parser therefore
returns bounded, provenance-preserving strings and explicit confidence/method
labels rather than pretending every boundary is exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})[ \t]+(?P<title>[^\n]+?)\s*$", re.MULTILINE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_BRACKET_MARKER_RE = re.compile(r"(?m)^[ \t]*\[(?P<number>\d{1,4})\][ \t]+")
_BRACKET_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9])\[(?P<label>[^]\n]{2,80})\][ \t]*"
)
_DOT_MARKER_RE = re.compile(r"(?m)^[ \t]*(?P<number>\d{1,4})[.)][ \t]+")
_AUTHOR_START_RE = re.compile(
    r"^[A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’`.-]{1,70}"
    r"(?:,|\s+(?:and|&)\s+|\s+[A-Z](?:\.|\b))"
)
_INITIAL_AUTHOR_START_RE = re.compile(
    r"^(?:[A-ZÀ-ÖØ-öø-ÿ]\.\s+){1,3}"
    r"[A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’`.-]{1,70}"
    r"(?:,|\s+(?:and|&)\s+|\.)"
)
_REFERENCE_TERMS = (
    "references",
    "reference",
    "bibliography",
    "bibliographic",
    "works cited",
    "literature cited",
    "literature references",
)


@dataclass(frozen=True)
class ReferenceSection:
    heading: str
    level: int
    body: str
    start: int
    end: int
    confidence: float
    method: str
    section_index: int = 1


@dataclass(frozen=True)
class ReferenceEntry:
    index: int
    raw_text: str
    split_method: str
    confidence: float
    marker: str = ""


def _clean_text(text: str) -> str:
    # Preserve offsets while making common OCR control bytes harmless.
    return (text or "").replace("\x00", " ").replace("\ufeff", "")


def _normalise_heading(title: str) -> str:
    title = re.sub(r"[`*_~]", "", title or "")
    title = re.sub(r"\s+", " ", title).strip().lower()
    return title.strip(" :>\t")


def _heading_kind(title: str) -> str | None:
    normalized = _normalise_heading(title)
    if not normalized:
        return None
    if any(term in normalized for term in ("works cited", "literature cited", "literature references")):
        return "high"
    # Permit breadcrumbs such as "Contents > References" but avoid ordinary
    # prose headings that merely mention a reference.
    if normalized == "bibliography" or normalized.endswith(" bibliography"):
        return "high"
    if normalized == "references" or normalized.endswith(" references"):
        return "high"
    if normalized == "bibliographic references" or normalized.endswith(" bibliographic references"):
        return "high"
    if normalized == "reference" or normalized.endswith(" reference"):
        return "singular"
    return None


def _section_body(text: str, headings: list[re.Match[str]], position: int) -> tuple[str, int]:
    heading = headings[position]
    level = len(heading.group("marks"))
    end = len(text)
    for next_heading in headings[position + 1 :]:
        if len(next_heading.group("marks")) <= level:
            end = next_heading.start()
            break
    body = text[heading.end() : end].strip("\n\r \t")
    return body, end


def _section_score(kind: str, body: str, position: int, total: int) -> float:
    score = 5.0 if kind == "high" else 1.0
    years = len(_YEAR_RE.findall(body))
    dois = len(_DOI_RE.findall(body))
    markers = len(_BRACKET_MARKER_RE.findall(body)) + len(_DOT_MARKER_RE.findall(body))
    if len(body) >= 100:
        score += 1.0
    if years:
        score += min(years, 20) / 20.0
    if dois:
        score += 0.5
    if markers >= 2:
        score += 0.75
    if len(body) < 60:
        score -= 2.0
    if re.search(r"(?im)^\s*(?:also see|see also|syntax|description)\b", body):
        score -= 1.0
    # Prefer later sections when otherwise tied; true bibliographies usually
    # occur near the end, while contents breadcrumbs occur near the front.
    if total > 1:
        score += 0.25 * position / (total - 1)
    return score


def _reference_candidates(
    text: str,
) -> list[tuple[re.Match[str], str, str, int, float]]:
    clean = _clean_text(text)
    headings = list(_HEADING_RE.finditer(clean))
    candidates: list[tuple[re.Match[str], str, str, int, float]] = []
    for position, heading in enumerate(headings):
        kind = _heading_kind(heading.group("title"))
        if kind is None:
            continue
        body, end = _section_body(clean, headings, position)
        score = _section_score(kind, body, position, len(headings))
        candidates.append((heading, kind, body, end, score))
    return candidates


def _usable_reference_body(body: str) -> bool:
    if len(body.strip()) < 20:
        return False
    return bool(
        _YEAR_RE.search(body)
        or _DOI_RE.search(body)
        or len(_BRACKET_MARKER_RE.findall(body)) >= 2
        or len(_DOT_MARKER_RE.findall(body)) >= 2
        or len(_BRACKET_LABEL_RE.findall(body)) >= 2
    )


def _make_reference_section(
    candidate: tuple[re.Match[str], str, str, int, float],
    section_index: int,
) -> ReferenceSection:
    heading, kind, body, end, score = candidate
    if kind == "high":
        confidence = min(0.99, 0.80 + min(max(score - 5.0, 0.0), 1.0) * 0.15)
        method = "heading"
    else:
        confidence = min(0.85, 0.60 + min(max(score - 1.0, 0.0), 1.0) * 0.20)
        method = "singular-heading"
    return ReferenceSection(
        heading=heading.group("title").strip(),
        level=len(heading.group("marks")),
        body=body,
        start=heading.end(),
        end=end,
        confidence=confidence,
        method=method,
        section_index=section_index,
    )


def extract_reference_sections(text: str) -> list[ReferenceSection]:
    """Extract all usable bibliography-like sections from one sidecar.

    Multiple plural sections are retained because Stata reference manuals and
    similar manuals contain one bibliography subsection per command. Repeated
    singular ``Reference`` headings remain excluded unless they are the lone
    usable candidate in the file.
    """
    candidates = _reference_candidates(text)
    high = [candidate for candidate in candidates if candidate[1] == "high" and _usable_reference_body(candidate[2])]
    if high:
        return [
            _make_reference_section(candidate, index)
            for index, candidate in enumerate(high, 1)
        ]

    singular = [
        candidate
        for candidate in candidates
        if candidate[1] == "singular" and _usable_reference_body(candidate[2])
    ]
    if len(singular) == 1:
        return [_make_reference_section(singular[0], 1)]
    return []


def extract_reference_section(text: str) -> ReferenceSection | None:
    """Extract the highest-scoring bibliography-like section."""
    sections = extract_reference_sections(text)
    if not sections:
        return None
    return max(sections, key=lambda section: section.confidence)


def _normalise_entry(value: str, limit: int = 4000) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    if len(value) > limit:
        return value[: limit - 3].rstrip() + "..."
    return value


def _markers_are_plausible(matches: list[re.Match[str]]) -> bool:
    if len(matches) < 2:
        return False
    numbers = [int(match.group("number")) for match in matches]
    if len(set(numbers)) != len(numbers):
        return False
    sequential = sum(
        1 for left, right in zip(numbers, numbers[1:]) if right == left + 1
    )
    return sequential >= max(1, (len(numbers) - 1) // 2)


def _split_marked(
    body: str,
    pattern: re.Pattern[str],
    method: str,
    confidence: float,
) -> list[ReferenceEntry] | None:
    matches = list(pattern.finditer(body))
    if not _markers_are_plausible(matches):
        return None
    entries: list[ReferenceEntry] = []
    prefix = body[: matches[0].start()].strip()
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
        raw = body[start:end]
        if position == 0 and prefix:
            raw = prefix + " " + raw
        raw = _normalise_entry(raw)
        if raw:
            entries.append(
                ReferenceEntry(
                    index=len(entries) + 1,
                    raw_text=raw,
                    split_method=method,
                    confidence=confidence,
                    marker=match.group("number"),
                )
            )
    return entries if len(entries) >= 2 else None


def _split_labeled_brackets(body: str) -> list[ReferenceEntry] | None:
    matches = [
        match
        for match in _BRACKET_LABEL_RE.finditer(body)
        if match.group("label").strip()
        and not match.group("label").strip()[0].isdigit()
        and any(char.isdigit() for char in match.group("label"))
    ]
    if len(matches) < 2:
        return None
    entries: list[ReferenceEntry] = []
    prefix = body[: matches[0].start()].strip()
    for position, match in enumerate(matches):
        start = match.start()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
        raw = body[start:end]
        if position == 0 and prefix:
            raw = prefix + " " + raw
        raw = _normalise_entry(raw)
        if raw:
            entries.append(
                ReferenceEntry(
                    index=len(entries) + 1,
                    raw_text=raw,
                    split_method="bracket-label",
                    confidence=0.82,
                    marker=match.group("label").strip(),
                )
            )
    return entries if len(entries) >= 2 else None


def _looks_like_entry_start(line: str) -> bool:
    line = line.strip()
    if not line or line.startswith(("http://", "https://")):
        return False
    if not (_YEAR_RE.search(line[:240]) or _DOI_RE.search(line[:240])):
        return False
    return (
        bool(_AUTHOR_START_RE.match(line))
        or bool(_INITIAL_AUTHOR_START_RE.match(line))
        or bool(re.match(r"^[A-Z][^\n]{1,100},", line))
    )


def _split_lines(body: str) -> list[ReferenceEntry] | None:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    starts = [index for index, line in enumerate(lines) if _looks_like_entry_start(line)]
    if len(starts) < 2:
        return None

    # A short preface is common in book bibliographies. Do not attach it to the
    # first citation unless it itself contains a year/DOI and looks like a
    # wrapped entry.
    first_start = starts[0]
    prefix = lines[:first_start]
    keep_prefix = any(
        _YEAR_RE.search(line[:240]) or _DOI_RE.search(line[:240]) for line in prefix
    )
    start_set = set(starts)
    entries: list[ReferenceEntry] = []
    current: list[str] = list(prefix) if keep_prefix else []
    for index, line in enumerate(lines):
        if index < first_start:
            continue
        if index in start_set and current:
            raw = _normalise_entry(" ".join(current))
            if raw:
                entries.append(ReferenceEntry(len(entries) + 1, raw, "line", 0.78))
            current = []
        current.append(line)
    raw = _normalise_entry(" ".join(current))
    if raw:
        entries.append(ReferenceEntry(len(entries) + 1, raw, "line", 0.78))
    if len(entries) < 2:
        return None

    # Refine lines that contain two citations after OCR removed the newline.
    refined: list[ReferenceEntry] = []
    for entry in entries:
        inline = _split_year_boundaries(entry.raw_text, require_url=True)
        if inline and len(inline) >= 2:
            refined.extend(
                ReferenceEntry(
                    index=0,
                    raw_text=part.raw_text,
                    split_method="inline-year-boundary",
                    confidence=0.55,
                    marker=part.marker,
                )
                for part in inline
            )
        else:
            refined.append(entry)
    return [
        ReferenceEntry(index, entry.raw_text, entry.split_method, entry.confidence, entry.marker)
        for index, entry in enumerate(refined, 1)
    ] if len(refined) >= 2 else None


def _split_compound_years(entries: list[ReferenceEntry]) -> list[ReferenceEntry]:
    """Split obvious same-line citations, chiefly OCR'd web bibliographies."""
    boundary_re = re.compile(r"(?<=\.)[ \t]+(?=(?:19|20)\d{2}[a-z]?\.)", re.IGNORECASE)
    refined: list[ReferenceEntry] = []
    for entry in entries:
        # Require multiple URLs before splitting. This avoids turning a single
        # work with several publication years (translations/revisions) into
        # false entries.
        if len(_YEAR_RE.findall(entry.raw_text)) < 2 or len(
            re.findall(r"https?://", entry.raw_text, re.IGNORECASE)
        ) < 2:
            refined.append(entry)
            continue
        starts = [
            match.start() + 1
            for match in boundary_re.finditer(entry.raw_text)
            if "http" in entry.raw_text[: match.start()].lower()
        ]
        if not starts:
            refined.append(entry)
            continue
        chunks: list[str] = []
        first = entry.raw_text[: starts[0]].strip()
        if first:
            chunks.append(first)
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(entry.raw_text)
            chunk = entry.raw_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        if len(chunks) < 2:
            refined.append(entry)
            continue
        refined.extend(
            ReferenceEntry(index, _normalise_entry(chunk), "compound-year", 0.65)
            for index, chunk in enumerate(chunks, len(refined) + 1)
        )

    return [
        ReferenceEntry(index, entry.raw_text, entry.split_method, entry.confidence, entry.marker)
        for index, entry in enumerate(refined, 1)
    ]


def _split_year_boundaries(
    body: str,
    require_url: bool = False,
) -> list[ReferenceEntry] | None:
    # OCR sometimes collapses new entries onto a single line. Look for an
    # author-like phrase whose preceding local context already contains a
    # publication year; this avoids splitting at the first author/year pair.
    author_re = re.compile(
        r"(?<![A-Za-z])(?P<author>(?:(?:de|da|del|van|von|der|den|la|le)\s+)?"
        r"[A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’`.-]{1,70}"
        r"(?:,|\s+(?:and|&)\s+))"
    )
    boundary_context_re = re.compile(
        r"(?:https?://\S+|\d{1,5}\s*[-–—]\s*\d{1,5})[.;:)]*\s*$",
        re.IGNORECASE,
    )
    starts: list[int] = []
    for match in author_re.finditer(body):
        start = match.start("author")
        if start == 0:
            continue
        prefix = body[max(0, start - 260):start]
        # Require strong end-of-entry punctuation: an URL or page range. A
        # bare year/full stop is too common inside author names and titles.
        if not _YEAR_RE.search(prefix) or not boundary_context_re.search(prefix):
            continue
        if require_url and not re.search(r"https?://\S+[.;:)]*\s*$", prefix, re.IGNORECASE):
            continue
        if re.search(r"[,;]\s*$", prefix):
            continue
        starts.append(start)
    starts = sorted(set(starts))
    if not starts:
        return None
    chunks: list[str] = []
    first = body[: starts[0]].strip()
    if first:
        chunks.append(first)
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(body)
        chunk = body[start:end].strip()
        if chunk:
            chunks.append(chunk)
    entries = [
        ReferenceEntry(index, _normalise_entry(chunk), "year-boundary", 0.55)
        for index, chunk in enumerate(chunks, 1)
        if _normalise_entry(chunk)
    ]
    return entries if len(entries) >= 2 else None


def parse_reference_entries(body: str) -> list[ReferenceEntry]:
    """Split one bibliography body into bounded, ordered entries."""
    body = _clean_text(body).strip()
    if not body:
        return []

    entries = _split_marked(body, _BRACKET_MARKER_RE, "bracket-number", 0.95)
    if entries:
        return entries
    entries = _split_labeled_brackets(body)
    if entries:
        return entries
    entries = _split_marked(body, _DOT_MARKER_RE, "dot-number", 0.90)
    if entries:
        return entries

    entries = _split_lines(body)
    if entries:
        return _split_compound_years(entries)
    entries = _split_year_boundaries(body)
    if entries:
        return entries

    raw = _normalise_entry(body)
    return [ReferenceEntry(1, raw, "whole-section", 0.30)] if raw else []


def iter_reference_entries(text: str) -> tuple[ReferenceSection | None, list[ReferenceEntry]]:
    section = extract_reference_section(text)
    return section, parse_reference_entries(section.body) if section else []


def iter_all_reference_entries(
    text: str,
) -> Iterable[tuple[ReferenceSection, ReferenceEntry]]:
    """Yield entries from every usable section with a file-global ordinal."""
    ordinal = 0
    for section in extract_reference_sections(text):
        for entry in parse_reference_entries(section.body):
            ordinal += 1
            yield section, ReferenceEntry(
                index=ordinal,
                raw_text=entry.raw_text,
                split_method=entry.split_method,
                confidence=entry.confidence,
                marker=entry.marker,
            )


def reference_dois(raw_text: str) -> list[str]:
    """Return unique DOI-like strings in a parsed entry, preserving order."""
    seen: set[str] = set()
    dois: list[str] = []
    for match in _DOI_RE.finditer(raw_text or ""):
        value = match.group(0).rstrip(".,;:)]}>").lower()
        if value and value not in seen:
            seen.add(value)
            dois.append(value)
    return dois
