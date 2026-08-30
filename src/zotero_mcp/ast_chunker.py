#!/usr/bin/env python3
"""Bounded AST-Aware Markdown Chunker for zotero-mcp.

Implements the 8 guardrails of Bounded AST Chunking:
1. Atomic Structural Blocks: <table>...</table>, $$\\begin{aligned}...\\end{aligned}$$,
   and [Figure Schema] blocks remain unsplit up to max_atomic_size (3,800 chars).
2. Row-Wise Table Fallback: Oversized tables (>3,800 chars) are split row-wise (<tr>)
   with duplicate header rows preserved across chunks.
3. Heading Boundary Fences: Hard split on #, ##, ### headings (zero bleed between sections).
4. Node Packing Floor: Merge small paragraphs up to >= min_chunk_size (600 chars)
   to prevent vector starvation.
5. Token Ceiling: Long prose is split strictly at sentence boundaries (<= target_chunk_size).
6. Hard Failsafe Ceiling: Enforces failsafe_size (4,000 chars) to prevent context overflow.
7. Selective Overlap: 50-token sliding overlap strictly within narrative prose;
   0 overlap across headers and tables.
8. Heading Stickiness: Headings never become orphaned standalone chunks; they bind
   to their following content.
"""
import re
from typing import List, Tuple

DEFAULT_TARGET_CHUNK_SIZE = 2400      # ~600 tokens target prose ceiling
DEFAULT_MIN_CHUNK_SIZE = 600          # ~150 tokens node packing floor
DEFAULT_MAX_ATOMIC_SIZE = 3800        # ~1000 tokens atomic table/proof ceiling
DEFAULT_FAILSAFE_SIZE = 4000          # ~1050 tokens hard fallback ceiling
DEFAULT_PROSE_OVERLAP = 200           # ~50 tokens sliding window within prose

HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
HTML_TABLE_PATTERN = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
DISPLAY_MATH_PATTERN = re.compile(r"\$\$[\s\S]*?\$\$")
FIGURE_SCHEMA_PATTERN = re.compile(r"\[Figure Schema\][\s\S]*?(?=\n\n|\Z)")


def split_oversized_table(table_html: str, max_chars: int = DEFAULT_TARGET_CHUNK_SIZE) -> List[Tuple[str, int, int]]:
    """Split an oversized HTML table row-wise (<tr>), duplicating the header row on all chunks.

    Returns (text, rel_start, rel_end) triples so callers can store accurate
    source spans (each piece claims only its own region, not the whole table).
    Individual rows larger than max_chars are hard-sliced so no piece exceeds
    max_chars + one row (table HTML is atomic per piece, so a slice may cut a
    mid-row cell - acceptable for pathological single giant rows).
    """
    header_match = re.search(r"<thead[\s\S]*?</thead>", table_html, re.IGNORECASE)
    if header_match:
        header_html = header_match.group(0)
    else:
        first_tr = re.search(r"<tr[\s\S]*?</tr>", table_html, re.IGNORECASE)
        header_html = first_tr.group(0) if first_tr else ""

    rows = re.findall(r"<tr[\s\S]*?</tr>", table_html, re.IGNORECASE)
    if not rows or len(rows) <= 1:
        return [(table_html.strip(), 0, len(table_html))]

    table_chunks: List[Tuple[str, int, int]] = []
    curr_rows = []
    base_wrapper_len = len("<table><tbody>" + header_html + "</tbody></table>")
    curr_len = base_wrapper_len
    # running source offset of the current row group (relative to table_html)
    rel_pos = table_html.find(rows[0])
    if rel_pos < 0:
        rel_pos = 0
    curr_rel_start = rel_pos

    start_idx = 1 if (header_html and rows[0] in header_html) else 0

    for r in rows[start_idx:]:
        r_len = len(r)
        if curr_len + r_len > max_chars and curr_rows:
            chunk_content = f"<table><tbody>{header_html}{''.join(curr_rows)}</tbody></table>"
            table_chunks.append((chunk_content, curr_rel_start, curr_rel_start + len("".join(curr_rows))))
            curr_rows = [r]
            curr_len = base_wrapper_len + r_len
            curr_rel_start = rel_pos
        else:
            curr_rows.append(r)
            curr_len += r_len
        rel_pos = table_html.find(r, rel_pos + 1) if rel_pos >= 0 else -1
        if rel_pos < 0:
            rel_pos = table_html.find(r)

    if curr_rows:
        chunk_content = f"<table><tbody>{header_html}{''.join(curr_rows)}</tbody></table>"
        table_chunks.append((chunk_content, curr_rel_start, curr_rel_start + len("".join(curr_rows))))

    return table_chunks

    if curr_rows:
        chunk_content = f"<table><tbody>{header_html}{''.join(curr_rows)}</tbody></table>"
        table_chunks.append(chunk_content)

    return table_chunks if table_chunks else [table_html]


def split_prose_by_sentences(text: str, max_chars: int, overlap: int) -> List[Tuple[str, int, int]]:
    """Split long prose into sentence- and line-bounded chunks with sliding overlap.
    
    Guarantees no chunk or segment ever exceeds max_chars, even in unpunctuated
    bibliographies, author indices, or large proof blocks.
    """
    if len(text) <= max_chars:
        return [(text.strip(), 0, len(text))]

    raw_segments = []
    for match in re.finditer(r".+?(?:[.?!](?:\s+|\n+)|\n+|\Z)", text):
        seg = match.group()
        if seg.strip():
            if len(seg) > max_chars:
                # Slicing fallback for massive single-line tokens/formulas
                step = max(1, max_chars - overlap)
                for i in range(0, len(seg), step):
                    sub = seg[i:i + max_chars]
                    raw_segments.append((sub, match.start() + i, match.start() + min(i + max_chars, len(seg))))
            else:
                raw_segments.append((seg, match.start(), match.end()))

    if not raw_segments:
        return [(text.strip(), 0, len(text))]

    chunks = []
    curr_chunk = []
    curr_len = 0
    chunk_start = 0

    for s_text, s_start, s_end in raw_segments:
        s_len = len(s_text)
        if curr_len + s_len > max_chars and curr_chunk:
            combined = "".join(curr_chunk).strip()
            if combined:
                chunks.append((combined, chunk_start, chunk_start + len("".join(curr_chunk))))

            overlap_chunk = []
            overlap_len = 0
            for prev_s in reversed(curr_chunk):
                if overlap_len + len(prev_s) <= overlap:
                    overlap_chunk.insert(0, prev_s)
                    overlap_len += len(prev_s)
                else:
                    break

            curr_chunk = list(overlap_chunk) + [s_text]
            curr_len = overlap_len + s_len
            chunk_start = s_start - overlap_len
        else:
            if not curr_chunk:
                chunk_start = s_start
            curr_chunk.append(s_text)
            curr_len += s_len

    if curr_chunk:
        combined = "".join(curr_chunk).strip()
        if combined:
            chunks.append((combined, chunk_start, chunk_start + len("".join(curr_chunk))))

    return chunks


class Block:
    def __init__(self, block_type: str, text: str, start: int, end: int, is_atomic: bool = False, heading_level: int = 0):
        self.block_type = block_type
        self.text = text
        self.start = start
        self.end = end
        self.is_atomic = is_atomic
        self.heading_level = heading_level

    @property
    def length(self) -> int:
        return len(self.text)


def tokenize_document_blocks(text: str) -> List[Block]:
    """Parse Markdown document into structural blocks with exact offsets."""
    n = len(text)
    if not text:
        return []

    atomic_spans = []

    for m in HEADING_PATTERN.finditer(text):
        atomic_spans.append((m.start(), m.end(), "heading", False, len(m.group(1))))

    for m in HTML_TABLE_PATTERN.finditer(text):
        atomic_spans.append((m.start(), m.end(), "table", True, 0))

    for m in DISPLAY_MATH_PATTERN.finditer(text):
        atomic_spans.append((m.start(), m.end(), "math", True, 0))

    for m in FIGURE_SCHEMA_PATTERN.finditer(text):
        atomic_spans.append((m.start(), m.end(), "figure_schema", True, 0))

    atomic_spans.sort(key=lambda x: (x[0], -x[1]))

    filtered_spans = []
    last_end = 0
    for s_start, s_end, s_type, s_atomic, s_level in atomic_spans:
        if s_start >= last_end:
            filtered_spans.append((s_start, s_end, s_type, s_atomic, s_level))
            last_end = s_end

    blocks: List[Block] = []
    idx = 0

    for s_start, s_end, s_type, s_atomic, s_level in filtered_spans:
        if s_start > idx:
            gap_text = text[idx:s_start]
            p_offset = idx
            for p in re.split(r"(\n\s*\n+)", gap_text):
                if p.strip():
                    p_start = text.find(p, p_offset)
                    p_end = p_start + len(p)
                    blocks.append(Block("prose", p, p_start, p_end, is_atomic=False))
                    p_offset = p_end
                else:
                    p_offset += len(p)

        block_text = text[s_start:s_end]
        if block_text.strip():
            blocks.append(Block(s_type, block_text, s_start, s_end, is_atomic=s_atomic, heading_level=s_level))
        idx = s_end

    if idx < n:
        gap_text = text[idx:n]
        p_offset = idx
        for p in re.split(r"(\n\s*\n+)", gap_text):
            if p.strip():
                p_start = text.find(p, p_offset)
                p_end = p_start + len(p)
                blocks.append(Block("prose", p, p_start, p_end, is_atomic=False))
                p_offset = p_end
            else:
                p_offset += len(p)

    return blocks


def bounded_ast_split_passages(
    text: str,
    chunk_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    overlap: int = DEFAULT_PROSE_OVERLAP,
    max_chunks: int = 3000,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    max_atomic_size: int = DEFAULT_MAX_ATOMIC_SIZE,
    failsafe_size: int = DEFAULT_FAILSAFE_SIZE,
) -> List[Tuple[str, int, int]]:
    """Split *text* using Bounded AST-Aware rules."""
    text = (text or "").strip()
    if not text:
        return []

    blocks = tokenize_document_blocks(text)
    if not blocks:
        return []

    passages: List[Tuple[str, int, int]] = []
    curr_blocks: List[Block] = []
    curr_len = 0

    def flush_chunk(force_overlap: bool = False):
        nonlocal curr_blocks, curr_len
        if not curr_blocks or len(passages) >= max_chunks:
            curr_blocks = []
            curr_len = 0
            return

        start_idx = curr_blocks[0].start
        end_idx = curr_blocks[-1].end
        # [ast chunker fix] join the ACTUAL block texts instead of slicing the raw
        # document range: a block split via Rule 2/3/4 between the first and last
        # accumulated block would otherwise be duplicated into the flush chunk
        # (47/101 items affected; see 02_Memories/Chunking-Bug.md). Keeps chunk
        # text == the concatenation of the blocks it claims to contain.
        chunk_str = "\n".join(b.text.strip() for b in curr_blocks).strip()

        if chunk_str:
            passages.append((chunk_str, start_idx, end_idx))

        if force_overlap and overlap > 0 and len(curr_blocks) > 1:
            overlap_blocks = []
            overlap_accum = 0
            for b in reversed(curr_blocks):
                if not b.is_atomic and b.block_type == "prose" and (overlap_accum + b.length <= overlap):
                    overlap_blocks.insert(0, b)
                    overlap_accum += b.length
                else:
                    break
            curr_blocks = overlap_blocks
            curr_len = overlap_accum
        else:
            curr_blocks = []
            curr_len = 0

    for block in blocks:
        if len(passages) >= max_chunks:
            break

        # Rule 1: Heading boundary fence (#, ##, ###)
        if block.block_type == "heading" and block.heading_level in (1, 2, 3):
            if curr_len >= min_chunk_size:
                flush_chunk(force_overlap=False)
            curr_blocks.append(block)
            curr_len += block.length
            continue

        # Rule 2: HTML Tables
        if block.block_type == "table":
            if block.length <= max_atomic_size:
                if (curr_len + block.length > chunk_size and curr_len >= min_chunk_size) or curr_len + block.length > failsafe_size:
                    flush_chunk(force_overlap=False)
                curr_blocks.append(block)
                curr_len += block.length
                if curr_len >= chunk_size:
                    flush_chunk(force_overlap=False)
            else:
                if curr_len:
                    flush_chunk(force_overlap=False)
                for t_chunk, t_rel_start, t_rel_end in split_oversized_table(block.text, chunk_size):
                    if len(passages) < max_chunks:
                        passages.append((t_chunk, block.start + t_rel_start, block.start + t_rel_end))
            continue

        # Rule 3: Math and Figure Schemas (Display math, [Figure Schema])
        if block.is_atomic:
            if block.length <= max_atomic_size:
                if (curr_len + block.length > chunk_size and curr_len >= min_chunk_size) or curr_len + block.length > failsafe_size:
                    flush_chunk(force_overlap=False)
                curr_blocks.append(block)
                curr_len += block.length
                if curr_len >= chunk_size:
                    flush_chunk(force_overlap=False)
            else:
                if curr_len:
                    flush_chunk(force_overlap=False)
                sub_chunks = split_prose_by_sentences(block.text, chunk_size, overlap)
                for s_text, s_rel_start, s_rel_end in sub_chunks:
                    if len(passages) < max_chunks:
                        passages.append((s_text, block.start + s_rel_start, block.start + s_rel_end))
            continue

        # Rule 4: Standard Prose Block
        if block.length > chunk_size:
            if curr_len:
                flush_chunk(force_overlap=False)
            sub_chunks = split_prose_by_sentences(block.text, chunk_size, overlap)
            for s_text, s_rel_start, s_rel_end in sub_chunks:
                if len(passages) < max_chunks:
                    passages.append((s_text, block.start + s_rel_start, block.start + s_rel_end))
            continue

        # Standard paragraph packing
        if (curr_len + block.length > chunk_size and curr_len >= min_chunk_size) or curr_len + block.length > failsafe_size:
            flush_chunk(force_overlap=True)

        curr_blocks.append(block)
        curr_len += block.length

    if curr_blocks:
        flush_chunk(force_overlap=False)

    return passages
