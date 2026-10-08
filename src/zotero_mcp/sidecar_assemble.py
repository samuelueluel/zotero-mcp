"""Assemble Surya batch output into blocks.json, reliability.json and a sidecar.

Input: a batch directory written by :func:`surya_runner.run_batch`.
Per item output (in ``sidecar_dir``):

- ``<KEY>.blocks.json``      canonical per-block record (page, PDF bbox, label,
                             html, status, findings, repairs)
- ``<KEY>.reliability.json`` paper-level summary for retrieval warnings
- ``<KEY>.md``               Markdown sidecar in the existing contract
                             (headings, HTML tables, ``$$`` math) plus
                             ``<!-- pdf-page: N -->`` anchors and visible
                             status lines above every non-verified table.
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import pymupdf

from .sidecar_math import (
    check_equation,
    check_inline_math,
    html_marks,
    layer_math,
    marks_findings,
    repair_confusables,
)
from .sidecar_native import (
    glyph_profile,
    image_bbox_to_pdf,
    invisible_text_fraction,
    native_number_tokens,
    page_lines,
)
from .sidecar_verify import (
    INFORMATIONAL,
    STRUCTURAL,
    numeric_key,
    text_numbers,
    verify_table,
    withhold_numbers,
)

ASSEMBLER_VERSION = "sidecar-assemble/14"
SKIP_LABELS = {"PageHeader", "PageFooter"}
FIGURE_LABELS = {"Picture", "Figure", "Diagram", "ChemicalBlock"}
NUMERIC_STATUSES = ("verified", "repaired", "single-route", "unresolved")
#: Blocks whose numbers belong to them, not to a table they overlap.
OWNER_LABELS = {"Caption", "Footnote", "Text", "SectionHeader", "ListGroup", "PageHeader", "PageFooter"}
#: Share of invisible text-layer characters that marks a region's text as OCR.
OCR_LAYER_SHARE = 0.5


class PageText:
    """One page's text layer, extracted once and shared by every block check."""

    def __init__(self, page, page_no: int, profile):
        self.page, self.page_no, self.profile = page, page_no, profile
        self._lines = self._trace = None

    @property
    def lines(self):
        if self._lines is None:
            self._lines = page_lines(self.page)
        return self._lines

    def tokens(self, clip):
        return native_number_tokens(self.page, self.page_no, clip=clip, profile=self.profile, lines=self.lines)

    def words(self, clip) -> list[str]:
        """Text-layer text inside ``clip``, one string per line."""
        return [text for text, _bbox in self.text_lines(clip)]

    def text_lines(self, clip) -> list[tuple[str, tuple[float, float, float, float]]]:
        """``(text, bbox)`` per text-layer line inside ``clip`` (displayed-page coordinates)."""
        out = []
        for _vertical, chars in self.lines:
            inside = [c for c in chars
                      if clip[0] <= (c[2][0] + c[2][2]) / 2 <= clip[2] and clip[1] <= (c[2][1] + c[2][3]) / 2 <= clip[3]]
            if inside:
                out.append(("".join(c[0] for c in inside),
                            (min(c[2][0] for c in inside), min(c[2][1] for c in inside),
                             max(c[2][2] for c in inside), max(c[2][3] for c in inside))))
        return out

    def word_boxes(self, clip) -> list[tuple[str, tuple[float, float, float, float], bool]]:
        """Words of the text-layer lines inside ``clip``: ``(word, displayed-page box, vertical)``."""
        out = []
        for vertical, chars in self.lines:
            word: list = []
            for ch in [*chars, (" ", "", (0, 0, 0, 0))]:
                if not ch[0].isspace():
                    word.append(ch)
                    continue
                if word:
                    box = (min(c[2][0] for c in word), min(c[2][1] for c in word),
                           max(c[2][2] for c in word), max(c[2][3] for c in word))
                    if clip[0] <= (box[0] + box[2]) / 2 <= clip[2] and clip[1] <= (box[1] + box[3]) / 2 <= clip[3]:
                        out.append(("".join(c[0] for c in word), box, bool(vertical)))
                word = []
        return out

    def ocr_layer(self, clip) -> bool:
        """True when the text under ``clip`` is mostly invisible OCR text over an image."""
        if self._trace is None:
            self._trace = self.page.get_texttrace()
        return invisible_text_fraction(self.page, clip, trace=self._trace) >= OCR_LAYER_SHARE


def owner_keys(blocks: list[dict]) -> list[tuple[list[float], set[str]]]:
    """``(pdf_bbox, number keys)`` for the blocks on one page that can own numbers."""
    return [
        (b["pdf_bbox"], {numeric_key(n.lstrip("-")) for n in text_numbers(b.get("html", ""))})
        for b in blocks if b["label"] in OWNER_LABELS
    ]


def table_native(text: PageText, pdf_bbox, owners, scan_like: bool):
    """Text-layer tokens for a table region and whether they are an OCR layer.

    A token that another sidecar block over the same spot also contains (a
    caption or note Surya emitted separately) is flagged ``owned``.
    """
    tokens = text.tokens(pdf_bbox)
    for t in tokens:
        cx, cy = (t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2
        key = numeric_key(t.magnitude)
        t.owned = any(x0 <= cx <= x1 and y0 <= cy <= y1 and key in keys
                      for (x0, y0, x1, y1), keys in owners)
    return tokens, scan_like or text.ocr_layer(pdf_bbox)


#: Blocks this close to a table (points) can hold its title and notes.
NEIGHBOUR_MARGIN = 30
_NOTE_START = re.compile(r"^\s*(?:<[^>]+>\s*)*(?:notes?|sources?)\b", re.I)


def neighbour_html(pdf_bbox, blocks) -> list[str]:
    """HTML of the blocks next to a table that can hold its own text: caption,
    title, footnotes, and text blocks that open as a note.

    ``blocks`` are ``(html, pdf_bbox, label)`` triples for the page's other
    blocks. Body paragraphs are left out: they discuss the table's labels and
    would hide labels the table lost.
    """
    m = NEIGHBOUR_MARGIN
    return [h for h, b, label in blocks
            if (label in ("Caption", "Footnote", "SectionHeader") or (label == "Text" and _NOTE_START.match(h)))
            and b[0] - m <= pdf_bbox[2] and pdf_bbox[0] <= b[2] + m and b[1] - m <= pdf_bbox[3] and pdf_bbox[1] <= b[3] + m]

_MATH_BLOCK = re.compile(r'<math\b[^>]*display="block"[^>]*>(.*?)</math>', re.S | re.I)
_MATH_INLINE = re.compile(r"<math\b[^>]*>(.*?)</math>", re.S | re.I)
_BR = re.compile(r"<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")


def html_to_markdown(fragment: str) -> str:
    # Math is unescaped only after the tag strip: an unescaped ``p &lt; 0.001``
    # leaves a bare ``<`` that _TAG would read as a tag opening and delete
    # everything up to the next ``>``, often the rest of the paragraph.
    math: list[str] = []

    def hold(latex: str) -> str:
        math.append(latex)
        return f"\x00{len(math) - 1}\x00"

    text = _MATH_BLOCK.sub(lambda m: "\n" + hold("$$\n" + _html.unescape(m.group(1)).strip() + "\n$$") + "\n", fragment)
    text = _MATH_INLINE.sub(lambda m: hold("$" + _html.unescape(m.group(1)).strip() + "$"), text)
    text = re.sub(r"<sup>(.*?)</sup>", r"^\1", text, flags=re.S | re.I)
    text = re.sub(r"<(b|strong)>(.*?)</\1>", r"**\2**", text, flags=re.S | re.I)
    text = re.sub(r"<(i|em)>(.*?)</\1>", r"*\2*", text, flags=re.S | re.I)
    text = re.sub(r"<li\b[^>]*>", "\n- ", text, flags=re.I)
    text = _BR.sub("\n", text)
    text = _html.unescape(_TAG.sub("", text))
    text = re.sub(r"\x00(\d+)\x00", lambda m: math[int(m.group(1))], text)
    # Leading spaces would turn lines into Markdown code blocks.
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _status_line(block: dict) -> str | None:
    status = block["status"]
    if status == "verified" or block["label"] != "Table":
        return None
    page = block["page"]
    reasons = ", ".join(sorted({f["kind"] for f in block["findings"]
                                if f["kind"] not in INFORMATIONAL | STRUCTURAL})) or "no independent check"
    if status == "unresolved":
        return f"[Table status: UNRESOLVED ({reasons}); unverified numbers withheld. Verify on PDF p. {page}.]"
    if status == "repaired":
        routes = [r.get("route", "") for r in block["repairs"]]
        cells, signs = routes.count("native_cell"), routes.count("native_font_validated")
        stars = routes.count("native_stars")
        how = "re-read by a second model and matched to the PDF text layer" if "vlm_table_reread" in routes \
            else "; ".join(x for x in (
                f"{cells} cell(s) corrected or filled from the PDF text layer" if cells else "",
                f"{signs} sign fix(es) from the PDF text layer" if signs else "",
                f"{stars} significance star(s) corrected from the PDF text layer" if stars else "") if x)
        return f"[Table status: REPAIRED ({how}). Check PDF p. {page} before quoting.]"
    kinds = {f["kind"] for f in block["findings"]}
    if "math_marks_missing" in kinds:
        what = "; ".join(next(f["values"] for f in block["findings"] if f["kind"] == "math_marks_missing"))
        return (f"[Table status: SINGLE-ROUTE (primes, tildes, accents or relations (< ≤ ±) on the page are missing or read differently here: {what}); "
                f"check names and labels on PDF p. {page} before quoting.]")
    if "vlm_agreement" in kinds and "vlm_rewrite" not in kinds and "label_text_lost" not in kinds:
        return (f"[Table status: SINGLE-ROUTE (no PDF text layer; Surya and a second model agree on every number, "
                f"nothing else confirms them). Check PDF p. {page} before quoting decisive numbers.]")
    if kinds & {"vlm_rewrite", "label_text_lost", "header_structure", "stars_unconfirmed"}:
        lost = next((f["values"] for f in block["findings"] if f["kind"] == "label_text_lost"), [])
        why = ("re-read by a second model" if "vlm_rewrite" in kinds else
               "printed labels missing: " + ", ".join(lost[:6]) if lost else
               "column headers do not line up with the page" if "header_structure" in kinds else
               "significance stars on some values are not in the PDF text layer")
        return (f"[Table status: SINGLE-ROUTE ({why}); numbers match the PDF text layer, but row and column "
                f"labels are unchecked. Check PDF p. {page} before quoting.]")
    return f"[Table status: SINGLE-ROUTE ({reasons}); numbers not independently confirmed. Check PDF p. {page} before quoting.]"


#: Single-character OCR confusions the text layer may settle (unordered pairs).
CONFUSABLE = {frozenset(p) for p in (("I", "J"), ("I", "l"), ("l", "1"), ("O", "0"), ("S", "5"), ("B", "8"))}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_SEGMENT = re.compile(r"(<[^>]+>)")


def _one_confusable_apart(a: str, b: str) -> bool:
    if len(a) != len(b) or a == b or len(a) < 3:
        return False
    # Never change a short token's character class ("B" -> "8" in "Figure 8B").
    if sum(c.isalpha() for c in a) < 2 or sum(c.isalpha() for c in b) < 2:
        return False
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diffs) == 1 and frozenset(diffs[0]) in CONFUSABLE


def repair_words(fragment: str, page, clip) -> tuple[str, list[dict]]:
    """Fix Surya word misreads (``IPE`` for ``JPE``) from a born-digital text layer.

    A word is replaced only when it is absent from the native words inside
    the block and exactly one native word differs from it by a single
    confusable character. Numbers are never touched here; math and tags are
    skipped.
    """
    # Map dot-free form -> native surface form, so "J.P.E." matches "IPE".
    native: dict[str, str] = {}
    rect = pymupdf.Rect(clip)
    if page.rotation:
        # Text extraction clips in unrotated space; the block box is on the displayed page.
        rect = rect * page.derotation_matrix
        rect.normalize()
    for w in page.get_text("words", clip=rect):
        surface = w[4].strip(",;:()[]\"'*\u201c\u201d").rstrip(".")
        key = surface.replace(".", "")
        if key:
            native.setdefault(key, surface)
    if not native:
        return fragment, []
    repairs: list[dict] = []
    out = []
    in_math = False
    for seg in _SEGMENT.split(fragment):
        if seg.startswith("<"):
            low = seg.lower()
            if low.startswith("<math"):
                in_math = True
            elif low.startswith("</math"):
                in_math = False
            out.append(seg)
            continue
        if in_math:
            out.append(seg)
            continue

        def fix(m: re.Match) -> str:
            word = m.group(0)
            if word in native or not any(c.isalpha() for c in word):
                return word
            cands = [k for k in native if _one_confusable_apart(word, k)]
            if len(cands) != 1:
                return word
            after = native[cands[0]]
            repairs.append({"kind": "word", "before": word, "after": after, "route": "native_text_layer"})
            return after

        out.append(_WORD.sub(fix, seg))
    return "".join(out), repairs


#: A token with inner dashes (``ln_populat-n``) that may stand for a tilde.
_DASHED = re.compile(r"[\w.#]+(?:[-\u2013][\w.#]+)+")
_TILDE_RUN = re.compile(r"^~{3,}$")
_DASH_RUN = re.compile(r"(?<![\w-])-{3,}(?![\w-])")


def restore_marks(fragment: str, page, clip) -> tuple[str, list[dict]]:
    """Put back tildes Surya read as dashes, from a born-digital text layer.

    Stata abbreviates long names with a tilde (``ln_populat~n``) and marks
    dynamic-document code with ``~~~~`` fences; Surya writes ``-`` and
    ``-----``. A dashed word is replaced only when it is absent from the
    native words inside the block and the native words hold exactly that word
    with tildes for the dashes. Dash runs become the layer's tilde fences only
    when the layer has as many fence lines and no dash-only line. Math and
    tags are skipped.
    """
    rect = pymupdf.Rect(clip)
    if page.rotation:
        rect = rect * page.derotation_matrix
        rect.normalize()
    words = page.get_text("words", clip=rect)
    if not any("~" in w[4] for w in words):
        return fragment, []
    native = {w[4].strip(",;:()[]\"'*\u201c\u201d").rstrip(".") for w in words}
    tilde = {}
    for w in native:
        if "~" in w and w.strip("~"):
            tilde.setdefault(w.replace("~", "-"), set()).add(w)
    lines = [ln.strip() for ln in page.get_text("text", clip=rect).splitlines()]
    fences = [ln for ln in lines if _TILDE_RUN.match(ln)]
    dash_lines = any(re.fullmatch(r"-{3,}", ln) for ln in lines)
    repairs: list[dict] = []
    parts = _SEGMENT.split(fragment)
    in_math = False
    text_parts = []
    for i, seg in enumerate(parts):
        if seg.startswith("<"):
            low = seg.lower()
            in_math = True if low.startswith("<math") else False if low.startswith("</math") else in_math
        elif not in_math:
            text_parts.append(i)

    def fix(m: re.Match) -> str:
        word = m.group(0)
        cands = tilde.get(word.replace("\u2013", "-"), set())
        if word in native or len(cands) != 1:
            return word
        after = next(iter(cands))
        repairs.append({"kind": "mark", "before": word, "after": after, "route": "native_text_layer"})
        return after

    for i in text_parts:
        parts[i] = _DASHED.sub(fix, parts[i])
    runs = sum(len(_DASH_RUN.findall(parts[i])) for i in text_parts)
    if fences and not dash_lines and runs == len(fences):
        queue = iter(fences)

        def fence(m: re.Match) -> str:
            after = next(queue)
            repairs.append({"kind": "mark", "before": m.group(0), "after": after, "route": "native_text_layer"})
            return after
        for i in text_parts:
            parts[i] = _DASH_RUN.sub(fence, parts[i])
    return "".join(parts), repairs


_MARK_CHARS = re.compile("[\u2032\u2033\u2034~\u223c\u02dc\u0303\u0302\u02c6\u0304\u00af\u02c9<>\u2264\u2265\u2a7d\u2a7e\u00b1\u2213]")


def _missing_marks(fragment: str, text: PageText, page, pdf_bbox, pad: float = 0.0) -> list[dict]:
    """``math_marks_missing`` when the text layer prints primes, tildes or accents the block lacks."""
    if not _MARK_CHARS.search("".join(text.words(pdf_bbox))):
        return []
    found = layer_math(page, pdf_bbox, pad=pad)
    if found.untrusted:
        return []
    return marks_findings(html_marks(fragment), found.marks)


def _check_text_numbers(fragment: str, tokens) -> tuple[str, list[dict]]:
    nums = text_numbers(fragment)
    if not nums:
        return "no-numbers", []
    # In prose "170, 182" is two numbers even when set with a thin space.
    pool = Counter(part for t in tokens
                   for part in (t.text.split(",") if t.spaced_group else [t.text]) if part)
    missing = []
    for n in nums:
        if pool[n] > 0:
            pool[n] -= 1
            continue
        # A dotted run broken across lines ("0199245282.001." / "0001" in a DOI).
        parts = n.split(".")
        for i in range(1, len(parts) if len(parts) > 2 else 0):
            head, tail = ".".join(parts[:i]), ".".join(parts[i:])
            if pool[head] > 0 and pool[tail] > 0:
                pool[head] -= 1
                pool[tail] -= 1
                break
        else:
            missing.append(n)
    if not missing:
        return "verified", []
    return "single-route", [{"kind": "text_numbers_unmatched", "detail": "", "values": missing[:20]}]


FIGURE_ZOOM = 200 / 72  # crops for Qwen enrichment; clears its 300 px size filter


def save_figure_crop(page, pdf_bbox, out_dir: Path, item_key: str, name: str) -> str | None:
    """Render a figure region from the PDF itself; return the sidecar image ref."""
    rect = pymupdf.Rect(pdf_bbox) & page.rect
    if rect.is_empty or rect.width < 20 or rect.height < 20:
        return None
    img_dir = out_dir / f"{item_key}.images"
    img_dir.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(FIGURE_ZOOM, FIGURE_ZOOM), clip=rect)
    pix.save(img_dir / name)
    return f"images/{name}"


#: Share of dark pixels below which a scanned page is blank. Measured on a
#: 1000 px render with pixels darker than 200 of 255: blank scans with dust
#: specks print 0.0003 to 0.05 percent, real sparse pages (a dedication, an
#: index) 0.11 percent and up. Li-Racine p. 435 prints 0.0005 percent, and
#: Surya invented a table on it.
BLANK_INK = 0.0003
_DARK = bytes(range(200))


def page_ink(page) -> float:
    """Share of dark pixels on a grayscale render of the page."""
    zoom = 1000 / max(page.rect.width, page.rect.height)
    s = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY).samples
    return (len(s) - len(s.translate(None, _DARK))) / len(s)


def guard_blank_pages(blocks: list[dict], blank: set[int]) -> None:
    """Withhold what OCR read on a blank scanned page: nothing printed there."""
    for b in blocks:
        if b["page"] in blank and b["label"] not in SKIP_LABELS and re.sub(r"<[^>]+>", "", b["html"]).strip():
            b["surya_html"], b["html"] = b["html"], ""
            b["status"] = "unresolved"
            b["findings"] = [{"kind": "blank_page_content",
                              "detail": "the scanned page is blank, but OCR returned content", "values": []}]


def assemble_item(item_key: str, meta: dict, results: dict, out_dir: Path, batch_dir: Path) -> dict:
    doc = pymupdf.open(meta["pdf_path"])
    profile = glyph_profile(doc)
    blocks: list[dict] = []
    blank: set[int] = set()
    for stem in sorted(meta["pages"], key=lambda s: meta["pages"][s]["page"]):
        pm = meta["pages"][stem]
        page_no = pm["page"]
        page = doc[page_no - 1]
        if pm["scan_like"] and page_ink(page) < BLANK_INK:
            blank.add(page_no)
        text = PageText(page, page_no, profile)
        pres = (results.get(stem) or [{}])[0]
        page_recs = []
        for order, b in enumerate(pres.get("blocks", [])):
            pdf_bbox = image_bbox_to_pdf(b["bbox"], pm["image_size"], pm["page_size"], rotation=pm["rotation"])
            page_recs.append(({
                "id": f"{item_key}:p{page_no}:b{order}",
                "page": page_no,
                "order": order,
                "label": b.get("label", "Text"),
                "pdf_bbox": [round(x, 2) for x in pdf_bbox],
                "rotation": pm["rotation"],
                "scan_like": pm["scan_like"],
                # Surya can emit NUL for a glyph it cannot read ("Druță" ->
                # "Drut\x00a"); a NUL breaks Chroma's full-text index.
                "html": b.get("html", "").replace("\x00", "\ufffd"),
                "status": "unchecked",
                "findings": [],
                "repairs": [],
                "surya_error": bool(b.get("error")),
            }, b, pdf_bbox))
        owners = owner_keys([rec for rec, _b, _bb in page_recs])
        for rec, b, pdf_bbox in page_recs:
            label, order = rec["label"], rec["order"]
            if label in FIGURE_LABELS:
                rec["status"] = "not-text"
                rec["image"] = save_figure_crop(page, pdf_bbox, out_dir, item_key, f"p{page_no:03d}_b{order}.png")
            elif label in SKIP_LABELS or b.get("skipped"):
                rec["status"] = "not-text"
            elif label == "Table":
                tokens, ocr = table_native(text, pdf_bbox, owners, pm["scan_like"])
                marks: list[dict] = []
                if not ocr:
                    restored, marks = restore_marks(rec["html"], page, pdf_bbox)
                    if marks:
                        rec["surya_html"], rec["html"] = rec["html"], restored
                lines = None if ocr else text.text_lines(pdf_bbox)
                check = verify_table(
                    rec["html"], tokens, native_is_ocr=ocr,
                    printed_words=None if lines is None else [t for t, _b in lines],
                    context_html=neighbour_html(
                        pdf_bbox, [(r["html"], bb, r["label"]) for r, _b, bb in page_recs if r is not rec]),
                    printed_lines=lines,
                    printed_word_boxes=None if ocr else text.word_boxes(pdf_bbox))
                if any(r.get("route") in ("native_cell", "native_stars") for r in check.repairs):
                    rec.setdefault("surya_html", rec["html"])
                rec["status"] = check.status
                rec["html"] = check.html
                rec["findings"] = [{"kind": f.kind, "detail": f.detail, "values": f.values[:40]} for f in check.findings]
                rec["repairs"] = marks + check.repairs
                if not ocr and (missing := _missing_marks(rec["html"], text, page, pdf_bbox)):
                    rec["findings"] += missing
                    if rec["status"] == "verified":
                        rec["status"] = "single-route"
                rec["counts"] = {"surya": check.surya_numbers, "native": check.native_numbers}
                if ocr and not pm["scan_like"]:
                    rec["ocr_layer"] = True
            elif label == "Equation":
                ocr = pm["scan_like"] or text.ocr_layer(pdf_bbox)
                math = None if ocr else check_equation(rec["html"], page, pdf_bbox)
                if math is None or math.status == "no-layer":
                    rec["status"] = "single-route"
                    rec["findings"] = [{"kind": "math_single_route",
                                        "detail": "no PDF text layer to check the LaTeX against", "values": []}]
                elif math.status == "agree":
                    # Same symbols, but fractions, sub- and superscripts are unchecked.
                    rec["status"] = "single-route"
                    rec["findings"] = [{"kind": "math_symbols_agree",
                                        "detail": "symbols match the PDF text layer; structure unchecked", "values": []}]
                else:
                    rec["status"] = "unresolved"
                    rec["findings"] = math.findings
            else:
                ocr = pm["scan_like"] or text.ocr_layer(pdf_bbox)
                if not ocr:
                    rec["html"], rec["repairs"] = repair_words(rec["html"], page, pdf_bbox)
                    rec["html"], marks = restore_marks(rec["html"], page, pdf_bbox)
                    rec["repairs"] += marks
                status, findings = _check_text_numbers(rec["html"], text.tokens(pdf_bbox))
                if ocr and status == "verified":
                    status = "single-route"
                if not ocr and "<math" in rec["html"]:
                    math = check_inline_math(rec["html"], page, pdf_bbox)
                    if math.status != "no-layer":
                        rec["_math_checked"] = status
                    if math.status == "differ":
                        findings = findings + math.findings
                        if status in ("verified", "no-numbers"):
                            status = "single-route"
                elif not ocr and (missing := _missing_marks(rec["html"], text, page, pdf_bbox, pad=4.0)):
                    findings = findings + missing
                    if status in ("verified", "no-numbers"):
                        status = "single-route"
                rec["status"], rec["findings"] = status, findings
            if rec["surya_error"]:
                rec["status"] = "unresolved"
                rec["findings"].append({"kind": "surya_block_error", "detail": "", "values": []})
            blocks.append(rec)

    gate_math(blocks)
    repair_math(blocks, doc)
    guard_blank_pages(blocks, blank)
    provenance = {
        "assembler": ASSEMBLER_VERSION,
        "batch_dir": str(batch_dir),
        "pdf_path": meta["pdf_path"],
        "pdf_sha256": meta["pdf_sha256"],
        "attachment_key": meta.get("attachment_key"),
        "assembled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return write_outputs(item_key, meta, blocks, provenance, Path(out_dir))


#: Share of checked math that may disagree with the text layer before the
#: layer, not Surya, is the likelier culprit (glyphs without Unicode meaning).
MATH_GATE_SHARE = 0.25
_MATH_DIFFER = ("math_symbols_differ", "equation_number_differs")


def gate_math(blocks: list[dict]) -> None:
    """Drop math disagreements in a document whose text layer does not encode its math.

    Born-digital TeX books disagree on 2 to 4 percent of equations; papers set
    with fonts that map Greek to Latin letters or brackets to private code
    points disagree on nearly all. There a disagreement is not evidence, so
    those equations go back to unchecked single-route and prose keeps its
    number-check status.
    """
    checked = [b for b in blocks if b["label"] == "Equation"
               and any(f["kind"] in _MATH_DIFFER + ("math_symbols_agree",) for f in b["findings"])]
    differ = [b for b in checked if any(f["kind"] in _MATH_DIFFER for f in b["findings"])]
    if len(checked) >= 8 and len(differ) > MATH_GATE_SHARE * len(checked):
        detail = (f"PDF text layer does not encode this document's math reliably "
                  f"({len(differ)} of {len(checked)} equations differ)")
        for b in differ:
            b["findings"] = [f for f in b["findings"] if f["kind"] not in _MATH_DIFFER]
            b["findings"].insert(0, {"kind": "math_single_route", "detail": detail, "values": []})
            if not b.get("surya_error") and not _has_marks(b):
                b["status"] = "single-route"
    prose = [b for b in blocks if "_math_checked" in b]
    flagged = [b for b in prose if any(f["kind"] == "inline_math_symbols_differ" for f in b["findings"])]
    if len(prose) >= 10 and len(flagged) > MATH_GATE_SHARE * len(prose):
        for b in flagged:
            b["findings"] = [f for f in b["findings"] if f["kind"] != "inline_math_symbols_differ"]
            if not b.get("surya_error") and not _has_marks(b):
                b["status"] = b["_math_checked"]


def _has_marks(block: dict) -> bool:
    return any(f["kind"] == "math_marks_missing" for f in block["findings"])


def repair_math(blocks: list[dict], doc) -> None:
    """Swap look-alike symbols (v for nu) back from the text layer, then re-check.

    Runs after ``gate_math``, so a document whose text layer does not encode
    its math is never repaired from it. A repaired equation or paragraph
    agrees with the text layer in full; its structure is still unchecked.
    """
    for b in blocks:
        kinds = {f["kind"] for f in b["findings"]}
        if b.get("surya_error"):
            pass
        elif b["label"] == "Equation" and kinds == {"math_symbols_differ"}:
            page, bbox = doc[b["page"] - 1], b["pdf_bbox"]
            fixed = repair_confusables(b["html"], check_equation(b["html"], page, bbox),
                                       lambda h, page=page, bbox=bbox: check_equation(h, page, bbox))
            if fixed:
                b["surya_html"] = b["html"]
                b["html"], repairs = fixed
                b["repairs"] = b["repairs"] + repairs
                b["status"] = "repaired"
                b["findings"] = [{"kind": "math_symbols_agree",
                                  "detail": "symbols match the PDF text layer after the repair; structure unchecked",
                                  "values": []}]
        elif "_math_checked" in b and "inline_math_symbols_differ" in kinds:
            page, bbox = doc[b["page"] - 1], b["pdf_bbox"]
            fixed = repair_confusables(b["html"], check_inline_math(b["html"], page, bbox),
                                       lambda h, page=page, bbox=bbox: check_inline_math(h, page, bbox))
            if fixed:
                b["surya_html"] = b["html"]
                b["html"], repairs = fixed
                b["repairs"] = b["repairs"] + repairs
                b["findings"] = [f for f in b["findings"] if f["kind"] != "inline_math_symbols_differ"]
                b["status"] = "repaired" if b["_math_checked"] in ("verified", "no-numbers") else b["_math_checked"]
    for b in blocks:
        b.pop("_math_checked", None)


def atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` in one step, so an interrupted write leaves the old file intact."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_outputs(item_key: str, meta: dict, blocks: list[dict], provenance: dict, out_dir: Path) -> dict:
    """Write the Markdown sidecar, reliability.json and blocks.json.

    Shared by assembly and the repair pass. Overwrites the sidecar, so it must
    run before figure enrichment inserts [Figure Schema] blocks. Each file is
    replaced atomically and blocks.json goes last: after an interrupted repair
    it still records the tables as unresolved, so a resumed run redoes them.
    """
    import hashlib

    reliability = summarize(item_key, meta, blocks)
    markdown = render_markdown(blocks)
    out_dir.mkdir(parents=True, exist_ok=True)
    reliability.update(provenance)
    reliability["sidecar_sha256"] = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    atomic_write_text(out_dir / f"{item_key}.md", markdown)
    atomic_write_text(out_dir / f"{item_key}.reliability.json", json.dumps(reliability, indent=1))
    atomic_write_text(out_dir / f"{item_key}.blocks.json", json.dumps(
        {"item_key": item_key, **provenance, "meta": meta, "blocks": blocks}, indent=1))
    return reliability


def summarize(item_key: str, meta: dict, blocks: list[dict]) -> dict:
    tables = [b for b in blocks if b["label"] == "Table"]
    table_counts = Counter(b["status"] for b in tables)
    problems = [
        {"page": b["page"], "status": b["status"], "block_id": b["id"],
         "reasons": sorted({f["kind"] for f in b["findings"]})}
        for b in tables if b["status"] != "verified"
    ]
    pages = meta["pages"].values()
    scan_pages = sorted(p["page"] for p in pages if p["scan_like"])
    rotated = sorted(p["page"] for p in pages if p["rotation"])
    n_pages = len(meta["pages"])
    equations = [b for b in blocks if b["label"] == "Equation"]
    eq_unresolved = sum(1 for b in equations if b["status"] == "unresolved")
    inline_flags = sum(1 for b in blocks if b["label"] != "Equation" and any(
        f["kind"] in ("inline_math_symbols_differ", "math_marks_missing") for f in b["findings"]))
    if table_counts.get("unresolved"):
        level = "warn"
    elif scan_pages or table_counts.get("single-route") or table_counts.get("repaired") or eq_unresolved:
        level = "caution"
    else:
        level = "ok"
    warning = None
    if level != "ok":
        parts = []
        if table_counts.get("unresolved"):
            parts.append(f"{table_counts['unresolved']} of {len(tables)} tables unresolved (numbers withheld)")
        if table_counts.get("single-route"):
            parts.append(f"{table_counts['single-route']} tables single-route")
        if table_counts.get("repaired"):
            parts.append(f"{table_counts['repaired']} tables sign-repaired")
        if eq_unresolved:
            parts.append(f"{eq_unresolved} of {len(equations)} equations differ from the PDF text layer")
        if scan_pages:
            parts.append(f"{len(scan_pages)} of {n_pages} pages scanned")
        warning = "; ".join(parts) + ". Verify table numbers on the PDF pages listed in problem_tables."
        if eq_unresolved:
            warning += " Flagged equations carry their PDF page in an [Equation status] line."
    return {
        "item_key": item_key,
        "level": level,
        "warning": warning,
        "pages": n_pages,
        "scan_pages": scan_pages,
        "rotated_pages": rotated,
        "tables": dict(table_counts),
        "equations": len(equations),
        "equation_status": dict(Counter(b["status"] for b in equations)),
        "inline_math_flags": inline_flags,
        "problem_tables": problems,
        "missing_pages": meta.get("missing_pages", []),
    }


def _math_status_line(block: dict) -> str | None:
    """Visible marker for math the text layer contradicts or repaired; retrieval reads it."""
    page = block["page"]
    swaps = [r for r in block.get("repairs", []) if r.get("route") == "native_symbol"]
    if swaps:
        what = "; ".join(f"{r['after']} where the sidecar read {r['before']}" for r in swaps)
        kind = "Equation" if block["label"] == "Equation" else "Math"
        return f"[{kind} status: REPAIRED ({what}, from the PDF text layer). Check PDF p. {page} before quoting.]"
    for f in block["findings"]:
        what = "; ".join(f["values"][:4])
        if f["kind"] == "math_symbols_differ" or f["kind"] == "equation_number_differs":
            why = ("equation number differs from the PDF" if f["kind"] == "equation_number_differs"
                   else "symbols differ from the PDF text layer")
            return f"[Equation status: UNRESOLVED ({why}: {what}). Check PDF p. {page} before quoting.]"
        if f["kind"] == "math_marks_missing":
            if block["label"] == "Equation":
                return (f"[Equation status: UNRESOLVED (primes, tildes, accents or relations (< ≤ ±) on the page are missing or read differently here: "
                        f"{what}). Check PDF p. {page} before quoting.]")
            if block["label"] != "Table":
                return (f"[Math status: SINGLE-ROUTE (primes, tildes, accents or relations (< ≤ ±) on the page are missing or read differently here: "
                        f"{what}). Check PDF p. {page} before quoting.]")
        if f["kind"] == "inline_math_symbols_differ":
            return (f"[Math status: SINGLE-ROUTE (inline math or text differs from the PDF text layer: {what}). "
                    f"Check PDF p. {page} before quoting.]")
    return None


def render_markdown(blocks: list[dict]) -> str:
    out: list[str] = []
    current_page = None
    for b in blocks:
        if b["page"] != current_page:
            current_page = b["page"]
            out.append(f"<!-- pdf-page: {current_page} -->")
        label = b["label"]
        if label in SKIP_LABELS:
            continue
        if any(f["kind"] == "blank_page_content" for f in b["findings"]):
            out.append(f"[Page status: UNRESOLVED (the scanned page is blank, but OCR returned text; "
                       f"withheld). Verify on PDF p. {b['page']}.]")
            continue
        if label in FIGURE_LABELS:
            if b.get("image"):
                out.append(f"![Figure, PDF p. {b['page']}]({b['image']})")
            else:
                out.append(f"[Figure on PDF p. {b['page']}; see caption]")
            continue
        if label == "Table":
            line = _status_line(b)
            if line:
                out.append(line)
            html = withhold_numbers(b["html"], b["page"]) if b["status"] == "unresolved" else b["html"]
            out.append(html.strip())
            continue
        text = html_to_markdown(b["html"])
        if not text:
            continue
        line = _math_status_line(b)
        if line:
            out.append(line)
        if label == "SectionHeader":
            out.append("## " + " ".join(text.split()))
        elif label == "Equation" and "$$" not in text:
            out.append("$$\n" + text + "\n$$")
        else:
            out.append(text)
    return "\n\n".join(out).strip() + "\n"


def assemble_batch(batch_dir: Path, out_dir: Path) -> dict[str, dict]:
    batch_dir = Path(batch_dir)
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads(next(iter(sorted((batch_dir / "ocr").glob("*/results.json")))).read_text(encoding="utf-8"))
    return {
        key: assemble_item(key, meta, results, Path(out_dir), batch_dir)
        for key, meta in manifest["items"].items()
    }
