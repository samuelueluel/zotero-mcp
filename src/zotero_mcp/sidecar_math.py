"""Check Surya's LaTeX against a born-digital PDF text layer.

The check compares symbol multisets: Latin letters, digits, Greek letters and
big operators (sum, product, integral, union, intersection, partial, nabla,
infinity). Accents, bold, spacing, brackets and layout are ignored, so a match
says the same symbols are present, not that the structure is right: ``1/k``
and ``k/1`` agree. A mismatch is a real signal (``3/k`` for ``1/k``, ``v``
for ``\\nu``, one integral where the page has two).

Equation numbers are compared on their own: Surya often keeps a margin label
("4.14", "(50)") that the equation box on the page may or may not include.

Primes and tildes are counted on their own (:func:`latex_marks`): a dropped
transpose (``X'`` read as ``X``) or a tilde accent read as a bar changes the
meaning while every symbol above still matches. Only marks the page has and
the sidecar lacks are flagged; a layer often omits them (a prime set as an
unmapped glyph), so the reverse is no evidence.
"""

from __future__ import annotations

import html as html_lib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

_GREEK_NAMES = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma "
                "tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega")
GREEK = {
    n: unicodedata.lookup(("GREEK SMALL LETTER " if n[0].islower() else "GREEK CAPITAL LETTER ")
                          + n.upper().replace("LAMBDA", "LAMDA"))
    for n in _GREEK_NAMES.split()
}
GREEK.update(varepsilon="ε", vartheta="θ", varphi="φ", varrho="ρ", varsigma="σ", varpi="π")
OPERATORS = {"sum": "∑", "prod": "∏", "coprod": "∐", "int": "∫", "iint": "∫∫", "iiint": "∫∫∫", "oint": "∫",
             "infty": "∞", "bigcup": "∪", "bigcap": "∩", "cup": "∪", "cap": "∩", "partial": "∂", "nabla": "∇"}
#: Commands whose letters print (``\log``, ``\max``).
WORD_COMMANDS = frozenset(
    "liminf limsup log ln lg exp max min sup inf lim det sin cos tan sinh cosh tanh cot sec csc arcsin arccos "
    "arctan arg deg dim ker hom gcd Pr".split())
SYMBOLS = frozenset("∑∏∐∫∞∪∩∂∇")
#: One glyph, several code points.
_SAME = {"ϵ": "ε", "ǫ": "ε", "ϑ": "θ", "ϕ": "φ", "ϱ": "ρ", "ς": "σ", "ϖ": "π", "µ": "μ", "ı": "i", "ȷ": "j", "∆": "Δ", "Ω": "Ω",
         # Empty set: some fonts print it as the letter Ø; \emptyset is not compared.
         "Ø": "∅", "ø": "∅"}
#: TeX extension-font glyphs as the text layer reports them. Other characters
#: in these fonts are delimiter, brace and radical pieces.
_EXTENSION = {"P": "∑", "X": "∑", "Q": "∏", "Y": "∏", "R": "∫", "Z": "∫", "S": "∪", "[": "∪", "T": "∩", "\\": "∩",
              "∑": "∑", "∏": "∏", "∫": "∫", "⋃": "∪", "⋂": "∩"}
#: Old-style blackboard bold set as roman I + letter ("IR" for the reals).
_BLACKBOARD = re.compile(r"(?<![A-Za-z])I(?=[RNCZQB](?![A-Z]))")

_LABEL_TEXT = r"\d+(?:\.\d+)*[a-z]?"
_SURYA_TAG = re.compile(r"\\tag\*?\{\s*\(?(" + _LABEL_TEXT + r")\)?\s*\}")
#: "(48)" after alignment or spacing, anywhere (rows of aligned blocks end in "& (9) \\").
_SURYA_PAREN = re.compile(r"(?:&|\\q?quad|\\hfill|\\;|~)\s*(?:\\text\{\s*)?\((" + _LABEL_TEXT + r")\)\s*\}?")
#: Margin numbers written first: "4.14 \quad ..." or "\\ 8.12 & ...", never a
#: matrix entry such as 0.0000.
_MARGIN = r"[1-9]\d*\.\d{1,2}[a-z]?"
_SURYA_LEADING = re.compile(r"(^|\\\\)\s*(" + _MARGIN + r")\s*(?=&|\\q?quad)")
#: A label at the end or start of a text-layer line.
_LAYER_TRAILING = re.compile(r"\((" + _LABEL_TEXT + r")\)\s*$")
_LAYER_LEADING = re.compile(r"^(" + _MARGIN + r")(?=\s|$)")


def _norm(ch: str) -> str:
    return _SAME.get(ch, ch)


def _counts(ch: str) -> bool:
    ch = _norm(ch)
    return (ch.isalnum() and unicodedata.category(ch) != "Lm") or ch in SYMBOLS


def split_labels(latex: str) -> tuple[str, list[str]]:
    """Remove equation numbers from Surya's LaTeX; return (body, labels)."""
    labels: list[str] = []

    def take(m: re.Match) -> str:
        labels.append(m.group(m.lastindex))
        return " "
    latex = _SURYA_TAG.sub(take, latex)
    latex = _SURYA_PAREN.sub(take, latex)

    def lead(m: re.Match) -> str:
        labels.append(m.group(2))
        return m.group(1) + " "
    latex = _SURYA_LEADING.sub(lead, latex.strip())
    return latex, labels


def latex_symbols(latex: str, *, markup: bool = True) -> Counter:
    """Symbols of LaTeX; ``markup=False`` when tags are gone and "<" is a relation."""
    s = re.sub(r"\\begin\{array\}\s*\{[^}]*\}", " ", latex)
    s = re.sub(r"\\(?:begin|end)\{[A-Za-z*]+\}|&amp;|&lt;|&gt;|&nbsp;", " ", s)
    if markup:
        s = re.sub(r"<[^>]+>", " ", s)

    def command(m: re.Match) -> str:
        name = m.group(1)
        if name in GREEK:
            return f" {GREEK[name]} "
        if name in OPERATORS:
            return f" {OPERATORS[name]} "
        if name in ("top", "intercal"):  # transpose, printed as T
            return " T "
        if name in ("Re", "Im"):
            return f" {name[0]} "
        if name in WORD_COMMANDS:
            return f" {name} "
        return " "
    s = re.sub(r"\\([A-Za-z]+)", command, s)
    return Counter(_norm(ch) for ch in unicodedata.normalize("NFKC", s.translate(_ELL)) if _counts(ch))


#: Script l: many math fonts leave it unmapped, so it is not compared at all.
_ELL = str.maketrans("", "", "ℓ")


def prose_symbols(text: str) -> Counter:
    return Counter(_norm(ch) for ch in unicodedata.normalize("NFKC", text.translate(_ELL)) if _counts(ch))


def _unaccent(symbols: Counter) -> Counter:
    """Drop accents; the layer may compose a hat with its letter (î)."""
    out: Counter = Counter()
    for ch, n in symbols.items():
        for c in unicodedata.normalize("NFKD", ch):
            if not unicodedata.combining(c) and _counts(c):
                out[c] += n
    return out


def _fold(symbols: Counter) -> Counter:
    """Prose comparison: drop accents (the layer may set them apart) and case (small caps)."""
    out: Counter = Counter()
    for ch, n in symbols.items():
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        for c in base.lower():
            if _counts(c):
                out[c] += n
    return out


def _list_markers(html: str) -> str:
    """Write out the markers an ordered list prints ("(a)", "5") but the HTML implies."""
    def number(m: re.Match) -> str:
        first = re.search(r"<li\b[^>]*>\s*(?:<[^>]+>\s*)*([^<]{0,6})", m.group(0))
        if "list-style-type: none" in m.group(0)[:200] or (first and re.match(r"\(?\w{1,3}[.)]", first.group(1))):
            return m.group(0)  # the markers are in the item text already
        kind, start = m.group(1) or "1", int(m.group(2) or 1)
        items = len(re.findall(r"<li\b", m.group(0)))
        marks = [chr(ord("a") + start - 1 + i) if kind in "aA" else str(start + i) for i in range(items)]
        return " ".join(marks) + " " + m.group(0)
    return re.sub(r'<ol\b(?=[^>]*)(?:[^>]*type="([1aA])")?(?:[^>]*start="(\d+)")?[^>]*>.*?</ol>', number, html, flags=re.S)


def html_symbols(html: str) -> Counter:
    """Symbols of a prose block: its text plus the symbols of its inline math."""
    html = _list_markers(html)
    out: Counter = Counter()
    for part in re.split(r"(<math[^>]*>.*?</math>)", html, flags=re.S):
        if part.startswith("<math"):
            out += latex_symbols(part)
        else:
            text = re.sub(r"<[^>]+>", " ", part)
            out += prose_symbols(text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
    return out


#: Code points that stand in for glyphs a font never mapped to Unicode: Elsevier
#: delimiters (ð Þ þ), Word math brackets in Malayalam or Ethiopic, "ǁ" for a norm.
_FAKE = frozenset("ðÞþǁŽÝ")


def _trusted(ch: str) -> bool:
    """Whether a text-layer character can be taken at face value."""
    cat = unicodedata.category(ch)
    if cat in ("Cc", "Co", "Cs", "Cn") or ch == "\ufffd" or ch in _FAKE:
        return False
    folded = unicodedata.normalize("NFKC", _norm(ch))  # ligatures, micro sign
    if len(folded) != 1 or folded != ch:
        return all(_trusted(c) for c in folded)
    if not ch.isalnum() or cat == "Lm":
        return True
    o = ord(ch)
    if o < 0x80 or 0x370 <= o <= 0x3FF or 0x1F00 <= o <= 0x1FFF or 0x2100 <= o <= 0x214F or 0x1D400 <= o <= 0x1D7FF:
        return True
    # Accented Latin (names in prose, hats composed with their letter in math).
    return 0xC0 <= o <= 0x24F or 0x1E00 <= o <= 0x1EFF


#: Prime glyphs and how many primes each prints.
PRIMES = {"′": 1, "″": 2, "‴": 3}
_TEXT_GROUPS = re.compile(r"\\(?:text\w*|mbox|operatorname)\s*\{[^{}]*\}")


def _tilde_count(text: str, composed: bool = False) -> int:
    """Tildes: accents (combining or spacing ˜), the operator ∼ and ASCII ~.

    ``composed`` also counts letters with a tilde (ñ): the sidecar's side,
    since a layer may print "n˜" for the ñ that Surya writes composed. Such
    letters on the layer side are not counted, so names never raise a flag.
    """
    n = sum(1 for c in text if c in "\u0303∼~˜")
    if composed:
        n += sum(1 for c in text if len(d := unicodedata.normalize("NFD", c)) > 1 and "\u0303" in d)
    return n


def latex_marks(latex: str) -> Counter:
    """Primes and tildes of LaTeX. ASCII ``~`` is a space there, not a tilde."""
    s = _TEXT_GROUPS.sub(" ", latex)
    primes = s.count("'") + len(re.findall(r"\\prime(?![A-Za-z])", s)) + sum(PRIMES.get(c, 0) for c in s)
    # \cong and \simeq print a tilde over a bar; text layers often split them ("∼=").
    tildes = len(re.findall(r"\\(?:tilde|widetilde|sim|simeq|cong|approxeq)(?![A-Za-z])", s)) \
        + _tilde_count(s.replace("~", " "), True) + sum(1 for c in s if c in "≅≃")
    return Counter(prime=primes, tilde=tildes) + Counter()


def prose_marks(text: str) -> Counter:
    """Primes and tildes of prose; apostrophes are not primes."""
    return Counter(prime=sum(PRIMES.get(c, 0) for c in text), tilde=_tilde_count(text, True)) + Counter()


def html_marks(html: str) -> Counter:
    """Primes and tildes of a block: its prose plus its inline or display math."""
    out: Counter = Counter()
    for part in re.split(r"(<math[^>]*>.*?</math>)", html, flags=re.S):
        if part.startswith("<math"):
            out += latex_marks(html_lib.unescape(re.sub(r"<[^>]+>", " ", part)))
        else:
            out += prose_marks(html_lib.unescape(re.sub(r"<[^>]+>", " ", part)))
    return out


def _line_marks(chars) -> Counter:
    out: Counter = Counter()
    for font, c in chars:
        ch = c["c"]
        if _extension(font):
            continue
        if "cmsy" in font and ch == "0":  # the prime glyph of TeX's symbol font, unmapped
            out["prime"] += 1
        else:
            out["prime"] += PRIMES.get(ch, 0)
            out["tilde"] += _tilde_count(ch)
    return out + Counter()


def marks_findings(ours: Counter, page: Counter) -> list[dict]:
    """A finding when the page prints primes or tildes the sidecar lacks."""
    missing = page - ours
    if not missing:
        return []
    glyph = {"prime": "′", "tilde": "~"}
    return [{"kind": "math_marks_missing",
             "detail": "the PDF prints primes or tildes the sidecar lacks",
             "values": [f"PDF has {n} more {glyph[k]}" for k, n in sorted(missing.items())]}]


@dataclass
class LayerMath:
    symbols: Counter = field(default_factory=Counter)
    labels: list[str] = field(default_factory=list)
    #: Primes and tildes (:func:`latex_marks`).
    marks: Counter = field(default_factory=Counter)
    #: Characters that are not what they print (unmapped glyphs, fake code points).
    untrusted: int = 0
    #: Extension-font glyphs with no Unicode meaning: delimiter pieces, or a big
    #: operator whose code the font does not map.
    unmapped: int = 0


def layer_math(page, pdf_bbox, *, find_labels: bool = False, pad: float = 0.0) -> LayerMath:
    """Symbols of the text layer whose glyph centres fall inside ``pdf_bbox``."""
    rect = pymupdf.Rect(pdf_bbox) + (-pad, 0, pad, 0)
    if page.rotation:
        rect = rect * page.derotation_matrix
        rect.normalize()
    out = LayerMath()
    width = max(rect.width, 1.0)
    for block in page.get_text("rawdict", clip=rect)["blocks"]:
        for line in block.get("lines", []):
            # A line belongs to the box when most of its glyphs do; otherwise the
            # few inside are subscripts or descenders of a neighbouring line.
            every = [(sp["font"].lower(), c) for sp in line["spans"] for c in sp["chars"]
                     if not c["c"].isspace() or unicodedata.category(c["c"]) == "Cc"]
            chars = [(f, c) for f, c in every
                     if rect.contains(pymupdf.Point((c["bbox"][0] + c["bbox"][2]) / 2,
                                                    (c["bbox"][1] + c["bbox"][3]) / 2))]
            # Fake code points anywhere in the box mean the font's encoding is not Unicode.
            out.untrusted += sum(1 for f, c in chars if not _extension(f) and not _trusted(c["c"]))
            if 2 * len(chars) < len(every):
                continue
            out.unmapped += sum(1 for f, c in chars if _extension(f) and c["c"] not in _EXTENSION
                                and unicodedata.category(c["c"]) == "Cc")
            if not chars:
                continue
            text = "".join(c["c"] for _f, c in chars).strip()
            x0 = min(c["bbox"][0] for _f, c in chars)
            x1 = max(c["bbox"][2] for _f, c in chars)
            if find_labels:
                # Labels sit at the box edges: drop their characters from the count.
                m = _LAYER_TRAILING.search(text)
                if m and x1 > rect.x1 - 0.2 * width:
                    out.labels.append(m.group(1))
                    chars = chars[:len(chars) - _tail_len(chars, m.group(0))]
                m = _LAYER_LEADING.match(text)
                if m and x0 < rect.x0 + 0.2 * width:
                    out.labels.append(m.group(1))
                    chars = chars[_head_len(chars, m.group(0)):]
            out.symbols += _line_symbols(chars)
            out.marks += _line_marks(chars)
    return out


def _tail_len(chars, label: str) -> int:
    """Number of trailing chars that spell ``label`` (spaces ignored)."""
    want, n = label.replace(" ", ""), 0
    got = ""
    for _f, c in reversed(chars):
        n += 1
        if not c["c"].isspace():
            got = c["c"] + got
        if got == want:
            return n
    return 0


def _head_len(chars, label: str) -> int:
    want, n, got = label.replace(" ", ""), 0, ""
    for _f, c in chars:
        n += 1
        if not c["c"].isspace():
            got += c["c"]
        if got == want:
            return n
    return 0


def _extension(font: str) -> bool:
    """TeX extension fonts: big operators plus delimiter pieces with no Unicode meaning."""
    # WWDOC fonts build brackets from letter glyphs (Oxford and Elsevier journals).
    return any(x in font for x in ("cmex", "extension", "symbolext", "esint", "txex", "pxex", "mathex", "wwdoc")) \
        or font.endswith("-ex")


def _line_symbols(chars) -> Counter:
    out: Counter = Counter()
    # Runs of one font, so blackboard "IR" is seen whole.
    runs: list[tuple[str, str]] = []
    for font, c in chars:
        if runs and runs[-1][0] == font:
            runs[-1] = (font, runs[-1][1] + c["c"])
        else:
            runs.append((font, c["c"]))
    for font, text in runs:
        if _extension(font):
            text = "".join(_EXTENSION.get(ch, "") for ch in text)
        elif "cmsy" in font or "msbm" in font or "symbol" in font:
            # Digits in the symbol font are glyph pieces ("7" of \mapsto, "0" of a prime).
            text = "".join(ch for ch in text if not ch.isdigit())
        else:
            text = _BLACKBOARD.sub("", text)
        for ch in unicodedata.normalize("NFKC", text.translate(_ELL)):
            if _counts(ch):
                out[_norm(ch)] += 1
    return out


@dataclass
class MathCheck:
    status: str  # "agree", "differ", "no-layer"
    findings: list[dict] = field(default_factory=list)
    #: Compared symbols (after label, operator and accent handling), for the repair.
    sidecar: Counter = field(default_factory=Counter)
    pdf: Counter = field(default_factory=Counter)


def _diff_values(sidecar: Counter, pdf: Counter) -> list[str]:
    extra = sorted((sidecar - pdf).elements())
    missing = sorted((pdf - sidecar).elements())
    out = []
    if extra:
        out.append("sidecar has: " + " ".join(extra[:20]))
    if missing:
        out.append("PDF has: " + " ".join(missing[:20]))
    return out


def check_equation(html: str, page, pdf_bbox) -> MathCheck:
    ours: Counter = Counter()
    surya_labels: list[str] = []
    for part in re.split(r"(<math[^>]*>.*?</math>)", html, flags=re.S):
        if part.startswith("<math"):
            body, labels = split_labels(html_lib.unescape(re.sub(r"<[^>]+>", " ", part)))
            ours += latex_symbols(body, markup=False)
            surya_labels += labels
        else:
            ours += html_symbols(part)
    our_marks = html_marks(html)
    layer = layer_math(page, pdf_bbox, find_labels=True)
    if layer.untrusted or sum(layer.symbols.values()) < max(3, sum(ours.values()) // 4):
        return MathCheck("no-layer")
    # A label only one side recognised as a label: remove it from the other side too.
    for label in set(layer.labels) - set(surya_labels):
        digits = prose_symbols(label)
        if not digits - ours:
            ours -= digits
    for label in set(surya_labels) - set(layer.labels):
        digits = prose_symbols(label)
        if not digits - layer.symbols:
            layer.symbols -= digits
    # Each unmapped extension glyph may be one of the big operators Surya read.
    spare = layer.unmapped
    for op in "∑∏∫∪∩":
        excess = min(spare, (ours - layer.symbols)[op])
        if excess:
            ours[op] -= excess
            spare -= excess
    ours += Counter()  # drop zero counts
    ours, layer.symbols = _unaccent(ours), _unaccent(layer.symbols)
    findings = []
    if ours != layer.symbols:
        findings.append({"kind": "math_symbols_differ",
                         "detail": "symbols differ from the PDF text layer",
                         "values": _diff_values(ours, layer.symbols)})
    if layer.labels and set(surya_labels) - set(layer.labels):
        findings.append({"kind": "equation_number_differs", "detail": "",
                         "values": [f"sidecar ({', '.join(surya_labels)})", f"PDF ({', '.join(layer.labels)})"]})
    findings += marks_findings(our_marks, layer.marks)
    return MathCheck("differ" if findings else "agree", findings, ours, layer.symbols)


def check_inline_math(html: str, page, pdf_bbox) -> MathCheck:
    """Whole-block symbols of a prose block with inline math against the layer."""
    ours = _fold(html_symbols(html))
    # Surya's prose boxes can stop a few points short of the last letter.
    found = layer_math(page, pdf_bbox, pad=4.0)
    layer = _fold(found.symbols)
    if found.untrusted or sum(layer.values()) < max(3, sum(ours.values()) // 2):
        return MathCheck("no-layer")
    marks = marks_findings(html_marks(html), found.marks)
    if ours == layer:
        return MathCheck("differ", marks, ours, layer) if marks else MathCheck("agree")
    return MathCheck("differ", [{"kind": "inline_math_symbols_differ",
                                 "detail": "symbols differ from the PDF text layer",
                                 "values": _diff_values(ours, layer)}, *marks], ours, layer)


#: Look-alikes Surya confuses, as (what Surya wrote, what the page has). The
#: page side is always a Greek code point or a case twin: the text layer may
#: store a Greek glyph as a Latin letter (Symbol-style fonts put eta at "n"),
#: never the reverse, so a Latin letter in the layer is not trusted over
#: Surya's Greek.
CONFUSABLE = frozenset({
    ("v", "ν"), ("v", "υ"), ("u", "υ"), ("u", "μ"), ("t", "ι"), ("I", "ι"), ("i", "ι"), ("l", "ι"),
    ("n", "η"), ("p", "ρ"), ("w", "ω"), ("k", "κ"), ("a", "α"), ("x", "χ"), ("o", "σ"), ("e", "ε"),
    ("ν", "υ"), ("υ", "ν"), ("ν", "ι"), ("ι", "ν"),
    *((a, b) for c in "cosuvwxz" for a, b in ((c, c.upper()), (c.upper(), c))),
})
_COMMAND_OF = {v: k for k, v in reversed(list(GREEK.items()))}
#: Groups whose letters are words or environment names, not math symbols.
_TEXT_GROUP = re.compile(r"\\(?:text\w*|mbox|operatorname|mathrm)\s*\{[^{}]*\}"
                         r"|\\begin\{array\}\s*\{[^{}]*\}|\\(?:begin|end)\{[A-Za-z*]+\}")
_TOKEN = re.compile(r"\\[A-Za-z]+|.", re.S)


def _math_letters(latex: str, ch: str) -> int:
    """Occurrences of ``ch`` as a bare math symbol (a letter or Greek command)."""
    body = _TEXT_GROUP.sub(" ", latex)
    return sum(1 for t in _TOKEN.findall(body)
               if t == ch or (t.startswith("\\") and GREEK.get(t[1:]) == ch))


def _swap(latex: str, wrong: str, right: str) -> str:
    """Replace every bare ``wrong`` symbol outside text groups with ``right``."""
    new = f"\\{_COMMAND_OF[right]} " if right in _COMMAND_OF else right
    out, pos = [], 0
    for m in _TEXT_GROUP.finditer(latex):
        out.append(_swap_plain(latex[pos:m.start()], wrong, new))
        out.append(m.group(0))
        pos = m.end()
    out.append(_swap_plain(latex[pos:], wrong, new))
    return "".join(out)


def _swap_plain(text: str, wrong: str, new: str) -> str:
    return "".join(new if t == wrong or (t.startswith("\\") and GREEK.get(t[1:]) == wrong) else t
                   for t in _TOKEN.findall(text))


def confusable_pairs(check: MathCheck) -> list[tuple[str, str, int]] | None:
    """The look-alike swaps that explain a disagreement completely, or None.

    Every symbol the sidecar has too many of must pair with exactly one
    symbol the page has too many of, in equal numbers, through ``CONFUSABLE``.
    """
    extra, missing = check.sidecar - check.pdf, check.pdf - check.sidecar
    if not extra or sorted(extra.values()) != sorted(missing.values()):
        return None
    pairs = []
    for wrong, n in extra.items():
        rights = [r for r in missing if (wrong, r) in CONFUSABLE and missing[r] == n]
        if len(rights) != 1:
            return None
        pairs.append((wrong, rights[0], n))
    if len({r for _w, r, _n in pairs}) != len(pairs):
        return None
    return pairs


def repair_confusables(html: str, check: MathCheck, recheck) -> tuple[str, list[dict]] | None:
    """Swap look-alike symbols in the math of ``html`` to what the text layer holds.

    Applies only when every counted instance of the wrong symbol is a bare
    math symbol and all of them are wrong (their number equals the excess), and
    when ``recheck(new_html)`` then agrees. Returns the new HTML and repair
    records, or None.
    """
    pairs = confusable_pairs(check)
    if not pairs:
        return None
    parts = re.split(r"(<math[^>]*>)(.*?)(</math>)", html, flags=re.S)
    # re.split yields text, open tag, LaTeX, close tag, text, ...
    maths = range(2, len(parts), 4)
    latex = {i: html_lib.unescape(parts[i]) for i in maths}
    for wrong, _right, n in pairs:
        if sum(_math_letters(t, wrong) for t in latex.values()) != n:
            return None
    for i, t in latex.items():
        for wrong, right, _n in pairs:
            t = _swap(t, wrong, right)
        parts[i] = html_lib.escape(t, quote=False)
    new = "".join(parts)
    if recheck(new).status != "agree":
        return None
    return new, [{"kind": "symbol", "before": w, "after": r, "count": n, "route": "native_symbol"}
                 for w, r, n in pairs]
