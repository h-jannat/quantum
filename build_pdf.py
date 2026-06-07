"""Build paper.pdf from paper.md.

Extends the markdown-to-PDF rendering with:
  * LaTeX equations rendered via matplotlib mathtext ($...$ inline,
    $$...$$ display),
  * Figure groups (`![Fig.N](a.png|b.png|...)`) laid out as a grid.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "serif"

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_MPL_FONTS = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("STIX",        str(_MPL_FONTS / "STIXGeneral.ttf")))
pdfmetrics.registerFont(TTFont("STIX-Bold",   str(_MPL_FONTS / "STIXGeneralBol.ttf")))
pdfmetrics.registerFont(TTFont("STIX-Italic", str(_MPL_FONTS / "STIXGeneralItalic.ttf")))
pdfmetrics.registerFont(TTFont("STIX-BoldIt", str(_MPL_FONTS / "STIXGeneralBolIta.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily(
    "STIX", normal="STIX", bold="STIX-Bold",
    italic="STIX-Italic", boldItalic="STIX-BoldIt",
)

from reportlab.lib.colors import black, grey
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "paper.md"
DST = ROOT / "paper.pdf"
EQ_CACHE = ROOT / ".eq_cache"
EQ_CACHE.mkdir(exist_ok=True)

PAGE_W, PAGE_H = A4
L_MARGIN = R_MARGIN = 2.0 * cm
T_MARGIN = B_MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - L_MARGIN - R_MARGIN


# ---------------------------------------------------------------------------
# LaTeX -> PNG via matplotlib mathtext
# ---------------------------------------------------------------------------

_LATEX_SUBS = [
    (r"\\ge\b", r"\\geq"),
    (r"\\le\b", r"\\leq"),
    (r"\\neq\b", r"\\neq"),
    (r"\\operatorname\{([^}]+)\}", r"\\mathrm{\1}"),
    (r"\\boldsymbol\{([^}]+)\}", r"\\mathbf{\1}"),
    (r"\\lVert", r"\\|"),
    (r"\\rVert", r"\\|"),
    (r"\\lvert", r"|"),
    (r"\\rvert", r"|"),
    (r"\\tfrac\b", r"\\frac"),
    (r"\\dfrac\b", r"\\frac"),
    (r"\\qquad\b", r"\\ \\ \\ \\ "),
    (r"\\quad\b", r"\\ \\ "),
    (r"\\;", r"\\,"),
    (r"\\!", r""),
    (r"\\vartheta", r"\\theta"),
    (r"\\mathrm\{\\N\}", r"\\mathrm{N}"),
    (r"\\\\", r" "),
    (r"\\top\b", r"T"),
    (r"\\infty\b", r"\\infty"),
    (r"\\bigl\b", r""),
    (r"\\bigr\b", r""),
    (r"\\Bigl\b", r""),
    (r"\\Bigr\b", r""),
    (r"\\big\b", r""),
    (r"\\Big\b", r""),
    (r"\\left\b", r""),
    (r"\\right\b", r""),
    (r"\\varphi", r"\\phi"),
    (r"\\epsilon", r"\\varepsilon"),
    (r"\\Theta", r"\\Theta"),
    (r"\\Omega", r"\\Omega"),
]


def _normalise_latex(src: str) -> str:
    # Mathtext is a single-line parser; join lines with a thin space.
    out = re.sub(r"\s*\n\s*", " ", src.strip())
    for pat, repl in _LATEX_SUBS:
        out = re.sub(pat, repl, out)
    return out


def render_equation(latex: str, display: bool = False, fontsize: float = 11.0) -> Path:
    latex = _normalise_latex(latex)
    key = hashlib.md5(f"{latex}|{display}|{fontsize}".encode()).hexdigest()
    out = EQ_CACHE / f"{key}.png"
    if out.exists():
        return out

    fs = fontsize * (1.35 if display else 1.0)
    fig = plt.figure(figsize=(0.01, 0.01), dpi=300)
    fig.patch.set_alpha(0)
    txt = fig.text(0, 0, f"${latex}$", fontsize=fs)
    fig.canvas.draw()
    bbox = txt.get_window_extent()
    w_in = bbox.width / fig.dpi + 0.05
    h_in = bbox.height / fig.dpi + 0.05
    fig.set_size_inches(max(0.2, w_in), max(0.1, h_in))
    txt.set_position((0.02, 0.25))
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02,
                transparent=True)
    plt.close(fig)
    return out


def eq_image_flowable(latex: str, display: bool = False, fontsize: float = 11.0):
    path = render_equation(latex, display=display, fontsize=fontsize)
    # Size in points; at 300 dpi each px is 72/300 pt.
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    w_pt = w_px * 72.0 / 300.0
    h_pt = h_px * 72.0 / 300.0
    max_w = CONTENT_W - 0.2 * cm
    if w_pt > max_w:
        scale = max_w / w_pt
        w_pt *= scale
        h_pt *= scale
    img = Image(str(path), width=w_pt, height=h_pt)
    img.hAlign = "CENTER" if display else "LEFT"
    return img


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def make_styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=base["Title"], fontName="STIX-Bold",
        fontSize=16, leading=19, alignment=TA_CENTER, spaceAfter=10,
        textColor=black,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="STIX-Bold",
        fontSize=12, leading=14, spaceBefore=12, spaceAfter=5,
        textColor=black, alignment=TA_LEFT,
    )
    s["h3"] = ParagraphStyle(
        "h3", parent=base["Heading3"], fontName="STIX-BoldIt",
        fontSize=11, leading=13, spaceBefore=7, spaceAfter=3,
        textColor=black, alignment=TA_LEFT,
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["BodyText"], fontName="STIX",
        fontSize=10, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5,
    )
    s["abstract"] = ParagraphStyle(
        "abstract", parent=base["BodyText"], fontName="STIX",
        fontSize=9.5, leading=12, alignment=TA_JUSTIFY,
        leftIndent=0.6 * cm, rightIndent=0.6 * cm, spaceAfter=5,
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=base["BodyText"], fontName="STIX",
        fontSize=9, leading=11.5, alignment=TA_JUSTIFY,
        leftIndent=0.3 * cm, rightIndent=0.3 * cm, spaceBefore=4,
        spaceAfter=8,
    )
    s["ref"] = ParagraphStyle(
        "ref", parent=base["BodyText"], fontName="STIX",
        fontSize=9, leading=11, alignment=TA_JUSTIFY,
        leftIndent=0.8 * cm, firstLineIndent=-0.8 * cm, spaceAfter=3,
    )
    s["list"] = ParagraphStyle(
        "list", parent=base["BodyText"], fontName="STIX",
        fontSize=10, leading=12.5, alignment=TA_JUSTIFY,
        leftIndent=0.6 * cm, firstLineIndent=-0.25 * cm, spaceAfter=3,
    )
    return s


# ---------------------------------------------------------------------------
# Inline parsing
# ---------------------------------------------------------------------------

MATH_PLACEHOLDER = "\x00MATH{}\x00"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline_md_to_flowables(text: str, style) -> list:
    """Convert a mixed text-with-inline-math paragraph to a list of flowables.

    Splits on $...$ boundaries and renders the math pieces as tiny inline
    PNGs. For simplicity, if inline math is present we still emit a single
    Paragraph per run (reportlab doesn't support true inline images in
    Paragraphs without a custom <img> tag — which it does support).
    """
    # Replace $...$ with <img src="..."/> tags referencing cached pngs.
    def repl(m):
        latex = m.group(1).strip()
        path = render_equation(latex, display=False, fontsize=style.fontSize)
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w_px, h_px = im.size
        h_pt = style.fontSize * 1.15
        scale = h_pt / (h_px * 72.0 / 300.0)
        w_pt = (w_px * 72.0 / 300.0) * scale
        return f'<img src="{path}" width="{w_pt:.1f}" height="{h_pt:.1f}" valign="-2"/>'

    # Use a non-greedy match for inline $...$. Escape first.
    # We need to pull math out BEFORE HTML escaping to avoid mangling.
    parts = []
    i = 0
    out = []
    while True:
        m = re.search(r"\$([^$]+)\$", text[i:])
        if not m:
            out.append(_escape(text[i:]))
            break
        start = i + m.start()
        end = i + m.end()
        out.append(_escape(text[i:start]))
        out.append(repl(m))
        i = end
    html = "".join(out)

    # Bold / italic / code on non-math portions — the <img> tags stay intact.
    html = re.sub(r"\*\*([^*]+?)\*\*", r"<b>\1</b>", html)
    html = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", r"<i>\1</i>", html)
    html = re.sub(r"`([^`]+?)`", r'<font face="Courier" size="9">\1</font>', html)
    html = html.replace("—", "&mdash;").replace("–", "&ndash;")
    return [Paragraph(html, style)]


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------

def parse_table(lines: list[str], i: int, styles) -> tuple[Table, int]:
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    data_lines = []
    j = i + 2
    while j < len(lines) and lines[j].strip().startswith("|"):
        data_lines.append(lines[j])
        j += 1
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in data_lines]

    th_style = ParagraphStyle(
        "th", fontName="STIX-Bold", fontSize=9.5, leading=12, alignment=TA_LEFT,
    )
    td_style = ParagraphStyle(
        "td", fontName="STIX", fontSize=9.5, leading=12, alignment=TA_LEFT,
    )

    def cell(text: str, st):
        return inline_md_to_flowables(text, st)[0]

    data = [[cell(c, th_style) for c in header]]
    for r in rows:
        data.append([cell(c, td_style) for c in r])

    t = Table(data, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t, j


def figure_group(paths: list[str]):
    """Lay out a group of images inline. Supports 1, 2 or 4 panels."""
    from PIL import Image as PILImage
    flowables = []
    n = len(paths)
    if n == 1:
        p = paths[0]
        with PILImage.open(p) as im:
            w_px, h_px = im.size
        target_w = min(CONTENT_W * 0.80, 12 * cm)
        scale = target_w / w_px * 72.0 / 72.0  # handled below
        h_pt = h_px * (target_w / w_px)
        img = Image(p, width=target_w, height=h_pt)
        img.hAlign = "CENTER"
        return [img]
    if n == 2:
        cell_w = (CONTENT_W - 0.4 * cm) / 2.0
        imgs = []
        for p in paths:
            with PILImage.open(p) as im:
                w_px, h_px = im.size
            h_pt = h_px * (cell_w / w_px)
            imgs.append(Image(p, width=cell_w, height=h_pt))
        t = Table([imgs], colWidths=[cell_w, cell_w])
        t.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        t.hAlign = "CENTER"
        return [t]
    if n == 4:
        cell_w = (CONTENT_W - 0.4 * cm) / 2.0
        cells = []
        for p in paths:
            with PILImage.open(p) as im:
                w_px, h_px = im.size
            h_pt = h_px * (cell_w / w_px)
            cells.append(Image(p, width=cell_w, height=h_pt))
        rows = [cells[:2], cells[2:]]
        t = Table(rows, colWidths=[cell_w, cell_w])
        t.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        t.hAlign = "CENTER"
        return [t]
    # Fallback: stack.
    out = []
    for p in paths:
        with PILImage.open(p) as im:
            w_px, h_px = im.size
        target_w = min(CONTENT_W * 0.8, 12 * cm)
        h_pt = h_px * (target_w / w_px)
        im2 = Image(p, width=target_w, height=h_pt)
        im2.hAlign = "CENTER"
        out.append(im2)
    return out


def is_table_start(lines: list[str], i: int) -> bool:
    if not lines[i].strip().startswith("|"):
        return False
    if i + 1 >= len(lines):
        return False
    sep = lines[i + 1].replace("|", "").replace(" ", "")
    return bool(sep) and set(sep) <= set("-:")


def build_story():
    styles = make_styles()
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()

    story = []
    i = 0
    in_abstract = False
    in_refs = False

    while i < len(lines):
        line = lines[i]
        s = line.strip()

        # Horizontal rule
        if s == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.4, color=grey,
                                    spaceBefore=0, spaceAfter=4))
            i += 1
            continue

        # Title
        if s.startswith("# "):
            story.append(Paragraph(_escape(s[2:]), styles["title"]))
            i += 1
            continue

        # Section heading
        if s.startswith("## "):
            heading = s[3:]
            in_abstract = heading.lower().startswith("abstract")
            in_refs = heading.lower().startswith("references")
            story.append(Paragraph(_escape(heading), styles["h2"]))
            i += 1
            continue

        # Subsection heading
        if s.startswith("### "):
            story.append(Paragraph(_escape(s[4:]), styles["h3"]))
            i += 1
            continue

        # Blank
        if not s:
            i += 1
            continue

        # Display math $$...$$
        if s.startswith("$$"):
            # collect until closing $$
            content = s[2:]
            if content.endswith("$$"):
                latex = content[:-2].strip()
                i += 1
            else:
                buf = [content]
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    buf.append(lines[i])
                    i += 1
                if i < len(lines):
                    buf.append(lines[i].rstrip("$").rstrip())
                    i += 1
                latex = "\n".join(buf).strip()
            # Handle \tag{N} suffix
            tag = None
            m = re.search(r"\\tag\{([^}]+)\}", latex)
            if m:
                tag = m.group(1)
                latex = (latex[:m.start()] + latex[m.end():]).strip()
            eq = eq_image_flowable(latex, display=True, fontsize=11.0)
            if tag:
                wrap = Table(
                    [[eq, Paragraph(f"({tag})",
                                    ParagraphStyle(
                                        "tag", fontName="STIX",
                                        fontSize=10, alignment=TA_CENTER))]],
                    colWidths=[CONTENT_W - 1.8 * cm, 1.8 * cm])
                wrap.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(wrap)
            else:
                story.append(Spacer(1, 2))
                story.append(eq)
                story.append(Spacer(1, 4))
            continue

        # Figure group  ![Fig.N](a.png|b.png|...)
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            paths = [p.strip() for p in m.group(2).split("|")]
            # Resolve relative to project root.
            resolved = [str((ROOT / p).resolve()) for p in paths]
            story.append(Spacer(1, 4))
            for fl in figure_group(resolved):
                story.append(fl)
            story.append(Spacer(1, 2))
            i += 1
            continue

        # Table
        if is_table_start(lines, i):
            tbl, i = parse_table(lines, i, styles)
            story.append(Spacer(1, 3))
            story.append(tbl)
            story.append(Spacer(1, 5))
            continue

        # Numbered or bulleted list item
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*(?:[-*]|\d+\.)\s+", lines[i]):
                m2 = re.match(r"^\s*(?:([-*])|(\d+)\.)\s+(.*)$", lines[i])
                marker = "•" if m2.group(1) else f"{m2.group(2)}."
                item = m2.group(3)
                while (i + 1 < len(lines)
                       and lines[i + 1].startswith(("   ", "\t"))
                       and not re.match(r"^\s*(?:[-*]|\d+\.)\s+", lines[i + 1])
                       and lines[i + 1].strip()):
                    item += " " + lines[i + 1].strip()
                    i += 1
                items.append((marker, item))
                i += 1
            for marker, it in items:
                flows = inline_md_to_flowables(f"{marker}  {it}", styles["list"])
                story.extend(flows)
            story.append(Spacer(1, 3))
            continue

        # Paragraph: join consecutive non-blank, non-special lines.
        para_lines = [line]
        while i + 1 < len(lines):
            nxt = lines[i + 1]
            nxt_s = nxt.strip()
            if (not nxt_s
                    or nxt_s.startswith(("#", "|", "---", "$$"))
                    or nxt_s.startswith(("- ", "* "))
                    or re.match(r"^\s*\d+\.\s+", nxt)
                    or re.match(r"^!\[", nxt_s)):
                break
            i += 1
            para_lines.append(lines[i])
        paragraph = " ".join(l.strip() for l in para_lines)

        if in_abstract:
            style = styles["abstract"]
        elif in_refs and paragraph.startswith("["):
            style = styles["ref"]
        elif paragraph.startswith(("**Fig.", "**Table ")):
            style = styles["caption"]
        else:
            style = styles["body"]

        flows = inline_md_to_flowables(paragraph, style)
        story.extend(flows)
        i += 1

    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(grey)
    canvas.drawCentredString(PAGE_W / 2.0, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def build():
    story = build_story()
    doc = SimpleDocTemplate(
        str(DST), pagesize=A4,
        leftMargin=L_MARGIN, rightMargin=R_MARGIN,
        topMargin=T_MARGIN, bottomMargin=B_MARGIN,
        title="Classical vs Quantum-Inspired STFT",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {DST} ({DST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
