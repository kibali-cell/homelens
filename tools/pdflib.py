"""Small reportlab toolkit for the study-guide: fonts, styles, code/output/figure/callout helpers."""
import json
import re
import textwrap
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
FIGS, OUTS, STEPS = ROOT / "reports/figures", ROOT / "reports/outputs", ROOT / "steps"
R = json.loads((OUTS / "results.json").read_text())

# ---------------------------------------------------------------- fonts
_fd = Path(matplotlib.get_data_path()) / "fonts/ttf"
for name, f in [("Sans", "DejaVuSans.ttf"), ("Sans-B", "DejaVuSans-Bold.ttf"), ("Sans-I", "DejaVuSans-Oblique.ttf"),
                ("Sans-BI", "DejaVuSans-BoldOblique.ttf"), ("Mono", "DejaVuSansMono.ttf"), ("Mono-B", "DejaVuSansMono-Bold.ttf"),
                ("Mono-I", "DejaVuSansMono-Oblique.ttf"), ("Mono-BI", "DejaVuSansMono-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(_fd / f)))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-BI")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-B", italic="Mono-I", boldItalic="Mono-BI")

NAVY, ORANGE, GREEN, RED = colors.HexColor("#1f4e79"), colors.HexColor("#d9822b"), colors.HexColor("#2e7d5b"), colors.HexColor("#b23a48")
INK, GREY, LIGHT = colors.HexColor("#1f2937"), colors.HexColor("#6b7280"), colors.HexColor("#f3f4f6")
PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

S = {}
S["body"] = ParagraphStyle("body", fontName="Sans", fontSize=9.2, leading=13.8, textColor=INK, alignment=TA_LEFT, spaceAfter=5.5)
S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=13, bulletIndent=2, spaceAfter=2.5)
S["h1"] = ParagraphStyle("h1", fontName="Sans-B", fontSize=22, leading=27, textColor=NAVY, spaceBefore=0, spaceAfter=10)
S["h1c"] = ParagraphStyle("h1c", fontName="Sans-B", fontSize=22, leading=27, textColor=NAVY, spaceAfter=10)
S["h2"] = ParagraphStyle("h2", fontName="Sans-B", fontSize=14, leading=18, textColor=NAVY, spaceBefore=14, spaceAfter=6)
S["h3"] = ParagraphStyle("h3", fontName="Sans-B", fontSize=10.8, leading=14, textColor=ORANGE, spaceBefore=10, spaceAfter=4)
S["cap"] = ParagraphStyle("cap", parent=S["body"], fontName="Sans-I", fontSize=8, leading=10.5, textColor=GREY, alignment=TA_CENTER, spaceAfter=9)
S["small"] = ParagraphStyle("small", parent=S["body"], fontSize=8, leading=11)
S["cell"] = ParagraphStyle("cell", parent=S["body"], fontSize=8, leading=10.5, spaceAfter=0)
S["cellb"] = ParagraphStyle("cellb", parent=S["cell"], fontName="Sans-B", textColor=colors.white)
S["code"] = ParagraphStyle("code", fontName="Mono", fontSize=6.7, leading=8.4, textColor=colors.HexColor("#111827"))
S["out"] = ParagraphStyle("out", fontName="Mono", fontSize=6.7, leading=8.4, textColor=colors.HexColor("#0f3d2e"))
S["tag"] = ParagraphStyle("tag", fontName="Sans-B", fontSize=7, leading=9, textColor=GREY, spaceBefore=4, spaceAfter=1.5)
S["toc1"] = ParagraphStyle("toc1", fontName="Sans-B", fontSize=10, leading=15, leftIndent=0, textColor=NAVY)
S["toc2"] = ParagraphStyle("toc2", fontName="Sans", fontSize=8.6, leading=12, leftIndent=14, textColor=INK)
S["qa_q"] = ParagraphStyle("qa_q", parent=S["body"], fontName="Sans-B", spaceAfter=1.5, spaceBefore=4, textColor=NAVY)
S["qa_a"] = ParagraphStyle("qa_a", parent=S["body"], leftIndent=10, spaceAfter=4)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(t):
    """Light markup: **bold**, *italic*, `mono`. Raw <sub>/<super>/<b>/<i> tags are allowed too."""
    t = re.sub(r"`([^`]+)`", lambda m: '<font name="Mono" size="8" color="#7c2d12">' + esc(m.group(1)) + "</font>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    return t


def P(t, style="body"): return Paragraph(inline(t), S[style])
def H1(t): return Paragraph(inline(t), S["h1"])
def H2(t): return Paragraph(inline(t), S["h2"])
def H3(t): return Paragraph(inline(t), S["h3"])
def SP(h=6): return Spacer(1, h)


def BL(items, style="bullet"):
    return [Paragraph(inline(i), S[style], bulletText="•") for i in items]


def NL(items):
    return [Paragraph(inline(i), S["bullet"], bulletText=f"{n}.") for n, i in enumerate(items, 1)]


def callout(kind, title, text):
    pal = {"key": (NAVY, "#e8f0f8"), "warn": (RED, "#fbeaec"), "tip": (GREEN, "#e6f4ee"), "think": (ORANGE, "#fdf1e3")}
    c, bg = pal[kind]
    body = [Paragraph(f'<font name="Sans-B" color="{c.hexval()}">{inline(title)}</font>', S["cell"])]
    body += [Paragraph(inline(x), ParagraphStyle("c", parent=S["cell"], fontSize=8.6, leading=12.2, spaceAfter=3)) for x in (text if isinstance(text, list) else [text])]
    t = Table([[body]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)), ("LINEBEFORE", (0, 0), (0, -1), 3, c),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return [t, SP(7)]


def table(rows, widths=None, header=True, font=8, zebra=True, align_right_from=None):
    st = ParagraphStyle("t", parent=S["cell"], fontSize=font, leading=font + 2.6)
    sth = ParagraphStyle("th", parent=st, fontName="Sans-B", textColor=colors.white)
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(inline(str(c)), sth if (header and i == 0) else st) for c in r])
    n = len(rows[0])
    widths = widths or [CONTENT_W / n] * n
    widths = [w * CONTENT_W / sum(widths) for w in widths]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db"))]
    if header: cmds.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0: cmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(cmds))
    return [t, SP(8)]


# ---------------------------------------------------------------- code / output / figures
def _blocks(step):
    src = (STEPS / f"{step}.py").read_text().splitlines()
    blocks, name, buf = {}, "header", []
    for line in src:
        m = re.match(r"# %%\s+(\S+)", line)
        if m:
            blocks[name] = buf; name, buf = m.group(1), []
        else:
            buf.append(line)
    blocks[name] = buf
    return blocks


def _wrap_lines(lines, width):
    out = []
    for ln in lines:
        if len(ln) <= width:
            out.append(ln); continue
        indent = len(ln) - len(ln.lstrip())
        parts = textwrap.wrap(ln, width, subsequent_indent=" " * (indent + 4), break_long_words=True, replace_whitespace=False)
        out.extend(parts)
    return out


def _shaded(lines, style, bg, bar, chunk=34):
    flow = []
    lines = _wrap_lines(lines, 112)
    for i in range(0, len(lines), chunk):
        txt = "\n".join(lines[i:i + chunk])
        t = Table([[Preformatted(txt, S[style])]], colWidths=[CONTENT_W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)), ("LINEBEFORE", (0, 0), (0, -1), 2.5, bar),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        flow.append(t)
    return flow


def code(step, block, label=None, skip_lines=0, drop_prefix=(), stop_at=None):
    lines = [l for l in _blocks(step)[block]]
    while lines and not lines[0].strip(): lines.pop(0)
    while lines and not lines[-1].strip(): lines.pop()
    lines = lines[skip_lines:]
    if stop_at:
        for i, l in enumerate(lines):
            if l.startswith(stop_at):
                lines = lines[:i]; break
        while lines and not lines[-1].strip(): lines.pop()
    lines = [l for l in lines if not any(l.startswith(p) for p in drop_prefix)]
    tag = Paragraph(f'CODE  ·  steps/{step}.py  ·  block “{block}”' + (f"  ·  {label}" if label else ""), S["tag"])
    return [tag] + _shaded(lines, "code", "#f6f7f9", NAVY)


def output(step, block, max_lines=40, label=None):
    p = OUTS / step / f"{block}.txt"
    lines = [l for l in p.read_text().rstrip().splitlines() if not l.startswith("[figure saved]")]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines omitted; see reports/outputs/{step}/{block}.txt)"]
    tag = Paragraph("OUTPUT  ·  produced by running the block above" + (f"  ·  {label}" if label else ""), S["tag"])
    return [tag] + _shaded(lines, "out", "#eef7f2", GREEN) + [SP(6)]


def snippet(text, label="CODE"):
    return [Paragraph(label, S["tag"])] + _shaded(text.strip("\n").splitlines(), "code", "#f6f7f9", NAVY) + [SP(6)]


def fig(name, caption, width=1.0):
    path = FIGS / f"{name}.png"
    from PIL import Image as PI
    w, h = PI.open(path).size
    W = CONTENT_W * width
    img = Image(str(path), width=W, height=W * h / w)
    return [KeepTogether([img, Paragraph(inline(caption), S["cap"])])]


def QA(items):
    out = []
    for n, (q, a) in enumerate(items, 1):
        out.append(KeepTogether([Paragraph(f"Q{n}. {inline(q)}", S["qa_q"]), Paragraph(inline(a), S["qa_a"])]))
    return out


# ---------------------------------------------------------------- document
class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=22 * mm, bottomMargin=20 * mm,
                         title="Machine Learning Fundamentals - A Project-Based Study Guide", author="Claude", **kw)
        fr = Frame(MARGIN, 20 * mm, CONTENT_W, PAGE_H - 42 * mm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="cover", frames=[fr], onPage=self._cover),
                               PageTemplate(id="main", frames=[fr], onPageEnd=self._page)])
        self.running = ""

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph):
            st = fl.style.name
            if st in ("h1", "h2"):
                text = fl.getPlainText()
                lvl = 0 if st == "h1" else 1
                key = f"k{self.seq.nextf('toc')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=lvl, closed=True)
                self.notify("TOCEntry", (lvl, text, self.page, key))
                if st == "h1": self.running = text

    def beforeDocument(self):
        self.running = ""

    def _cover(self, c, d): pass

    def _page(self, c, d):
        c.saveState()
        c.setStrokeColor(colors.HexColor("#d1d5db")); c.setLineWidth(0.5)
        c.line(MARGIN, PAGE_H - 16 * mm, PAGE_W - MARGIN, PAGE_H - 16 * mm)
        c.setFont("Sans", 7.5); c.setFillColor(GREY)
        c.drawString(MARGIN, PAGE_H - 13.5 * mm, "HomeLens · Machine Learning Fundamentals Study Guide")
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 13.5 * mm, self.running[:70])
        c.line(MARGIN, 15 * mm, PAGE_W - MARGIN, 15 * mm)
        c.drawCentredString(PAGE_W / 2, 10.5 * mm, f"{d.page}")
        c.restoreState()


def toc():
    t = TableOfContents()
    t.levelStyles = [S["toc1"], S["toc2"]]
    return t
