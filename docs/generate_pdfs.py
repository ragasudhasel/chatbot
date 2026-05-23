"""
PDF Generator — Converts README.md and eval_report.md to styled PDFs
Uses reportlab (already installed) — no pandoc needed
"""

import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Color Palette ─────────────────────────────────────────────────────────────
INDIGO       = colors.HexColor("#6366f1")
EMERALD      = colors.HexColor("#10b981")
VIOLET       = colors.HexColor("#8b5cf6")
DARK_BG      = colors.HexColor("#0f172a")
CARD_BG      = colors.HexColor("#1e293b")
TEXT_MAIN    = colors.HexColor("#f1f5f9")
TEXT_MUTED   = colors.HexColor("#94a3b8")
BORDER       = colors.HexColor("#334155")
WHITE        = colors.white
AMBER        = colors.HexColor("#f59e0b")
ROSE         = colors.HexColor("#f43f5e")

# ── Styles ────────────────────────────────────────────────────────────────────
def make_styles():
    styles = getSampleStyleSheet()

    base = dict(fontName="Helvetica", textColor=TEXT_MAIN,
                backColor=DARK_BG, leading=16)

    return {
        "doc_title": ParagraphStyle("doc_title",
            fontName="Helvetica-Bold", fontSize=26,
            textColor=WHITE, leading=34, spaceAfter=4,
            alignment=TA_LEFT),

        "doc_subtitle": ParagraphStyle("doc_subtitle",
            fontName="Helvetica", fontSize=13,
            textColor=TEXT_MUTED, leading=18, spaceAfter=16,
            alignment=TA_LEFT),

        "h1": ParagraphStyle("h1",
            fontName="Helvetica-Bold", fontSize=18,
            textColor=INDIGO, leading=24,
            spaceBefore=18, spaceAfter=6),

        "h2": ParagraphStyle("h2",
            fontName="Helvetica-Bold", fontSize=14,
            textColor=EMERALD, leading=20,
            spaceBefore=14, spaceAfter=4),

        "h3": ParagraphStyle("h3",
            fontName="Helvetica-Bold", fontSize=12,
            textColor=VIOLET, leading=18,
            spaceBefore=10, spaceAfter=3),

        "body": ParagraphStyle("body",
            fontName="Helvetica", fontSize=10,
            textColor=TEXT_MAIN, leading=16,
            spaceBefore=2, spaceAfter=4),

        "muted": ParagraphStyle("muted",
            fontName="Helvetica-Oblique", fontSize=9,
            textColor=TEXT_MUTED, leading=14,
            spaceBefore=0, spaceAfter=3),

        "code": ParagraphStyle("code",
            fontName="Courier", fontSize=8.5,
            textColor=colors.HexColor("#a5f3fc"),
            backColor=colors.HexColor("#0c1829"),
            leading=13, leftIndent=10, rightIndent=10,
            spaceBefore=4, spaceAfter=4,
            borderColor=colors.HexColor("#1e3a5f"),
            borderWidth=1, borderPadding=8),

        "bullet": ParagraphStyle("bullet",
            fontName="Helvetica", fontSize=10,
            textColor=TEXT_MAIN, leading=15,
            leftIndent=16, firstLineIndent=-10,
            spaceBefore=1, spaceAfter=2),

        "bold": ParagraphStyle("bold",
            fontName="Helvetica-Bold", fontSize=10,
            textColor=WHITE, leading=16,
            spaceBefore=2, spaceAfter=2),

        "meta": ParagraphStyle("meta",
            fontName="Helvetica", fontSize=9,
            textColor=TEXT_MUTED, leading=14,
            spaceBefore=1, spaceAfter=1),

        "section_label": ParagraphStyle("section_label",
            fontName="Helvetica-Bold", fontSize=8,
            textColor=INDIGO, leading=12,
            spaceBefore=0, spaceAfter=2,
            alignment=TA_LEFT),
    }


def clean_line(text):
    """Remove markdown syntax for inline text."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<font name="Courier" color="#a5f3fc">\1</font>', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove emoji-heavy chars that reportlab can't render
    text = re.sub(r'[^\x00-\x7F📊🏆📐💡🔍📉✅🚀🌐🤗📝]', lambda m: m.group() if ord(m.group()) < 0x10000 else '', text)
    return text


def md_line_to_flowables(line, styles):
    """Convert a single markdown line to reportlab flowables."""
    out = []
    stripped = line.strip()

    if not stripped:
        out.append(Spacer(1, 6))
        return out

    if stripped.startswith("# "):
        out.append(Spacer(1, 8))
        out.append(Paragraph(clean_line(stripped[2:]), styles["h1"]))
        out.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceAfter=4))
        return out

    if stripped.startswith("## "):
        out.append(Spacer(1, 4))
        out.append(Paragraph(clean_line(stripped[3:]), styles["h2"]))
        return out

    if stripped.startswith("### "):
        out.append(Paragraph(clean_line(stripped[4:]), styles["h3"]))
        return out

    if stripped.startswith("#### "):
        out.append(Paragraph(clean_line(stripped[5:]), styles["bold"]))
        return out

    if stripped == "---":
        out.append(Spacer(1, 4))
        out.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=4))
        return out

    if stripped.startswith("- ") or stripped.startswith("* "):
        text = clean_line(stripped[2:])
        out.append(Paragraph(f"• {text}", styles["bullet"]))
        return out

    if re.match(r'^\d+\. ', stripped):
        text = re.sub(r'^\d+\. ', '', stripped)
        out.append(Paragraph(f"  {clean_line(text)}", styles["bullet"]))
        return out

    if stripped.startswith("|") and stripped.endswith("|"):
        # Table row — handled separately via buffer
        return None  # Signal: table row

    if stripped.startswith("**") and stripped.endswith("**") and stripped.count("**") == 2:
        out.append(Paragraph(clean_line(stripped), styles["bold"]))
        return out

    if stripped.startswith("*") and stripped.endswith("*") and stripped.count("*") == 2:
        out.append(Paragraph(clean_line(stripped), styles["muted"]))
        return out

    out.append(Paragraph(clean_line(stripped), styles["body"]))
    return out


def build_table(rows_raw, styles_dict):
    """Build a styled Table from markdown table rows."""
    rows = []
    for row in rows_raw:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return None

    # Detect separator row (---|---) and remove it
    clean_rows = [r for r in rows if not all(re.match(r'^[-:]+$', c.strip()) for c in r)]

    if not clean_rows:
        return None

    # Style
    col_count = max(len(r) for r in clean_rows)
    for r in clean_rows:
        while len(r) < col_count:
            r.append("")

    para_rows = []
    for i, row in enumerate(clean_rows):
        style = styles_dict["bold"] if i == 0 else styles_dict["body"]
        para_rows.append([Paragraph(clean_line(c), style) for c in row])

    col_width = (A4[0] - 3*cm) / max(col_count, 1)
    t = Table(para_rows, colWidths=[col_width] * col_count, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#1e293b")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  INDIGO),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#0f172a"), colors.HexColor("#131f35")]),
        ("TEXTCOLOR",    (0, 1), (-1, -1), TEXT_MAIN),
        ("FONTSIZE",     (0, 1), (-1, -1), 8.5),
        ("GRID",         (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return t


def header_footer(canvas, doc):
    """Draw dark background + header/footer on every page."""
    canvas.saveState()
    W, H = A4

    # Dark page background
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # Top accent bar
    canvas.setFillColor(INDIGO)
    canvas.rect(0, H - 6, W, 6, fill=1, stroke=0)

    # Bottom bar
    canvas.setFillColor(CARD_BG)
    canvas.rect(0, 0, W, 28, fill=1, stroke=0)

    # Footer text
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(1.5*cm, 10, "AI Assistants Arena — Evaluation Report")
    canvas.drawRightString(W - 1.5*cm, 10, f"Page {doc.page}")

    canvas.restoreState()


def convert_md_to_pdf(md_path: str, pdf_path: str, doc_title: str, doc_subtitle: str):
    """Full pipeline: read MD → parse → build PDF."""
    styles = make_styles()

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=1.8*cm,
        bottomMargin=1.5*cm,
        title=doc_title,
        author="AI Assistants Arena",
    )

    story = []

    # Cover block
    story.append(Spacer(1, 12))
    story.append(Paragraph(doc_title, styles["doc_title"]))
    story.append(Paragraph(doc_subtitle, styles["doc_subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=INDIGO, spaceAfter=16))

    # Parse lines
    i = 0
    code_buffer = []
    table_buffer = []
    in_code = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Code block toggle
        if stripped.startswith("```"):
            if in_code:
                # End code block
                if code_buffer:
                    code_text = "\n".join(code_buffer)
                    story.append(Preformatted(code_text, styles["code"]))
                    story.append(Spacer(1, 4))
                code_buffer = []
                in_code = False
            else:
                # Start code block — flush table if any
                if table_buffer:
                    t = build_table(table_buffer, styles)
                    if t:
                        story.append(t)
                        story.append(Spacer(1, 8))
                    table_buffer = []
                in_code = True
            i += 1
            continue

        if in_code:
            code_buffer.append(line.rstrip())
            i += 1
            continue

        # Table rows
        if stripped.startswith("|"):
            table_buffer.append(stripped)
            i += 1
            continue
        else:
            if table_buffer:
                t = build_table(table_buffer, styles)
                if t:
                    story.append(t)
                    story.append(Spacer(1, 8))
                table_buffer = []

        # Normal lines
        flowables = md_line_to_flowables(line, styles)
        if flowables:
            story.extend(flowables)
        i += 1

    # Flush any remaining table
    if table_buffer:
        t = build_table(table_buffer, styles)
        if t:
            story.append(t)

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"✅ PDF created: {pdf_path}")


if __name__ == "__main__":
    base = r"d:\HOME\ai-assistants"

    # 1. Evaluation Report PDF
    convert_md_to_pdf(
        md_path=os.path.join(base, "docs", "eval_report.md"),
        pdf_path=os.path.join(base, "docs", "Evaluation_Report.pdf"),
        doc_title="LLM Evaluation Report",
        doc_subtitle="Gemini 2.5 Flash (Frontier) vs Qwen 2.5 72B Instruct (OSS) · May 2026",
    )

    # 2. README PDF
    convert_md_to_pdf(
        md_path=os.path.join(base, "README.md"),
        pdf_path=os.path.join(base, "docs", "README.pdf"),
        doc_title="AI Assistants Arena",
        doc_subtitle="Project Documentation · Setup · Architecture · Evaluation",
    )

    print("\n📁 Both PDFs saved to: d:\\HOME\\ai-assistants\\docs\\")
