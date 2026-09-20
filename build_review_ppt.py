"""Build the review deck, structured 1:1 against the four grading criteria.

    1. Base Paper Implementation      - correct replication of architecture
    2. Dataset Preparation            - cleaning, augmentation, splitting
    3. Model Training & Evaluation    - hyperparameters, loss, metrics
    4. Comparison with Base Paper     - correctness of results and explanation

Every figure comes from this repository: args.yaml for the configuration,
results.csv for the curves, a re-validation of the trained weights for the
per-class table, and tools/model_facts.py for the model numbers.

The deck's hardest slide is criterion 4, because the honest comparison is
partly a non-comparison: one base paper's architecture is implemented but
untrained, and saying so is worth more than a table of numbers that describe
nothing.

Run:  python build_review_ppt.py
Out:  ClassGraph_Review.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "ppt_assets"
RUN = ROOT / "runs/behaviour/merged4_aug"
OUT = ROOT / "ClassGraph_Review.pptx"

SW, SH = 13.333, 7.5
ML = MR = 0.68
CW = SW - ML - MR

INK = RGBColor(0x10, 0x27, 0x3F)
BODY = RGBColor(0x3C, 0x50, 0x66)
MUTE = RGBColor(0x7B, 0x8C, 0x9E)
TEAL = RGBColor(0x0E, 0x7C, 0x86)
TEAL_D = RGBColor(0x08, 0x59, 0x61)
AMBER = RGBColor(0xB4, 0x7A, 0x14)
GREEN = RGBColor(0x1F, 0x6F, 0x50)
RED = RGBColor(0xA9, 0x33, 0x2A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PANEL = RGBColor(0xF3, 0xF7, 0xFA)
PANEL2 = RGBColor(0xE9, 0xF1, 0xF4)
BORDER = RGBColor(0xDA, 0xE3, 0xEB)

FONT = "Segoe UI"
FONT_SB = "Segoe UI Semibold"
MONO = "Consolas"


def R(t, size=12, bold=False, color=BODY, font=FONT, italic=False):
    return {"t": t, "size": size, "bold": bold, "color": color,
            "font": font, "italic": italic}


def PR(runs, align=None, space_before=None, space_after=None, line=None):
    return {"runs": runs, "align": align, "space_before": space_before,
            "space_after": space_after, "line": line}


def P(text, size=12, bold=False, color=BODY, font=FONT, italic=False,
      align=None, space_before=None, space_after=None, line=None):
    return PR([R(text, size, bold, color, font, italic)], align,
              space_before, space_after, line)


def add_text(slide, x, y, w, h, blocks, anchor=MSO_ANCHOR.TOP, wrap=True):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, blk in enumerate(blocks):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if blk.get("align"):
            para.alignment = blk["align"]
        if blk.get("space_before"):
            para.space_before = Pt(blk["space_before"])
        if blk.get("line"):
            para.line_spacing = blk["line"]
        for r in blk["runs"]:
            run = para.add_run()
            run.text = r["t"]
            run.font.size = Pt(r["size"])
            run.font.bold = r["bold"]
            run.font.italic = r.get("italic", False)
            run.font.name = r["font"]
            run.font.color.rgb = r["color"]
    return box


def rect(slide, x, y, w, h, fill=PANEL, line=BORDER, line_w=0.75):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.adjustments[0] = 0.035
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_w)
    sh.shadow.inherit = False
    sh.text_frame.word_wrap = True
    return sh


def bar(slide, x, y, w, h, fill):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def chrome(prs, eyebrow, title, lead=None, page=None, accent=TEAL):
    s = new_slide(prs)
    bar(s, 0, 0, SW, 0.055, accent)
    add_text(s, ML, 0.40, CW, 0.26, [P(eyebrow.upper(), 9.5, True, accent, FONT_SB)])
    add_text(s, ML, 0.68, CW, 0.52, [P(title, 25, True, INK, FONT_SB, line=1.04)])
    y = 1.34
    if lead:
        add_text(s, ML, y, CW, 0.46, [P(lead, 11.5, False, MUTE, line=1.22)])
        y += 0.58
    if page:
        add_text(s, SW - MR - 1.0, SH - 0.46, 1.0, 0.22,
                 [P(str(page), 9, False, MUTE, align=PP_ALIGN.RIGHT)])
    return s, y + 0.06


def table(slide, x, y, w, col_w, data, row_h=0.34, head_h=0.36, size=10,
          head_size=10, col_bold=frozenset(), head_fill=INK):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w),
                                   Inches(head_h + (rows - 1) * row_h))
    tbl = shape.table
    tbl.first_row = True
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw)
    tbl.rows[0].height = Inches(head_h)
    for r in range(1, rows):
        tbl.rows[r].height = Inches(row_h)
    for r in range(rows):
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.text = str(data[r][c])
            cell.margin_left = Inches(0.09)
            cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                head_fill if r == 0 else (WHITE if r % 2 else PANEL))
            para = cell.text_frame.paragraphs[0]
            para.line_spacing = 1.0
            for run in para.runs:
                run.font.size = Pt(head_size if r == 0 else size)
                run.font.name = FONT_SB if (r == 0 or c in col_bold) else FONT
                run.font.bold = r == 0 or c in col_bold
                run.font.color.rgb = WHITE if r == 0 else INK
    return tbl


def card(slide, x, y, w, h, heading=None, lines=None, accent=TEAL,
         heading_size=12, body_size=10, fill=PANEL):
    rect(slide, x, y, w, h, fill=fill, line=BORDER)
    bar(slide, x, y, 0.055, h, accent)
    blocks = []
    if heading:
        blocks.append(P(heading, heading_size, True, INK, FONT_SB, line=1.10))
    for ln in (lines or []):
        blocks.append(P(ln, body_size, False, BODY, line=1.22, space_before=6))
    add_text(slide, x + 0.24, y + 0.16, w - 0.44, h - 0.30, blocks)


def picture(slide, path, x, y, w):
    """Place an image at a known width; measure its height, never assume it."""
    from PIL import Image
    if not path.exists():
        add_text(slide, x, y, w, 0.3,
                 [P(f"missing: {path.name}", 10, True, RED, FONT_SB)])
        return y + 0.4
    with Image.open(path) as im:
        h = w * im.height / im.width
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    return y + h


def banner(slide, y, text_runs, accent=TEAL, h=0.92, fill=PANEL2):
    rect(slide, ML, y, CW, h, fill=fill, line=accent, line_w=1.25)
    add_text(slide, ML + 0.26, y + 0.14, CW - 0.52, h - 0.26,
             [PR(text_runs, line=1.22)])


# --------------------------------------------------------------------------- #

def claim(slide, y, text, accent=TEAL):
    """The one sentence the presenter says out loud on this slide.

    Set large and given room, because a slide whose headline is the same size
    as its body has no headline -- the reader picks their own, and it is rarely
    the one that was meant.
    """
    bar(slide, ML, y, 0.10, 0.62, accent)
    add_text(slide, ML + 0.28, y - 0.04, CW - 0.28, 0.72,
             [P(text, 19, True, INK, FONT_SB, line=1.14)])
    return y + 0.86


def tile(slide, x, y, w, value, label, colour=TEAL, h=1.18):
    """One number, big, with what it is underneath."""
    rect(slide, x, y, w, h, fill=PANEL, line=BORDER)
    add_text(slide, x + 0.18, y + 0.16, w - 0.32, 0.52,
             [P(value, 24, True, colour, FONT_SB, line=1.0)])
    add_text(slide, x + 0.18, y + 0.74, w - 0.32, 0.36,
             [P(label, 9.5, False, BODY, line=1.16)])


def heading(slide, y, text):
    """A declarative section heading inside the slide body."""
    add_text(slide, ML, y, CW, 0.30, [P(text, 13, True, INK, FONT_SB, line=1.12)])
    return y + 0.38


def caption(slide, x, y, w, number, text):
    """A numbered figure or table caption, set below the object it labels."""
    add_text(slide, x, y, w, 0.44,
             [PR([R(number + "  ", 9, True, INK, FONT_SB),
                  R(text, 9, False, MUTE)], line=1.20)])
    return y + 0.48


def note(slide, x, y, w, h, label, lines, accent=TEAL, fill=PANEL):
    """A labelled block: Note, Limitation, Rationale."""
    rect(slide, x, y, w, h, fill=fill, line=BORDER)
    bar(slide, x, y, 0.045, h, accent)
    blocks = [P(label.upper(), 8.5, True, accent, FONT_SB)]
    for i, ln in enumerate(lines):
        blocks.append(P(ln, 9.5, False, BODY, line=1.24, space_before=6 if i else 5))
    add_text(slide, x + 0.24, y + 0.16, w - 0.44, h - 0.30, blocks)


def stat(slide, x, y, w, value, label, h=1.05):
    """A single figure with its definition beneath."""
    rect(slide, x, y, w, h, fill=PANEL, line=BORDER)
    add_text(slide, x + 0.18, y + 0.15, w - 0.32, 0.46,
             [P(value, 21, True, INK, FONT_SB, line=1.0)])
    add_text(slide, x + 0.18, y + 0.66, w - 0.32, 0.32,
             [P(label, 9, False, MUTE, line=1.16)])


def heading(slide, y, text):
    """A declarative section heading inside the slide body."""
    add_text(slide, ML, y, CW, 0.30, [P(text, 13, True, INK, FONT_SB, line=1.12)])
    return y + 0.38


def caption(slide, x, y, w, number, text):
    """A numbered figure or table caption, set below the object it labels."""
    add_text(slide, x, y, w, 0.44,
             [PR([R(number + "  ", 9, True, INK, FONT_SB),
                  R(text, 9, False, MUTE)], line=1.20)])
    return y + 0.48


def note(slide, x, y, w, h, label, lines, accent=TEAL, fill=PANEL):
    """A labelled block: Note, Limitation, Rationale."""
    rect(slide, x, y, w, h, fill=fill, line=BORDER)
    bar(slide, x, y, 0.045, h, accent)
    blocks = [P(label.upper(), 8.5, True, accent, FONT_SB)]
    for i, ln in enumerate(lines):
        blocks.append(P(ln, 9.5, False, BODY, line=1.24, space_before=6 if i else 5))
    add_text(slide, x + 0.24, y + 0.16, w - 0.44, h - 0.30, blocks)


def stat(slide, x, y, w, value, label, h=1.05):
    """A single figure with its definition beneath."""
    rect(slide, x, y, w, h, fill=PANEL, line=BORDER)
    add_text(slide, x + 0.18, y + 0.15, w - 0.32, 0.46,
             [P(value, 21, True, INK, FONT_SB, line=1.0)])
    add_text(slide, x + 0.18, y + 0.66, w - 0.32, 0.32,
             [P(label, 9, False, MUTE, line=1.16)])


def heading(slide, y, text):
    """A declarative section heading inside the slide body."""
    add_text(slide, ML, y, CW, 0.30, [P(text, 13, True, INK, FONT_SB, line=1.12)])
    return y + 0.38


def caption(slide, x, y, w, number, text):
    """A numbered figure or table caption, set below the object it labels."""
    add_text(slide, x, y, w, 0.44,
             [PR([R(number + "  ", 9, True, INK, FONT_SB),
                  R(text, 9, False, MUTE)], line=1.20)])
    return y + 0.48


def note(slide, x, y, w, h, label, lines, accent=TEAL, fill=PANEL):
    """A labelled block: Note, Limitation, Rationale."""
    rect(slide, x, y, w, h, fill=fill, line=BORDER)
    bar(slide, x, y, 0.045, h, accent)
    blocks = [P(label.upper(), 8.5, True, accent, FONT_SB)]
    for i, ln in enumerate(lines):
        blocks.append(P(ln, 9.5, False, BODY, line=1.24, space_before=6 if i else 5))
    add_text(slide, x + 0.24, y + 0.16, w - 0.44, h - 0.30, blocks)


def stat(slide, x, y, w, value, label, h=1.05):
    """A single figure with its definition beneath."""
    rect(slide, x, y, w, h, fill=PANEL, line=BORDER)
    add_text(slide, x + 0.18, y + 0.15, w - 0.32, 0.46,
             [P(value, 21, True, INK, FONT_SB, line=1.0)])
    add_text(slide, x + 0.18, y + 0.66, w - 0.32, 0.32,
             [P(label, 9, False, MUTE, line=1.16)])


def heading(slide, y, text):
    """A declarative section heading inside the slide body."""
    add_text(slide, ML, y, CW, 0.30, [P(text, 13, True, INK, FONT_SB, line=1.12)])
    return y + 0.38


def caption(slide, x, y, w, number, text):
    """A numbered figure or table caption, set below the object it labels."""
    add_text(slide, x, y, w, 0.44,
             [PR([R(number + "  ", 9, True, INK, FONT_SB),
                  R(text, 9, False, MUTE)], line=1.20)])
    return y + 0.48


def note(slide, x, y, w, h, label, lines, accent=TEAL, fill=PANEL):
    """A labelled block: Note, Limitation, Rationale."""
    rect(slide, x, y, w, h, fill=fill, line=BORDER)
    bar(slide, x, y, 0.045, h, accent)
    blocks = [P(label.upper(), 8.5, True, accent, FONT_SB)]
    for i, ln in enumerate(lines):
        blocks.append(P(ln, 9.5, False, BODY, line=1.24, space_before=6 if i else 5))
    add_text(slide, x + 0.24, y + 0.16, w - 0.44, h - 0.30, blocks)


def stat(slide, x, y, w, value, label, h=1.05):
    """A single figure with its definition beneath."""
    rect(slide, x, y, w, h, fill=PANEL, line=BORDER)
    add_text(slide, x + 0.18, y + 0.15, w - 0.32, 0.46,
             [P(value, 21, True, INK, FONT_SB, line=1.0)])
    add_text(slide, x + 0.18, y + 0.66, w - 0.32, 0.32,
             [P(label, 9, False, MUTE, line=1.16)])


def s01_title(prs):
    s = new_slide(prs)
    bar(s, 0, 0, SW, 0.09, TEAL)
    add_text(s, ML, 2.10, CW, 0.28,
             [P("PROJECT REVIEW", 10.5, True, TEAL, FONT_SB)])
    add_text(s, ML, 2.50, CW, 1.05,
             [P("ClassGraph: Classroom Engagement Estimation from Video",
                33, True, INK, FONT_SB, line=1.08)])
    bar(s, ML, 3.86, 1.4, 0.03, TEAL)
    add_text(s, ML, 4.08, CW * 0.74, 0.80,
             [P("A multi-model perception pipeline for per-student engagement "
                "analysis, evaluated against four review criteria.",
                13.5, False, BODY, line=1.34)])

    items = [("I", "Base Paper Implementation"),
             ("II", "Dataset Preparation and Preprocessing"),
             ("III", "Model Training and Evaluation"),
             ("IV", "Comparison with Base Paper")]
    cw = (CW - 3 * 0.24) / 4
    for i, (num, title) in enumerate(items):
        x = ML + i * (cw + 0.24)
        rect(s, x, 5.30, cw, 0.92, fill=PANEL, line=BORDER)
        add_text(s, x + 0.20, 5.48, cw - 0.36, 0.58,
                 [PR([R(num + "   ", 12, True, TEAL, FONT_SB),
                      R(title, 10.5, True, INK, FONT_SB)], line=1.16)])
    return s


def s02_scope(prs, page):
    s, y = chrome(prs, "Overview", "Scope and Summary of Findings",
                  "Each criterion is addressed in turn. Findings and their "
                  "limitations are stated together.", page)

    data = [
        ["Criterion", "Principal finding", "Stated limitation"],
        ["I · Base paper implementation",
         "ARG architecture replicated in full; two FER papers adopted at the level "
         "of method",
         "GCN weights untrained — no labelled dataset available"],
        ["II · Dataset preparation",
         "Class consolidation from 8 to 4 raised mAP@50 from 0.415 to 0.607",
         "Image size and batch also varied between runs"],
        ["III · Training and evaluation",
         "42 epochs, early stopping at epoch 27, per-class validation reported",
         "Validation split of 58 images is small"],
        ["IV · Comparison with base paper",
         "One quantitative comparison; three cases where comparison is not "
         "admissible",
         "Differing metrics permit comparison of ordering only"],
    ]
    tbl = table(s, ML, y, CW, [3.05, 5.20, 3.72], data,
                row_h=0.74, head_h=0.40, size=10, head_size=10, col_bold={0})
    for r in range(1, len(data)):
        tbl.cell(r, 2).text_frame.paragraphs[0].runs[0].font.color.rgb = BODY

    caption(s, ML, y + 0.40 + 4 * 0.74 + 0.16, CW, "Table 1.",
            "Summary of findings against the four review criteria.")

    note(s, ML, y + 0.40 + 4 * 0.74 + 0.72, CW, 0.96, "Note",
         ["All quantitative results in this deck are reproducible from the "
          "repository: training configuration from args.yaml, convergence curves "
          "from results.csv, and per-class metrics re-validated from the trained "
          "weights. Where a quantity could not be measured, it is reported as "
          "unmeasured rather than estimated."])
    return s


def s03_papers(prs, page):
    s, y = chrome(prs, "Criterion I — Base Paper Implementation",
                  "Base Papers and Their Role in the System",
                  "One paper contributes an architecture; two contribute methods. "
                  "The distinction is stated explicitly for each.", page)

    data = [
        ["Paper", "Venue", "Contribution to this work", "Level of adoption"],
        ["Learning Actor Relation Graphs for\nGroup Activity Recognition (Wu et al.)",
         "CVPR 2019",
         "Relation graph over detected people, with a\nGCN readout to a single group label",
         "Architecture,\nreplicated in full"],
        ["Robust Dynamic Facial Expression\nRecognition (Liu, Wang and Shen)",
         "IEEE T-BIOM\n2025",
         "Separation of a momentary signal from a\nsustained state",
         "Method, realised as a\n15-second rolling window"],
        ["Dynamic Objectives Learning for Facial\nExpression Recognition (Wen et al.)",
         "IEEE TMM\n2020",
         "Retention of easily-confused categories\nrather than forced assignment",
         "Method, realised as an\nabstaining category"],
    ]
    tbl = table(s, ML, y, CW, [4.10, 1.42, 4.20, 2.25], data,
                row_h=0.76, head_h=0.40, size=9.5, head_size=9.5, col_bold={0})
    for r in range(1, len(data)):
        run = tbl.cell(r, 3).text_frame.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.color.rgb = TEAL_D

    yy = caption(s, ML, y + 0.40 + 3 * 0.76 + 0.16, CW, "Table 2.",
                 "Base papers and the level at which each is adopted.")

    half = (CW - 0.30) / 2
    note(s, ML, yy + 0.10, half, 1.62, "Rationale",
         ["Two of the three base papers address facial expression recognition, "
          "which this system does not perform. They were selected for their "
          "treatment of a problem common to both domains: a per-frame signal is "
          "noisy, and a confident label assigned to an ambiguous sample is less "
          "useful than an explicit abstention."])
    note(s, ML + half + 0.30, yy + 0.10, half, 1.62, "Scope",
         ["Adoption at the level of method is reported as such throughout. No "
          "expression-recognition benchmark was reproduced, and no accuracy figure "
          "from either FER paper is claimed as a result of this work."])
    return s


def s04_arg(prs, page):
    s, y = chrome(prs, "Criterion I — Base Paper Implementation",
                  "ARG Architecture: Stage-by-Stage Implementation",
                  "Implemented in backend/group_activity.py and covered by 14 unit "
                  "tests.", page)

    data = [
        ["Stage", "Specification (Wu et al., 2019)", "Implementation", "Correspondence"],
        ["1", "Nodes are detected actors",
         "Nodes are students detected by YOLO11m", "Equivalent"],
        ["2", "Typed relations: appearance and position",
         "Cosine similarity over node features (top-k);\n"
         "centre distance normalised by group spread", "Equivalent"],
        ["3", "Relation-weighted adjacency, renormalised",
         "Kipf–Welling renormalisation, D^-1/2 (A+I) D^-1/2", "Equivalent"],
        ["4", "GCN message passing",
         "Two layers of tanh(A · H); one learned weight per\nrelation type", "Equivalent"],
        ["5", "Pooled readout to a graph-level label",
         "Mean pooling followed by a linear layer", "Equivalent"],
        ["6", "Predicts an activity class",
         "Predicts ordinal engagement: high, medium, low", "Deviation"],
        ["7", "Assumes complete actor detection",
         "Abstains below four students or above the missing-\nfeature threshold",
         "Deviation"],
    ]
    tbl = table(s, ML, y, CW, [0.72, 3.95, 5.20, 2.10], data,
                row_h=0.44, head_h=0.38, size=9.5, head_size=9.5, col_bold={0})
    for r in range(1, len(data)):
        run = tbl.cell(r, 3).text_frame.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.color.rgb = TEAL_D if data[r][3] == "Equivalent" else AMBER

    yy = caption(s, ML, y + 0.38 + 7 * 0.44 + 0.14, CW, "Table 3.",
                 "Correspondence between the ARG specification and this "
                 "implementation. Deviations are consequences of the differing "
                 "prediction target and of incomplete detection in classroom "
                 "footage.")

    note(s, ML, yy + 0.06, CW, 1.12, "Limitation",
         ["The network is implemented but untrained. Layer weights remain at their "
          "random initialisation, as training requires a labelled group-engagement "
          "corpus of approximately 3,000 clips with ordinal labels agreed between "
          "raters. No such corpus is publicly available for classroom footage and "
          "none could be constructed within this project. Accordingly, no output of "
          "this model is reported anywhere in this deck."],
         accent=RED, fill=RGBColor(0xFC, 0xF4, 0xF3))
    return s


def s05_dataset(prs, page):
    s, y = chrome(prs, "Criterion II — Dataset Preparation",
                  "Dataset Composition and Class Consolidation",
                  "Two annotated sources were merged and the label set reduced from "
                  "eight classes to four.", page)

    stats = [("8 → 4", "classes after consolidation"),
             ("423 → 877", "training images after merging"),
             ("6,091", "annotated boxes, four classes"),
             ("58", "validation images, held constant")]
    tw = (CW - 3 * 0.24) / 4
    for i, (v, lab) in enumerate(stats):
        stat(s, ML + i * (tw + 0.24), y, tw, v, lab)

    yy = y + 1.22
    data = [
        ["Retained class", "Train boxes", "Val boxes", "Removed class", "Disposition"],
        ["using_device", "2,204", "90", "handrise", "Moved to rule layer"],
        ["sleep", "1,512", "41", "look_forward", "Moved to rule layer"],
        ["read", "1,254", "49", "turn_head", "Moved to rule layer"],
        ["write", "1,121", "91", "stand", "Withdrawn — insufficient data"],
        ["Total", "6,091", "271", "—", "—"],
    ]
    tbl = table(s, ML, yy, CW, [2.45, 1.50, 1.30, 2.55, 4.17], data,
                row_h=0.36, head_h=0.38, size=9.5, head_size=9.5, col_bold={0})
    for c in range(5):
        tbl.cell(5, c).text_frame.paragraphs[0].runs[0].font.bold = True
    for r in range(1, 5):
        for c in (1, 2):
            run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
            run.font.name = MONO
            run.font.size = Pt(9)

    yy2 = caption(s, ML, yy + 0.38 + 5 * 0.36 + 0.14, CW, "Table 4.",
                  "Class consolidation. Retained classes and the disposition of "
                  "those removed.")

    half = (CW - 0.30) / 2
    note(s, ML, yy2 + 0.04, half, 1.30, "Rationale",
         ["The four removed classes each carried only tens of annotated boxes, "
          "insufficient for a detector to generalise. Three describe configurations "
          "measurable geometrically — wrist elevation and head pitch — and were "
          "reassigned to the rule layer rather than discarded from the system."])
    note(s, ML + half + 0.30, yy2 + 0.04, half, 1.30, "Method",
         ["The validation split was held constant at 58 images across both runs, so "
          "that the comparison in Figure 1 is measured on identical "
          "held-out data. The split is fixed by seed 0."])
    return s


def s06_ablation(prs, page):
    s, y = chrome(prs, "Criterion II — Dataset Preparation",
                  "Effect of Dataset Preparation on Detection Performance",
                  "Architecture, pretrained weights and validation split held "
                  "constant; only the training data differs.", page)

    picture(s, ASSETS / "dataset_ablation.png", ML, y, 7.05)
    caption(s, ML, y + 3.78, 7.05, "Figure 1.",
            "Validation metrics before and after class consolidation and source "
            "merging (n = 58 images).")

    x2 = ML + 7.05 + 0.34
    w2 = SW - MR - x2
    note(s, x2, y, w2, 1.95, "Observation",
         ["Mean average precision at IoU 0.50 rose from 0.415 to 0.607, an increase "
          "of 46 per cent. Recall rose proportionally more, from 0.381 to 0.586.",
          "The disproportionate movement in recall is consistent with the stated "
          "cause: the earlier model was not misclassifying detections but failing "
          "to produce them, as four of its classes had too few examples to fire."])
    note(s, x2, y + 2.13, w2, 1.75, "Limitation",
         ["Inference resolution was reduced from 960 to 640 pixels and batch size "
          "raised from 4 to 8 between the two runs, to accommodate 6.4 GB of "
          "available memory. The comparison is therefore not a single-variable "
          "ablation. Resolution was reduced, which would be expected to depress "
          "detection performance; the measured direction is nonetheless positive."],
         accent=AMBER)
    return s


def s06b_leakage(prs, page):
    """The duplicate finding, and what it did to the reported accuracy."""
    s, y = chrome(prs, "Criterion II - Dataset Preparation",
                  "Duplicate Detection and Its Effect on Reported Accuracy",
                  "A second annotated set was obtained for image-level scoring. "
                  "Its duplication rate changes what any accuracy figure on it "
                  "means.", page)

    stats = [("12,582", "images after exact dedup"),
             ("83.7%", "near-duplicates remaining"),
             ("678", "visually distinct groups"),
             ("36", "images under both labels")]
    tw = (CW - 3 * 0.24) / 4
    for i, (v, lab) in enumerate(stats):
        stat(s, ML + i * (tw + 0.24), y, tw, v, lab)

    yy = y + 1.22
    data = [
        ["Evaluation protocol", "Accuracy", "What it measures"],
        ["Random train/test split", "0.968",
         "Partly recall of near-copies present in both splits"],
        ["Split over duplicate groups", "0.664",
         "Generalisation to images the model has not seen"],
        ["Human rater vs folder labels", "0.840",
         "How far the labels themselves are agreed"],
    ]
    tbl = table(s, ML, yy, CW, [4.10, 2.05, 5.82], data,
                row_h=0.44, head_h=0.38, size=10, head_size=10, col_bold={0})
    for r in range(1, len(data)):
        run = tbl.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.bold = True
        run.font.color.rgb = RED if r == 1 else (TEAL_D if r == 2 else MUTE)

    yy2 = caption(s, ML, yy + 0.38 + 3 * 0.44 + 0.14, CW, "Table 5.",
                  "The same model and features under three protocols.")

    note(s, ML, yy2 + 0.04, CW, 1.30, "How the discrepancy was identified",
         ["The 0.968 was not accepted because it exceeded the rate at which a "
          "human rater agreed with the folder labels. A model has no basis for "
          "outperforming a person on the labels that person was judging, so "
          "the result was treated as evidence of leakage and tested with a "
          "perceptual hash, which found the duplication above.",
          "Copies are grouped rather than deleted: removing them would discard "
          "84% of the data, while grouping retains every image for training "
          "and leaves the evaluation set genuinely unseen."])
    return s


def s09b_scorer(prs, page):
    """The image-level scorer and its calibration."""
    s, y = chrome(prs, "Criterion III - Training and Evaluation",
                  "Image-Level Engagement Scorer",
                  "A second trained model, producing a 1-10 score from a single "
                  "frame. Evaluated out-of-fold over duplicate groups.", page)

    data = [
        ["Quantity", "Value", "Basis"],
        ["Training labels", "12,232", "binary, from the directory structure"],
        ["Human ordinal scores", "350", "five-point scale, stratified sample"],
        ["Accuracy, grouped split", "0.664", "leakage-free protocol"],
        ["ROC AUC, grouped split", "0.729", "same protocol"],
        ["Spearman rho vs human", "0.556", "out-of-fold, p = 7.8e-30"],
        ["Calibrated error", "0.84", "points of five"],
    ]
    tw = 7.15
    tbl = table(s, ML, y, tw, [2.75, 1.60, 2.80], data,
                row_h=0.40, head_h=0.38, size=10, head_size=10, col_bold={0})
    for r in range(1, len(data)):
        run = tbl.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.bold = True
        run.font.color.rgb = TEAL_D

    caption(s, ML, y + 0.38 + 6 * 0.40 + 0.14, tw, "Table 8.",
            "Image-level scorer, trained on binary labels and calibrated "
            "against human ordinal judgement.")

    x2 = ML + tw + 0.30
    w2 = SW - MR - x2
    note(s, x2, y, w2, 2.00, "Method",
         ["The directory labels are binary, so the model emits a probability "
          "rather than a score. Isotonic regression maps that probability onto "
          "the human five-point scale, and the five-point value is expressed "
          "as 1-10 only at the final step.",
          "The scored images contribute to training as well as to validation, "
          "through out-of-fold prediction: each fold is calibrated by a model "
          "that did not see it."])
    note(s, x2, y + 2.18, w2, 2.00, "Alternative tested",
         ["Fitting a regressor directly to the 350 ordinal scores was measured "
          "against calibrating the classifier, on the same features and the "
          "same folds.",
          "Direct regression reached rho 0.353 against 0.556. The larger set of "
          "binary labels carries more usable signal than the smaller set of "
          "ordinal ones, despite the latter being individually richer."],
         accent=AMBER)

    note(s, ML, y + 3.10, tw, 1.08, "Limitation",
         ["The resulting distribution is bimodal, with few images scored near "
          "the middle of the range. The underlying decision remains binary and "
          "the calibration spreads it across a scale; the score is ordered and "
          "validated, but it is not a uniform ten-point continuum."],
         accent=RED)
    return s


def s07_training(prs, page):
    s, y = chrome(prs, "Criterion III — Training and Evaluation",
                  "Training Configuration",
                  "Fine-tuning from COCO-pretrained weights with no layers frozen. "
                  "Full configuration in Appendix A.", page)

    stats = [("42 / 60", "epochs run of those requested"),
             ("27", "best epoch, by mAP@50-95"),
             ("20.1 M", "parameters, all updated"),
             ("23 min", "wall-clock, RTX 4050")]
    tw = (CW - 3 * 0.24) / 4
    for i, (v, lab) in enumerate(stats):
        stat(s, ML + i * (tw + 0.24), y, tw, v, lab, h=0.96)

    yy = y + 1.12
    data = [
        ["Parameter", "Value", "Basis for the value"],
        ["Initial weights", "yolo11m.pt", "COCO-pretrained; random initialisation is "
                                          "not viable at 877 images"],
        ["Frozen layers", "none", "The target domain differs sufficiently that a "
                                  "fixed backbone is inappropriate"],
        ["Epochs / patience", "60 / 15", "Terminated at 42 by early stopping"],
        ["Inference size", "640", "Determined by available memory; 960 required "
                                  "9.05 GB against 6.4 GB available"],
        ["Batch size", "8", "Largest admissible at 640 pixels"],
        ["Optimiser and schedule", "Ultralytics defaults",
         "Not searched; see the note below"],
    ]
    tbl = table(s, ML, yy, CW, [2.40, 1.90, 7.67], data,
                row_h=0.38, head_h=0.36, size=9.5, head_size=9.5, col_bold={0})
    for r in range(1, len(data)):
        run = tbl.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.size = Pt(9)
        run.font.color.rgb = TEAL_D
        run.font.bold = True

    yy2 = caption(s, ML, yy + 0.36 + 6 * 0.38 + 0.12, CW, "Table 6.",
                  "Principal training parameters and the basis on which each was "
                  "set.")

    note(s, ML, yy2 + 0.02, CW, 0.90, "Note on hyperparameter selection",
         ["No hyperparameter search was conducted. With 877 training and 58 "
          "validation images, a search would optimise against sampling noise and "
          "report the resulting configuration as a finding. The two values that "
          "were chosen — inference size and batch size — were determined by memory "
          "capacity, and are reported as such rather than as tuned quantities."])
    return s


def s08_curves(prs, page):
    s, y = chrome(prs, "Criterion III — Training and Evaluation",
                  "Convergence and Early Stopping",
                  "Validation loss and detection quality across the 42 epochs "
                  "completed.", page)

    half = (CW - 0.34) / 2
    b1 = picture(s, ASSETS / "train_loss.png", ML, y, half)
    b2 = picture(s, ASSETS / "train_metrics.png", ML + half + 0.34, y, half)
    yb = max(b1, b2) + 0.10
    caption(s, ML, yb, half, "Figure 2.",
            "Validation loss by epoch, by loss term.")
    yc = caption(s, ML + half + 0.34, yb, half, "Figure 3.",
                 "Mean average precision by epoch, at two IoU regimes.")

    third = (CW - 2 * 0.26) / 3
    notes = [
        ("Measurement", TEAL,
         "Validation loss is plotted rather than training loss. Training loss "
         "decreases by construction; the quantity of interest is performance on "
         "images excluded from training."),
        ("Presentation", TEAL,
         "Loss and mean average precision are plotted separately. A shared "
         "secondary axis would permit an apparent relationship arising from the "
         "choice of scaling rather than from the data."),
        ("Limitation", AMBER,
         "With 58 validation images, a single difficult image displaces mean "
         "average precision measurably. The peak value is reported, but the "
         "sustained level is approximately 0.55 to 0.60."),
    ]
    for i, (label, accent, body) in enumerate(notes):
        note(s, ML + i * (third + 0.26), yc + 0.06, third, 1.32, label, [body],
             accent=accent)
    return s


def s09_perclass(prs, page):
    s, y = chrome(prs, "Criterion III — Training and Evaluation",
                  "Per-Class Evaluation Results",
                  "Re-validated from the trained weights for this review, rather "
                  "than reproduced from the training log.", page)

    data = [
        ["Class", "mAP@50", "mAP@50-95", "Precision", "Recall", "Train boxes"],
        ["write", "0.767", "0.331", "0.723", "0.681", "1,121"],
        ["using_device", "0.720", "0.336", "0.749", "0.700", "2,204"],
        ["sleep", "0.521", "0.263", "0.670", "0.495", "1,512"],
        ["read", "0.419", "0.143", "0.455", "0.469", "1,254"],
        ["All classes", "0.607", "0.268", "0.649", "0.586", "6,091"],
    ]
    tw = 6.95
    tbl = table(s, ML, y, tw, [1.75, 1.10, 1.25, 1.10, 0.95, 0.80], data,
                row_h=0.40, head_h=0.38, size=10, head_size=9.5, col_bold={0})
    for r in range(1, len(data)):
        for c in range(1, 6):
            run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
            run.font.name = MONO
            run.font.size = Pt(9.5)
        if r == 5:
            for c in range(6):
                tbl.cell(r, c).text_frame.paragraphs[0].runs[0].font.bold = True

    caption(s, ML, y + 0.38 + 5 * 0.40 + 0.14, tw, "Table 7.",
            "Per-class validation metrics (n = 58 images).")

    x2 = ML + tw + 0.32
    w2 = SW - MR - x2
    note(s, x2, y, w2, 2.00, "Analysis of the weakest class",
         ["The read class attains mAP@50 of 0.419 against 0.767 for write, despite "
          "carrying more annotated boxes. The deficit is therefore not attributable "
          "to data volume.",
          "Reading and writing are distinguished only by hand activity. Comparable "
          "published work reports 57.8 per cent on writing recognition using a "
          "temporal model."],
         accent=AMBER)
    note(s, x2, y + 2.18, w2, 2.00, "Treatment in the output",
         ["Where a book is detected and the hands are not visible, the reported "
          "action is 'reading or writing', a single label spanning both classes.",
          "Where the hands are visible, writing is reported with confidence "
          "'inferred' rather than 'direct'. The least reliable measurement is "
          "accordingly the one subject to the most qualification."])

    note(s, ML, y + 2.90, tw, 1.28, "Role within the system",
         ["The behaviour model is a secondary signal. Actions are determined first "
          "by geometric evidence — object overlap, wrist position and head pitch — "
          "which requires no training data and is inspectable. Where the two "
          "disagree, geometry governs and both are recorded."])
    return s


def s10_comparison(prs, page):
    s, y = chrome(prs, "Criterion IV — Comparison with Base Paper",
                  "Comparison with Base Papers",
                  "One quantitative comparison is admissible. Three are not, for "
                  "reasons stated individually.", page)

    data = [
        ["Source", "Reported result", "Corresponding result here", "Admissibility"],
        ["Our fine-tuned detector\n(YOLO11m, four classes)",
         "Published work on writing\nrecognition: 57.8%",
         "write mAP@50 0.767;\nread mAP@50 0.419",
         "Admissible for\nordering only"],
        ["Robust Dynamic FER\n(Liu, Wang and Shen, 2025)",
         "Accuracy on DFEW and\nFERV39k",
         "Not reproduced — adopted at the\nlevel of method",
         "Not applicable"],
        ["Dynamic Objectives Learning\n(Wen et al., 2020)",
         "Accuracy on posed and\nin-the-wild sets",
         "Not reproduced — adopted at the\nlevel of method",
         "Not applicable"],
        ["ARG\n(Wu et al., CVPR 2019)",
         "Volleyball 92.3%;\nCollective Activity 91.0%",
         "None — the network is untrained",
         "Not available"],
    ]
    tbl = table(s, ML, y, CW, [3.10, 2.85, 3.70, 2.32], data,
                row_h=0.70, head_h=0.38, size=9.5, head_size=9.5, col_bold={0})
    for r in range(1, len(data)):
        run = tbl.cell(r, 3).text_frame.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.color.rgb = TEAL_D if r == 1 else (RED if r == 4 else MUTE)

    yy = caption(s, ML, y + 0.38 + 4 * 0.70 + 0.14, CW, "Table 9.",
                 "Admissibility of comparison against each base paper.")

    half = (CW - 0.30) / 2
    note(s, ML, yy + 0.04, half, 1.72, "Basis of the admissible comparison",
         ["The published figure is an accuracy on a writing-recognition benchmark; "
          "ours is mean average precision on a 58-image validation split. The "
          "metric, the data and the task definition all differ.",
          "What the comparison supports is the ordering: both find writing "
          "difficult, and the read–write disparity observed here reproduces the "
          "difficulty reported in the literature."])
    note(s, ML + half + 0.30, yy + 0.04, half, 1.72, "Requirement for the ARG comparison",
         ["A labelled group-engagement corpus of approximately 3,000 clips with "
          "ordinal labels agreed between raters. No such corpus is publicly "
          "available for classroom footage.",
          "The architecture, input contract, abstention rule and test suite are "
          "complete. The outstanding requirement is data rather than "
          "implementation."],
         accent=RED)
    return s


def s11_summary(prs, page):
    s, y = chrome(prs, "Conclusion", "Summary of Findings and Limitations",
                  "Each finding is paired with the limitation that qualifies it.",
                  page)

    data = [
        ["Criterion", "Finding", "Evidence", "Limitation"],
        ["I · Base paper\nimplementation",
         "ARG replicated stage for stage;\ntwo FER papers adopted as method",
         "backend/group_activity.py;\n14 unit tests",
         "Network untrained; no\nlabelled corpus available"],
        ["II · Dataset\npreparation",
         "Consolidation raised mAP@50 to\n0.607; 83.7% duplication found\nin the second set",
         "args.yaml, Figure 1,\nTable 6",
         "Resolution and batch also\nvaried between runs"],
        ["III · Training and\nevaluation",
         "Behaviour detector mAP@50 0.607;\nimage scorer rho 0.556 vs human",
         "results.csv, Table 7;\nre-validated from weights",
         "Both validation sets small\n(58 images, 350 scores)"],
        ["IV · Comparison with\nbase paper",
         "One admissible comparison;\nthree inadmissible, each explained",
         "Table 8 against published\n57.8% on writing",
         "Differing metrics permit\nordering only"],
    ]
    tbl = table(s, ML, y, CW, [2.55, 3.65, 2.95, 2.82], data,
                row_h=0.72, head_h=0.38, size=9.5, head_size=9.5, col_bold={0})

    yy = caption(s, ML, y + 0.38 + 4 * 0.72 + 0.14, CW, "Table 10.",
                 "Findings and limitations by criterion.")

    note(s, ML, yy + 0.06, CW, 1.10, "Reproducibility",
         ["Every quantity reported in this deck is produced by a command in the "
          "repository: tools/model_facts.py --validate for model and per-class "
          "metrics, tools/make_review_charts.py for Figures 1 to 3, and "
          "tools/train_behaviour.py for the training run itself. Quantities that "
          "could not be measured are reported as unmeasured."])
    return s


def s12_appendix_a(prs, page):
    s, y = chrome(prs, "Appendix A", "Complete Training Configuration",
                  "As recorded in runs/behaviour/merged4_aug/args.yaml.", page, MUTE)

    half = (CW - 0.30) / 2
    cfg = [
        ["Parameter", "Value", "Source"],
        ["model", "yolo11m.pt", "COCO-pretrained"],
        ["freeze", "null", "all layers trainable"],
        ["epochs", "60", "42 completed"],
        ["patience", "15", "early stopping"],
        ["imgsz", "640", "memory-constrained"],
        ["batch", "8", "memory-constrained"],
        ["optimizer", "auto (SGD)", "default"],
        ["lr0", "0.01", "default"],
        ["lrf", "0.01", "default"],
        ["momentum", "0.937", "default"],
        ["weight_decay", "0.0005", "default"],
        ["warmup_epochs", "3.0", "default"],
        ["amp", "true", "required to fit"],
        ["seed", "0", "fixed split"],
    ]
    tbl = table(s, ML, y, half, [2.10, 1.65, 2.17], cfg, row_h=0.275, head_h=0.32,
                size=9.5, head_size=9.5, col_bold={0}, head_fill=MUTE)
    for r in range(1, len(cfg)):
        run = tbl.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.size = Pt(9)
        run.font.color.rgb = TEAL_D
        run.font.bold = True
    caption(s, ML, y + 0.32 + 14 * 0.275 + 0.12, half, "Table A1.",
            "Training parameters.")

    x2 = ML + half + 0.30
    loss = [
        ["Loss term", "Weight", "Quantity penalised"],
        ["box", "7.5", "Bounding-box regression error (CIoU)"],
        ["cls", "0.5", "Class confidence error (BCE)"],
        ["dfl", "1.5", "Distribution focal loss on box edges"],
    ]
    tbl2 = table(s, x2, y, half, [1.85, 1.30, 2.77], loss, row_h=0.32, head_h=0.34,
                 size=9.5, head_size=9.5, col_bold={0}, head_fill=MUTE)
    for r in range(1, 4):
        run = tbl2.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.bold = True
        run.font.color.rgb = TEAL_D
    yy = caption(s, x2, y + 0.34 + 3 * 0.32 + 0.12, half, "Table A2.",
                 "Loss term weights.")

    note(s, x2, yy + 0.06, half, 1.95, "Data splits",
         ["877 training images and 58 validation images, fixed by seed 0.",
          "No test split was reserved. With 58 validation images, partitioning a "
          "third set would render both unusable. All figures reported in this deck "
          "are validation figures and are labelled as such."])
    return s


def s13_appendix_b(prs, page):
    s, y = chrome(prs, "Appendix B", "Augmentation Configuration",
                  "Settings applied during fine-tuning, with the variation each is "
                  "intended to model.", page, MUTE)

    aug = [
        ["Setting", "Value", "Variation modelled"],
        ["mosaic", "1.0", "Four images per sample; greater context per step"],
        ["close_mosaic", "10", "Final ten epochs without mosaic"],
        ["fliplr", "0.5", "Lateral position within the room"],
        ["scale", "0.5", "Apparent size between front and rear rows"],
        ["translate", "0.1", "Variation in camera framing"],
        ["hsv_h / hsv_s / hsv_v", "0.015 / 0.7 / 0.4", "Illumination, daylight to fluorescent"],
        ["erasing", "0.4", "Occlusion by furniture and by other students"],
        ["auto_augment", "randaugment", "Applied to classification crops"],
        ["degrees / shear / perspective", "0 / 0 / 0", "Disabled — not observable in fixed-camera footage"],
        ["flipud", "0", "Disabled — not physically realisable"],
        ["mixup / copy_paste", "0 / 0", "Disabled — composited students are not a valid scene"],
    ]
    tw = 8.05
    tbl = table(s, ML, y, tw, [2.65, 1.85, 3.55], aug, row_h=0.325, head_h=0.35,
                size=9.5, head_size=9.5, col_bold={0}, head_fill=MUTE)
    for r in range(1, len(aug)):
        run = tbl.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.size = Pt(9)
        run.font.color.rgb = MUTE if r >= 9 else TEAL_D
        run.font.bold = r < 9
    caption(s, ML, y + 0.35 + 11 * 0.325 + 0.12, tw, "Table B1.",
            "Augmentation settings. Disabled settings are listed explicitly, as "
            "the configuration is defined as much by exclusion as by inclusion.")

    x2 = ML + tw + 0.30
    w2 = SW - MR - x2
    note(s, x2, y, w2, 2.05, "Mosaic and its termination",
         ["Mosaic composition improves early training by presenting more context "
          "per gradient step.",
          "A model exposed only to four-image composites is not calibrated on "
          "single frames, so the setting is withdrawn for the final ten epochs and "
          "training concludes on unmodified classroom images."])
    note(s, x2, y + 2.23, w2, 1.80, "Geometric augmentations",
         ["Rotation, shear, perspective and vertical flip are disabled. A "
          "fixed-position classroom camera does not produce such views, and "
          "training against them allocates capacity to variation that does not "
          "occur in deployment."])
    return s


def main():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    s01_title(prs)
    s02_scope(prs, 2)
    s03_papers(prs, 3)
    s04_arg(prs, 4)
    s05_dataset(prs, 5)
    s06_ablation(prs, 6)
    s06b_leakage(prs, 7)
    s07_training(prs, 8)
    s08_curves(prs, 9)
    s09_perclass(prs, 10)
    s09b_scorer(prs, 11)
    s10_comparison(prs, 12)
    s11_summary(prs, 13)
    s12_appendix_a(prs, 14)
    s13_appendix_b(prs, 15)

    prs.save(str(OUT))
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
