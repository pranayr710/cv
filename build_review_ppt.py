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


def s01_title(prs):
    s = new_slide(prs)
    bar(s, 0, 0, SW, 0.09, TEAL)
    add_text(s, ML, 1.95, CW, 0.3, [P("PROJECT REVIEW", 11, True, TEAL, FONT_SB)])
    add_text(s, ML, 2.33, CW, 1.15,
             [P("ClassGraph \u2014 classroom engagement from video",
                38, True, INK, FONT_SB, line=1.03)])
    add_text(s, ML, 3.72, CW * 0.76, 0.6,
             [P("What we built, what we measured, and what we are not claiming.",
                15, False, BODY, line=1.30)])
    bar(s, ML, 4.62, 1.5, 0.03, TEAL)

    rows = [("1", "Base paper", "One architecture replicated,\ntwo methods adopted"),
            ("2", "Dataset", "8 classes \u2192 4 doubled the data\nand raised mAP 46%"),
            ("3", "Training", "42 epochs, mAP@50 0.607,\nhonest per-class numbers"),
            ("4", "Comparison", "One real comparison,\nand why the others cannot be")]
    cw = (CW - 3 * 0.24) / 4
    for i, (num, title, sub) in enumerate(rows):
        x = ML + i * (cw + 0.24)
        rect(s, x, 4.92, cw, 1.42, fill=PANEL, line=BORDER)
        bar(s, x, 4.92, cw, 0.05, TEAL)
        add_text(s, x + 0.20, 5.12, cw - 0.36, 1.10,
                 [PR([R(num + "   ", 16, True, TEAL, FONT_SB),
                      R(title, 13, True, INK, FONT_SB)]),
                  P(sub, 10, False, MUTE, line=1.20, space_before=6)])
    return s


# --------------------------------------------------------------- criterion 1

def s02_papers(prs, page):
    s, y = chrome(prs, "Criterion 1 \u2014 Base Paper Implementation",
                  "Three papers, two different kinds of borrowing", None, page)
    y = claim(s, y, "We replicated one architecture. The other two papers gave us "
                    "methods, not networks \u2014 and we say which is which.")

    items = [
        ("ARG", "Wu et al., CVPR 2019", GREEN,
         "Learning Actor Relation Graphs for Group Activity Recognition",
         "ARCHITECTURE REPLICATED",
         "People as graph nodes, two typed relations, a 2-layer GCN, one label for "
         "the whole group. Built stage for stage in backend/group_activity.py."),
        ("FER-1", "Liu, Wang & Shen, 2025", TEAL,
         "Robust Dynamic Facial Expression Recognition",
         "METHOD ADOPTED",
         "Their idea: separate a momentary signal from a sustained state. Ours: a "
         "15-second rolling window, so one glance away is not distraction."),
        ("FER-2", "Wen et al., IEEE TMM 2020", TEAL,
         "Dynamic Objectives Learning for Facial Expression Recognition",
         "METHOD ADOPTED",
         "Their idea: keep easily-confused categories apart instead of forcing a "
         "class. Ours: \u201chead down, no device\u201d stays its own category "
         "rather than being guessed as engaged or distracted."),
    ]
    third = (CW - 2 * 0.28) / 3
    for i, (tag, cite, accent, title, status, body) in enumerate(items):
        x = ML + i * (third + 0.28)
        rect(s, x, y, third, 3.05, fill=WHITE, line=BORDER)
        bar(s, x, y, third, 0.05, accent)
        add_text(s, x + 0.24, y + 0.22, third - 0.48, 0.28,
                 [PR([R(tag + "   ", 12, True, accent, FONT_SB),
                      R(cite, 9, False, MUTE)])])
        add_text(s, x + 0.24, y + 0.60, third - 0.48, 0.62,
                 [P(title, 11.5, True, INK, FONT_SB, line=1.12)])
        add_text(s, x + 0.24, y + 1.30, third - 0.48, 0.24,
                 [P(status, 9, True, accent, FONT_SB)])
        add_text(s, x + 0.24, y + 1.64, third - 0.48, 1.24,
                 [P(body, 10, False, BODY, line=1.24)])

    banner(s, y + 3.26,
           [R("Why facial-expression papers for a system that never classifies "
              "emotion:  ", 11, True, TEAL_D, FONT_SB),
            R("we took their method, not their task. Both solve the problem we "
              "actually have \u2014 a per-frame signal is noisy, and a confident "
              "label on a hard sample is worse than no label.", 11, False, INK)],
           h=0.88)
    return s


def s03_arg(prs, page):
    s, y = chrome(prs, "Criterion 1 \u2014 Base Paper Implementation",
                  "ARG, in three stages", None, page, GREEN)
    y = claim(s, y, "The architecture is built exactly as the paper specifies. "
                    "The weights were never trained \u2014 and that is a data gap, "
                    "not an engineering one.", GREEN)

    stages = [
        ("1", "Build the relation graph",
         "Each student is a node. Two typed relations, as ARG specifies:\n"
         "\u2022  appearance \u2014 cosine similarity, top-k neighbours only\n"
         "\u2022  position \u2014 centre distance, normalised by group spread"),
        ("2", "Pass messages over it",
         "Adjacencies summed, then renormalised with the exact Kipf & Welling "
         "rule D^-1/2 (A+I) D^-1/2.\nEach relation gets its own learned weight; "
         "two rounds of tanh(A \u00b7 H)."),
        ("3", "Read out one label",
         "Node features mean-pooled, then a linear layer to the label set.\n"
         "Our labels are ordinal engagement (high / medium / low) rather than "
         "ARG's activity classes."),
    ]
    third = (CW - 2 * 0.28) / 3
    for i, (num, title, body) in enumerate(stages):
        x = ML + i * (third + 0.28)
        rect(s, x, y, third, 1.98, fill=WHITE, line=BORDER)
        add_text(s, x + 0.24, y + 0.20, third - 0.48, 0.34,
                 [PR([R(num + "   ", 16, True, GREEN, FONT_SB),
                      R(title, 12.5, True, INK, FONT_SB)])])
        add_text(s, x + 0.24, y + 0.68, third - 0.48, 1.16,
                 [P(ln, 9.5, False, BODY, line=1.24) for ln in body.split("\n")])

    yy = y + 2.20
    half = (CW - 0.30) / 2
    card(s, ML, yy, half, 1.62,
         heading="What is real: everything except the weights",
         lines=["Graph construction, the renormalisation, the abstention rule and "
                "the degenerate-input handling are all implemented and covered by "
                "14 passing tests.",
                "Below 4 students, or with too many missing features, it returns "
                "\u201cno answer\u201d with a reason instead of a guess."],
         accent=GREEN, heading_size=11.5, body_size=9.5, fill=PANEL2)
    card(s, ML + half + 0.30, yy, half, 1.62,
         heading="What is missing: a labelled dataset",
         lines=["torch.nn.Linear initialises randomly. Nothing ever trained those "
                "weights, because training needs ~3,000 clips labelled high / "
                "medium / low by agreeing human raters.",
                "No such set exists publicly for classrooms, and we could not build "
                "one. So no number from this model appears anywhere in this deck."],
         accent=RED, heading_size=11.5, body_size=9.5, fill=PANEL)
    return s


# --------------------------------------------------------------- criterion 2

def s04_dataset(prs, page):
    s, y = chrome(prs, "Criterion 2 \u2014 Dataset Preparation",
                  "Eight classes became four", None, page, GREEN)
    y = claim(s, y, "Four of the eight classes were too sparse to learn. "
                    "Dropping them to the rule layer doubled the data behind every "
                    "class that remained.", GREEN)

    tiles = [("8 \u2192 4", "classes, after merging", GREEN),
             ("423 \u2192 877", "training images", GREEN),
             ("6,091", "labelled boxes across 4 classes", TEAL),
             ("58", "validation images, held fixed", AMBER)]
    tw = (CW - 3 * 0.24) / 4
    for i, (v, lab, c) in enumerate(tiles):
        tile(s, ML + i * (tw + 0.24), y, tw, v, lab, c)

    yy = y + 1.40
    half = (CW - 0.30) / 2
    card(s, ML, yy, half, 2.10,
         heading="The four we removed \u2014 and where they went",
         lines=["handrise, look_forward, turn_head and stand each had only a few "
                "dozen boxes. A detector trained on that memorises those boxes; it "
                "does not learn the class.",
                "Three of them are better measured by geometry anyway. Wrist above "
                "shoulder is a raised hand; head pitch is a bowed head. They moved "
                "to the rule layer \u2014 removed from the model, not from the "
                "system."],
         accent=AMBER, heading_size=11.5, body_size=10, fill=PANEL)
    card(s, ML + half + 0.30, yy, half, 2.10,
         heading="The four we kept",
         lines=["using_device 2,204 boxes  \u00b7  sleep 1,512  \u00b7  "
                "read 1,254  \u00b7  write 1,121.",
                "Two source sets were merged to get there, taking training images "
                "from 423 to 877.",
                "The validation split was deliberately left untouched at 58 images, "
                "so the before-and-after on the next slide is measured on exactly "
                "the same held-out data."],
         accent=GREEN, heading_size=11.5, body_size=10, fill=PANEL2)
    return s


def s05_ablation(prs, page):
    s, y = chrome(prs, "Criterion 2 \u2014 Dataset Preparation",
                  "What that bought", None, page, GREEN)
    y = claim(s, y, "Same model, same weights, same held-out split. Only the data "
                    "changed \u2014 and mAP@50 rose 46%.", GREEN)

    picture(s, ASSETS / "dataset_ablation.png", ML, y, 7.05)
    x2 = ML + 7.05 + 0.34
    w2 = SW - MR - x2
    card(s, x2, y, w2, 1.85,
         heading="Read the recall bar first",
         lines=["0.381 \u2192 0.586. Recall moving most is the signature that makes "
                "this believable: the old model was not confused about what it saw, "
                "it was failing to see at all, because four of its classes had too "
                "few examples to ever fire."],
         accent=GREEN, heading_size=11.5, body_size=10, fill=PANEL2)
    card(s, x2, y + 2.03, w2, 1.68,
         heading="The confound, before anyone asks",
         lines=["Image size also dropped 960 \u2192 640 and batch rose 4 \u2192 8, "
                "to fit a 6.4 GB card. So this is not a clean single-variable test.",
                "It still points the right way: resolution went DOWN, which should "
                "hurt a detector, and the score rose anyway."],
         accent=AMBER, heading_size=11.5, body_size=10, fill=PANEL)
    return s


# --------------------------------------------------------------- criterion 3

def s06_training(prs, page):
    s, y = chrome(prs, "Criterion 3 \u2014 Training & Evaluation",
                  "How it was trained", None, page, AMBER)
    y = claim(s, y, "Fine-tuned from COCO weights with every layer unfrozen. "
                    "Nothing was hyperparameter-searched \u2014 and that is "
                    "deliberate.", AMBER)

    tiles = [("42", "epochs run, of 60 requested", AMBER),
             ("27", "best epoch \u2014 early stopped", GREEN),
             ("20.1 M", "parameters, all updated", TEAL),
             ("23 min", "on an RTX 4050 laptop", TEAL)]
    tw = (CW - 3 * 0.24) / 4
    for i, (v, lab, c) in enumerate(tiles):
        tile(s, ML + i * (tw + 0.24), y, tw, v, lab, c)

    yy = y + 1.40
    third = (CW - 2 * 0.26) / 3
    notes = [
        ("Started from COCO, froze nothing", TEAL,
         "yolo11m.pt as the starting point, freeze: null. The backbone already "
         "knows edges and human shape; only the class meanings had to change, so "
         "all 20.1 M parameters were allowed to move."),
        ("Defaults, on purpose", RED,
         "Every optimiser value is the Ultralytics default. With 877 training "
         "images and 58 validation images, a hyperparameter search would tune "
         "against noise and report the winner as a finding."),
        ("The two we did choose", AMBER,
         "imgsz 640 and batch 8, chosen by what fits in 6.4 GB of VRAM. At 960 the "
         "run wanted 9.05 GB and slowed to an estimated 18 hours. We say so rather "
         "than implying they were optimised."),
    ]
    for i, (title, accent, body) in enumerate(notes):
        x = ML + i * (third + 0.26)
        rect(s, x, yy, third, 2.05, fill=PANEL, line=BORDER)
        bar(s, x, yy, 0.05, 2.05, accent)
        add_text(s, x + 0.24, yy + 0.18, third - 0.44, 1.70,
                 [P(title, 11.5, True, INK, FONT_SB, line=1.12),
                  P(body, 10, False, BODY, line=1.24, space_before=7)])
    add_text(s, ML, SH - 0.44, CW, 0.22,
             [P("Full hyperparameter and augmentation tables are in the appendix.",
                9, False, MUTE)])
    return s


def s07_curves(prs, page):
    s, y = chrome(prs, "Criterion 3 \u2014 Training & Evaluation",
                  "Loss and detection quality", None, page, AMBER)
    y = claim(s, y, "It learned, then it stopped learning \u2014 and we stopped "
                    "training. Peak at epoch 27, mAP@50 0.607.", AMBER)

    half = (CW - 0.34) / 2
    b1 = picture(s, ASSETS / "train_loss.png", ML, y, half)
    b2 = picture(s, ASSETS / "train_metrics.png", ML + half + 0.34, y, half)

    yy = max(b1, b2) + 0.24
    third = (CW - 2 * 0.26) / 3
    notes = [
        ("This is validation loss", TEAL,
         "Training loss always falls. The question worth answering is whether it "
         "fell on images the model never saw, so that is what is plotted."),
        ("Two charts, not one", GREEN,
         "Loss and mAP share an x-axis and nothing else. A second y-scale would "
         "let you read a relationship off where two lines happen to cross."),
        ("The curve is noisy \u2014 that is the split", AMBER,
         "58 validation images means one hard image visibly moves mAP. We quote the "
         "peak, but the honest read is a plateau around 0.55\u20130.60."),
    ]
    for i, (title, accent, body) in enumerate(notes):
        x = ML + i * (third + 0.26)
        rect(s, x, yy, third, 1.22, fill=PANEL, line=BORDER)
        bar(s, x, yy, 0.045, 1.22, accent)
        add_text(s, x + 0.22, yy + 0.14, third - 0.40, 0.96,
                 [P(title, 10.5, True, INK, FONT_SB, line=1.10),
                  P(body, 9.5, False, BODY, line=1.20, space_before=5)])
    return s


def s08_perclass(prs, page):
    s, y = chrome(prs, "Criterion 3 \u2014 Training & Evaluation",
                  "Per-class results", None, page, AMBER)
    y = claim(s, y, "Three classes work. \u201cread\u201d does not \u2014 and the "
                    "pipeline was already designed around that.", AMBER)

    data = [
        ["Class", "mAP@50", "Precision", "Recall", "Train boxes"],
        ["write", "0.767", "0.723", "0.681", "1,121"],
        ["using_device", "0.720", "0.749", "0.700", "2,204"],
        ["sleep", "0.521", "0.670", "0.495", "1,512"],
        ["read", "0.419", "0.455", "0.469", "1,254"],
        ["all", "0.607", "0.649", "0.586", "6,091"],
    ]
    tw = 6.55
    tbl = table(s, ML, y, tw, [1.75, 1.25, 1.25, 1.10, 1.20], data,
                row_h=0.42, head_h=0.40, size=10.5, head_size=10,
                col_bold={0}, head_fill=AMBER)
    for r in range(1, len(data)):
        for c in range(1, 5):
            run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
            run.font.name = MONO
            run.font.size = Pt(10)
        colour = (INK if data[r][0] == "all" else
                  GREEN if r <= 2 else (AMBER if r == 3 else RED))
        for c in (0, 1):
            run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
            run.font.color.rgb = colour
            run.font.bold = True

    x2 = ML + tw + 0.32
    w2 = SW - MR - x2
    card(s, x2, y, w2, 1.90,
         heading="Why \u201cread\u201d fails, and why more data will not fix it",
         lines=["read has MORE training boxes than write and scores far worse. So "
                "this is not a data-volume problem.",
                "Reading and writing differ only by what the hands are doing. The "
                "closest published work reaches 57.8% on writing even with a strong "
                "temporal model."],
         accent=RED, heading_size=11.5, body_size=10, fill=PANEL)
    card(s, x2, y + 2.08, w2, 1.90,
         heading="So the output never forces the choice",
         lines=["Book visible, hands not visible \u2192 reported as \u201creading "
                "or writing\u201d, one label covering both.",
                "Hands visible \u2192 writing is reported with confidence "
                "\u201cinferred\u201d, never \u201cdirect\u201d.",
                "The weakest number in the table is the one the system is most "
                "careful about."],
         accent=GREEN, heading_size=11.5, body_size=10, fill=PANEL2)

    banner(s, y + 4.16,
           [R("And it is a second opinion, never the primary signal:  ", 10.5, True,
              TEAL_D, FONT_SB),
            R("actions are decided first by geometry \u2014 object overlap, wrist "
              "position, head pitch \u2014 which needs no training data. Where the "
              "two disagree, geometry wins. That is why a 0.607 model is safe to "
              "ship inside this system.", 10.5, False, INK)],
           h=0.80)
    return s


# --------------------------------------------------------------- criterion 4

def s09_comparison(prs, page):
    s, y = chrome(prs, "Criterion 4 \u2014 Comparison with Base Paper",
                  "What we can and cannot compare", None, page, RED)
    y = claim(s, y, "One comparison is real. Two are not applicable. One is "
                    "impossible \u2014 and each has a specific reason.", RED)

    blocks = [
        ("REAL COMPARISON", GREEN,
         "Our fine-tuned detector",
         "write mAP@50 0.767  \u00b7  read 0.419",
         "Literature reports 57.8% on writing detection. Different metric, "
         "different data \u2014 so we claim only the ORDERING: both find writing "
         "hard, and our read/write gap reproduces the difficulty they report."),
        ("NOT APPLICABLE", MUTE,
         "The two FER papers",
         "no expression benchmark run",
         "We never reproduced their task. We took their methods \u2014 the rolling "
         "window and the refusal to force a confusable class. Claiming their "
         "accuracy numbers would be claiming work we did not do."),
        ("IMPOSSIBLE", RED,
         "ARG group activity",
         "Volleyball 92.3% \u2014 we have no number",
         "The GCN is implemented but untrained. Training needs ~3,000 clips "
         "labelled high/medium/low by agreeing raters. The architecture, contract, "
         "abstention rule and tests are done; the missing piece is data."),
    ]
    third = (CW - 2 * 0.28) / 3
    for i, (tag, accent, who, num, body) in enumerate(blocks):
        x = ML + i * (third + 0.28)
        rect(s, x, y, third, 3.10, fill=WHITE, line=BORDER)
        bar(s, x, y, third, 0.05, accent)
        add_text(s, x + 0.24, y + 0.22, third - 0.48, 0.24,
                 [P(tag, 9, True, accent, FONT_SB)])
        add_text(s, x + 0.24, y + 0.56, third - 0.48, 0.32,
                 [P(who, 13, True, INK, FONT_SB, line=1.10)])
        add_text(s, x + 0.24, y + 0.98, third - 0.48, 0.30,
                 [P(num, 10.5, True, accent, MONO, line=1.12)])
        add_text(s, x + 0.24, y + 1.42, third - 0.48, 1.52,
                 [P(body, 10, False, BODY, line=1.24)])

    banner(s, y + 3.32,
           [R("The point we want to land:  ", 11, True, TEAL_D, FONT_SB),
            R("a comparison we cannot make is a finding, not a gap in the report. "
              "Saying exactly which one is missing and exactly what it would take "
              "is worth more than a number that describes nothing.",
              11, False, INK)],
           h=0.84)
    return s


def s10_summary(prs, page):
    s, y = chrome(prs, "Summary", "The four criteria, in one line each", None, page)
    y = claim(s, y, "Every number here came from a command we can re-run in front "
                    "of you. Where we could not measure something, the slide says so.")

    rows = [
        ("1", "Base paper implementation", GREEN,
         "ARG replicated stage for stage, 14 tests passing. Two FER papers adopted "
         "as methods.", "GCN untrained \u2014 no labelled data exists"),
        ("2", "Dataset preparation", GREEN,
         "8 classes \u2192 4, 423 \u2192 877 images. mAP@50 0.415 \u2192 0.607 "
         "from data alone.", "imgsz and batch also changed \u2014 confounded"),
        ("3", "Training & evaluation", AMBER,
         "42 epochs, early stop at 27, per-class validation re-run from the weights.",
         "58-image validation split is small and noisy"),
        ("4", "Comparison with base paper", RED,
         "One real comparison, two not applicable, one impossible \u2014 each "
         "explained.", "different metric \u2014 ordering, not parity"),
    ]
    rh = 0.86
    for i, (num, title, accent, did, gap) in enumerate(rows):
        yy = y + i * (rh + 0.14)
        rect(s, ML, yy, CW, rh, fill=WHITE if i % 2 == 0 else PANEL, line=BORDER)
        bar(s, ML, yy, 0.05, rh, accent)
        add_text(s, ML + 0.26, yy + 0.16, 2.95, 0.52,
                 [PR([R(num + "   ", 14, True, accent, FONT_SB),
                      R(title, 11.5, True, INK, FONT_SB)])])
        add_text(s, ML + 3.35, yy + 0.18, 5.05, 0.56,
                 [P(did, 10, False, BODY, line=1.20)])
        add_text(s, ML + 8.60, yy + 0.18, CW - 8.85, 0.56,
                 [PR([R("gap:  ", 9.5, True, AMBER, FONT_SB),
                      R(gap, 9.5, False, BODY)], line=1.20)])
    return s


# ------------------------------------------------------------------ appendix

def s11_appendix_hyper(prs, page):
    s, y = chrome(prs, "Appendix", "Full hyperparameters, as args.yaml recorded them",
                  "Kept off the main deck so criterion 3 has one point, not forty. "
                  "Here when a question needs it.", page, MUTE)

    half = (CW - 0.30) / 2
    cfg = [
        ["Setting", "Value", "Note"],
        ["base weights", "yolo11m.pt", "COCO-pretrained"],
        ["freeze", "null", "all layers updated"],
        ["epochs / patience", "60 / 15", "ran 42"],
        ["best epoch", "27", "later epochs overfit"],
        ["imgsz", "640", "960 needed 9.05 GB"],
        ["batch", "8", "largest that fits"],
        ["optimizer", "auto (SGD)", "default"],
        ["lr0 / lrf", "0.01 / 0.01", "default"],
        ["momentum", "0.937", "default"],
        ["weight_decay", "0.0005", "default"],
        ["warmup_epochs", "3.0", "default"],
        ["amp", "true", "required to fit"],
        ["seed", "0", "reproducible split"],
    ]
    tbl = table(s, ML, y, half, [2.10, 1.65, 2.17], cfg, row_h=0.285, head_h=0.32,
                size=9.5, head_size=9.5, col_bold={0}, head_fill=MUTE)
    for r in range(1, len(cfg)):
        run = tbl.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.size = Pt(9)
        run.font.color.rgb = TEAL_D
        run.font.bold = True

    x2 = ML + half + 0.30
    loss = [
        ["Loss term", "Weight", "Penalises"],
        ["box", "7.5", "box regression error (CIoU)"],
        ["cls", "0.5", "class confidence error (BCE)"],
        ["dfl", "1.5", "distribution focal loss on edges"],
    ]
    tbl2 = table(s, x2, y, half, [1.85, 1.30, 2.77], loss, row_h=0.32, head_h=0.34,
                 size=9.5, head_size=9.5, col_bold={0}, head_fill=MUTE)
    for r in range(1, 4):
        run = tbl2.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.bold = True
        run.font.color.rgb = TEAL_D

    card(s, x2, y + 1.48, half, 2.55,
         heading="ARG stage-by-stage mapping",
         lines=["Nodes = detected students \u2014 same as paper.",
                "Typed relations, appearance + position, each with its own learned "
                "weight \u2014 same as paper.",
                "Renormalised adjacency D^-1/2 (A+I) D^-1/2 \u2014 same as paper.",
                "2-layer GCN message passing \u2014 same as paper.",
                "Mean-pool readout \u2014 same as paper.",
                "Deviations: ordinal engagement not activity classes; 2 layers not a "
                "deep stack; abstains when too few students are visible."],
         accent=GREEN, heading_size=11.5, body_size=9.5, fill=PANEL2)
    return s


def s12_appendix_aug(prs, page):
    s, y = chrome(prs, "Appendix", "Augmentation, and what each setting defends against",
                  "An augmentation set is defined as much by what is switched off as "
                  "by what is on.", page, MUTE)

    aug = [
        ["Augmentation", "Value", "Defends against"],
        ["mosaic", "1.0", "four images per sample: more context per step"],
        ["close_mosaic", "10", "last 10 epochs clean, so it ends on real layout"],
        ["fliplr", "0.5", "a student on the left is the same student"],
        ["scale", "0.5", "front row and back row differ hugely in pixels"],
        ["translate", "0.1", "the camera is not framed identically every time"],
        ["hsv_h / hsv_s / hsv_v", "0.015 / 0.7 / 0.4", "lighting, daylight to tube light"],
        ["erasing", "0.4", "occlusion by desks, bags and neighbours"],
        ["auto_augment", "randaugment", "applied to classification crops"],
        ["degrees / shear / perspective", "0 / 0 / 0", "OFF \u2014 a rotated classroom is not a real view"],
        ["flipud", "0", "OFF \u2014 nobody is upside down"],
        ["mixup / copy_paste", "0 / 0", "OFF \u2014 blended students are not a real scene"],
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

    x2 = ML + tw + 0.30
    w2 = SW - MR - x2
    card(s, x2, y, w2, 1.75,
         heading="The split",
         lines=["877 train / 58 validation, fixed with seed 0.",
                "No test split: with 58 validation images, carving out a third set "
                "would leave neither usable. Every number in this deck is a "
                "validation number and is labelled as one."],
         accent=AMBER, heading_size=11.5, body_size=9.5, fill=PANEL)
    card(s, x2, y + 1.93, w2, 1.95,
         heading="close_mosaic is the subtle one",
         lines=["Mosaic helps early: four images per sample means more context per "
                "gradient step.",
                "But a model that only ever sees four-image composites never "
                "calibrates on a real frame, so it is switched off for the final ten "
                "epochs and the run ends on genuine classroom layout."],
         accent=TEAL, heading_size=11.5, body_size=9.5, fill=PANEL2)
    return s


def main():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    s01_title(prs)
    s02_papers(prs, 2)
    s03_arg(prs, 3)
    s04_dataset(prs, 4)
    s05_ablation(prs, 5)
    s06_training(prs, 6)
    s07_curves(prs, 7)
    s08_perclass(prs, 8)
    s09_comparison(prs, 9)
    s10_summary(prs, 10)
    s11_appendix_hyper(prs, 11)
    s12_appendix_aug(prs, 12)

    prs.save(str(OUT))
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
