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

def s01_title(prs):
    s = new_slide(prs)
    bar(s, 0, 0, SW, 0.09, TEAL)
    add_text(s, ML, 1.90, CW, 0.3, [P("PROJECT REVIEW", 11, True, TEAL, FONT_SB)])
    add_text(s, ML, 2.28, CW, 1.15,
             [P("ClassGraph — classroom engagement from video",
                38, True, INK, FONT_SB, line=1.03)])
    add_text(s, ML, 3.66, CW * 0.80, 0.9,
             [P("Three base papers, one model fine-tuned on classroom footage, and a "
                "measured account of what worked, what did not, and which comparisons "
                "we are not entitled to make.", 14, False, BODY, line=1.32)])
    bar(s, ML, 4.92, 1.5, 0.03, TEAL)
    rows = [("1", "Base Paper Implementation", "ARG replicated; 2 FER papers as method"),
            ("2", "Dataset Preparation", "8 classes to 4; 423 to 877 images"),
            ("3", "Training & Evaluation", "42 epochs, mAP@50 0.607, per class"),
            ("4", "Comparison with Base Paper", "where we match, and where we cannot")]
    cw = (CW - 3 * 0.24) / 4
    for i, (num, title, sub) in enumerate(rows):
        x = ML + i * (cw + 0.24)
        rect(s, x, 5.20, cw, 1.28, fill=PANEL, line=BORDER)
        add_text(s, x + 0.20, 5.36, cw - 0.36, 0.96,
                 [PR([R(num + "   ", 15, True, TEAL, FONT_SB),
                      R(title, 11.5, True, INK, FONT_SB)]),
                  P(sub, 9.5, False, MUTE, line=1.18, space_before=5)])
    return s


def s02_papers(prs, page):
    s, y = chrome(prs, "Criterion 1 — Base Paper Implementation",
                  "Three base papers, and exactly what each contributes",
                  "Two are used for their method, one for its architecture. Saying "
                  "which is which is the whole of this criterion.", page)
    data = [
        ["Paper", "Venue", "What it gives us", "Status here"],
        ["Learning Actor Relation Graphs for\nGroup Activity Recognition (Wu et al.)",
         "CVPR 2019", "Architecture: relation graph + GCN readout",
         "Implemented, untrained"],
        ["Robust Dynamic Facial Expression\nRecognition (Liu, Wang & Shen)",
         "IEEE T-BIOM\n2025", "Method: separate a momentary signal from a\nsustained state",
         "Implemented as the\nrolling window"],
        ["Dynamic Objectives Learning for\nFacial Expression Recognition (Wen et al.)",
         "IEEE TMM\n2020", "Method: keep confusable categories apart\nrather than forcing a class",
         "Implemented as\nabstention"],
    ]
    tbl = table(s, ML, y, CW, [4.05, 1.45, 4.35, 2.12], data,
                row_h=0.72, head_h=0.38, size=10, head_size=10, col_bold={0})
    for r in range(1, 4):
        run = tbl.cell(r, 3).text_frame.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.color.rgb = AMBER if r == 1 else GREEN

    yy = y + 0.38 + 3 * 0.72 + 0.28
    half = (CW - 0.30) / 2
    card(s, ML, yy, half, 1.55,
         heading="Why two facial-expression papers for a system that does not classify emotion",
         lines=["We took their methods, not their task. Both papers solve the same "
                "underlying problem we have — a per-frame signal is noisy and a "
                "confident label on a hard sample is worse than no label.",
                "That is a claim about method transfer, and we state it that way "
                "rather than implying we reproduced their FER results."],
         accent=TEAL, heading_size=11, body_size=9.5, fill=PANEL)
    card(s, ML + half + 0.30, yy, half, 1.55,
         heading="Why ARG is the one architecture we replicate",
         lines=["It is the only one of the three whose structure matches our problem "
                "directly: people as nodes, typed relations as edges, one label for "
                "the whole group.",
                "Our target differs — ordinal engagement rather than an activity "
                "class — so the deviations are documented rather than glossed. Next "
                "slide."],
         accent=GREEN, heading_size=11, body_size=9.5, fill=PANEL2)
    return s


def s03_arg(prs, page):
    s, y = chrome(prs, "Criterion 1 — Base Paper Implementation",
                  "ARG: what the paper specifies, and what we built",
                  "Stage by stage against the paper, with the three deviations and "
                  "the reason each was forced.", page)
    data = [
        ["ARG stage (Wu et al., CVPR 2019)", "Our implementation", "Same?"],
        ["Nodes are detected actors", "Nodes are detected students, via YOLO11m", "yes"],
        ["Typed relations: appearance + position",
         "Same two relation types, each with its own learned weight", "yes"],
        ["Relation-weighted adjacency, renormalised",
         "_renormalize(adj) before message passing", "yes"],
        ["GCN message passing over that adjacency",
         "2 layers: tanh(A @ H), applied twice", "yes"],
        ["Pool node features to a graph prediction",
         "Mean-pool then a linear layer to the label set", "yes"],
        ["Predicts an activity class (volleyball, collective)",
         "Predicts ordinal engagement: high / medium / low", "deviation 1"],
        ["Deep stacks, thousands of labelled clips",
         "2-layer readout — our labelled data does not support more", "deviation 2"],
        ["Assumes every actor is detected every frame",
         "Abstains below min_students or above max_unknown_rate", "deviation 3"],
    ]
    tbl = table(s, ML, y, CW, [4.55, 5.45, 1.97], data,
                row_h=0.375, head_h=0.38, size=10, head_size=10, col_bold={0})
    for r in range(1, len(data)):
        run = tbl.cell(r, 2).text_frame.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.name = FONT_SB
        run.font.color.rgb = GREEN if data[r][2] == "yes" else AMBER

    yy = y + 0.38 + 8 * 0.375 + 0.26
    banner(s, yy,
           [R("Stated plainly, because a panel will find it anyway:  ", 11, True,
              RED, FONT_SB),
            R("the architecture is implemented and unit-tested, but its weights are "
              "randomly initialised. It has never been trained, because the labelled "
              "group-engagement data it needs (OUC-CGE style) is not something we "
              "have. backend/group_activity.py says this in its own docstring — "
              "“a named slot showing exactly what is still missing” — and no "
              "number from it appears anywhere in this deck.",
              11, False, INK)],
           accent=RED, h=1.18, fill=RGBColor(0xFC, 0xF3, 0xF2))
    return s


def s04_dataset_prep(prs, page):
    s, y = chrome(prs, "Criterion 2 — Dataset Preparation",
                  "Cleaning: eight classes became four",
                  "The single most consequential decision in the project, and the one "
                  "that produced the largest measured gain.", page, GREEN)

    half = (CW - 0.30) / 2
    before = [
        ["Original — 8 classes", "train boxes"],
        ["handrise", "sparse"], ["look_forward", "sparse"], ["read", "1,254"],
        ["sleep", "1,512"], ["stand", "sparse"], ["turn_head", "sparse"],
        ["using_device", "2,204"], ["write", "1,121"],
    ]
    table(s, ML, y, half - 0.4, [3.2, 2.2], before, row_h=0.30, head_h=0.34,
          size=9.5, head_size=9.5, col_bold={0}, head_fill=MUTE)

    after = [
        ["Merged — 4 classes", "train boxes", "val boxes"],
        ["using_device", "2,204", "90"],
        ["sleep", "1,512", "41"],
        ["read", "1,254", "49"],
        ["write", "1,121", "91"],
        ["total", "6,091", "271"],
    ]
    x2 = ML + half + 0.30
    tbl = table(s, x2, y, half, [2.6, 1.7, 1.55], after, row_h=0.30, head_h=0.34,
                size=9.5, head_size=9.5, col_bold={0}, head_fill=GREEN)
    for c in range(3):
        tbl.cell(5, c).text_frame.paragraphs[0].runs[0].font.bold = True

    yy = y + 2.62
    card(s, ML, yy, CW, 1.45,
         heading="Why four and not eight — and why this is cleaning, not cheating",
         lines=["Four of the eight classes were too sparse to learn: a class with a "
                "few dozen boxes produces a detector that has memorised those boxes. "
                "Three of them (handrise, look_forward, turn_head) are also better "
                "measured by geometry than by a detector — wrist position and head "
                "pitch give the same answer without training data, so they were moved "
                "to the rule layer rather than deleted from the system.",
                "Two source sets were then merged, taking train images from 423 to "
                "877. The validation split was left untouched at 58 images so the "
                "before/after comparison on the next slide is measured on the same "
                "held-out data."],
         accent=GREEN, heading_size=12, body_size=10, fill=PANEL2)
    return s


def s05_augmentation(prs, page):
    s, y = chrome(prs, "Criterion 2 — Dataset Preparation",
                  "Augmentation and splitting, exactly as configured",
                  "Read from runs/behaviour/merged4_aug/args.yaml — the configuration "
                  "the run actually used.", page, GREEN)

    aug = [
        ["Augmentation", "Value", "What it defends against"],
        ["mosaic", "1.0", "four images per sample: more context per step"],
        ["close_mosaic", "10", "last 10 epochs clean, so the model ends on real layout"],
        ["fliplr", "0.5", "a student on the left of the room is the same student"],
        ["scale", "0.5", "front row and back row differ hugely in pixels"],
        ["translate", "0.1", "the camera is not always framed identically"],
        ["hsv_h / hsv_s / hsv_v", "0.015 / 0.7 / 0.4", "classroom lighting, daylight to tube light"],
        ["erasing", "0.4", "occlusion by desks, bags and neighbours"],
        ["auto_augment", "randaugment", "applied to classification crops"],
        ["degrees / shear / perspective", "0 / 0 / 0", "off: a rotated classroom is not a real view"],
        ["flipud", "0", "off: nobody is upside down"],
        ["mixup / copy_paste", "0 / 0", "off: blended students are not a real scene"],
    ]
    tw = 8.05
    tbl = table(s, ML, y, tw, [2.65, 1.85, 3.55], aug, row_h=0.325, head_h=0.35,
                size=9.5, head_size=9.5, col_bold={0}, head_fill=GREEN)
    for r in range(1, len(aug)):
        run = tbl.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.size = Pt(9)
        run.font.color.rgb = MUTE if r >= 9 else TEAL_D
        run.font.bold = r < 9

    x2 = ML + tw + 0.30
    w2 = SW - MR - x2
    card(s, x2, y, w2, 1.95,
         heading="The split",
         lines=["877 train images / 58 validation images, a fixed split with seed 0.",
                "No test split. With 58 validation images, carving out a third set "
                "would leave neither usable — so every number in this deck is a "
                "validation number and is labelled as one."],
         accent=GREEN, heading_size=11.5, body_size=9.5, fill=PANEL2)
    card(s, x2, y + 2.10, w2, 2.32,
         heading="What the four 'off' rows are doing here",
         lines=["An augmentation set is defined as much by what is disabled as by "
                "what is on. Rotation, vertical flip and mixup all generate images "
                "that cannot occur in a fixed classroom camera, and training on them "
                "spends capacity defending against a world that does not exist.",
                "close_mosaic is the subtle one: mosaic helps early, but a model that "
                "only ever sees four-image composites never calibrates on a real "
                "frame, so it is switched off for the final ten epochs."],
         accent=AMBER, heading_size=11.5, body_size=9.5, fill=PANEL)
    return s


def s06_ablation(prs, page):
    s, y = chrome(prs, "Criterion 2 — Dataset Preparation",
                  "What the preparation bought, measured",
                  "Same architecture, same pretrained weights, same held-out split. "
                  "The only change is the data.", page, GREEN)
    bottom = picture(s, ASSETS / "dataset_ablation.png", ML, y, 7.30)

    x2 = ML + 7.30 + 0.34
    w2 = SW - MR - x2
    card(s, x2, y, w2, 2.10,
         heading="mAP@50 rose 46%, recall 54%",
         lines=["0.415 to 0.607, and recall 0.381 to 0.586. No change to the model, "
                "the pretrained weights, or the loss — only to the labels and the "
                "number of images behind each one.",
                "Recall moving most is the expected signature: the old model was not "
                "confused about what it saw, it was failing to see at all, because "
                "four of its eight classes had too few examples to fire on."],
         accent=GREEN, heading_size=11.5, body_size=9.5, fill=PANEL2)

    card(s, x2, y + 2.28, w2, 2.05,
         heading="The confound, stated before anyone asks",
         lines=["The two runs also differ in image size (960 to 640) and batch (4 to "
                "8), changed to fit a 6.4 GB card. So this is not a clean single-"
                "variable ablation and we do not present it as one.",
                "It is still evidence in the right direction: the resolution moved "
                "DOWN, which should hurt a detector, and the score rose anyway."],
         accent=AMBER, heading_size=11.5, body_size=9.5, fill=PANEL)

    if bottom < y + 3.4:
        banner(s, max(bottom + 0.25, y + 3.55),
               [R("Read the recall bar first.  ", 11, True, TEAL_D, FONT_SB),
                R("It is the one that changes what the system can do: a detector that "
                  "misses a behaviour gives the rule layer nothing to disagree with, "
                  "and a missed student is not a student with a weaker score — they "
                  "are absent from the record.", 11, False, INK)],
               h=0.86)
    return s


def s07_hyper(prs, page):
    s, y = chrome(prs, "Criterion 3 — Training & Evaluation",
                  "Hyperparameters, as the run recorded them",
                  "Not written down afterwards: these are the keys in args.yaml, "
                  "reproducible with tools/train_behaviour.py.", page, AMBER)

    half = (CW - 0.30) / 2
    cfg = [
        ["Setting", "Value", "Why this value"],
        ["base weights", "yolo11m.pt", "COCO-pretrained, not random init"],
        ["freeze", "null", "all 20.1 M parameters updated"],
        ["epochs / patience", "60 / 15", "ran 42, early-stopped"],
        ["best epoch", "27", "later epochs overfit the 58-image val set"],
        ["imgsz", "640", "960 wanted 9.05 GB on a 6.4 GB card"],
        ["batch", "8", "largest that fits at 640"],
        ["optimizer", "auto (SGD)", "Ultralytics default, not tuned"],
        ["lr0 / lrf", "0.01 / 0.01", "default cosine-free schedule"],
        ["momentum", "0.937", "default"],
        ["weight_decay", "0.0005", "default"],
        ["warmup_epochs", "3.0", "default"],
        ["amp", "true", "mixed precision, required to fit"],
        ["seed", "0", "fixed split, reproducible"],
    ]
    tbl = table(s, ML, y, half, [2.10, 1.70, 2.12], cfg, row_h=0.295, head_h=0.33,
                size=9.5, head_size=9.5, col_bold={0}, head_fill=AMBER)
    for r in range(1, len(cfg)):
        run = tbl.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.size = Pt(9)
        run.font.color.rgb = TEAL_D
        run.font.bold = True

    x2 = ML + half + 0.30
    loss = [
        ["Loss term", "Weight", "What it penalises"],
        ["box", "7.5", "box regression error (CIoU)"],
        ["cls", "0.5", "class confidence error (BCE)"],
        ["dfl", "1.5", "distribution focal loss on box edges"],
    ]
    tbl2 = table(s, x2, y, half, [2.10, 1.35, 2.47], loss, row_h=0.32, head_h=0.34,
                 size=9.5, head_size=9.5, col_bold={0}, head_fill=AMBER)
    for r in range(1, 4):
        run = tbl2.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.bold = True
        run.font.color.rgb = TEAL_D

    card(s, x2, y + 1.42, half, 1.52,
         heading="Nothing here was hyperparameter-searched",
         lines=["Every optimiser value is the Ultralytics default. With 877 training "
                "images and a 58-image validation split, a search would tune against "
                "noise and report the winner as a finding.",
                "The two values we did choose — imgsz and batch — were chosen by what "
                "fits in 6.4 GB of VRAM, and we say so rather than implying they were "
                "optimised."],
         accent=RED, heading_size=11.5, body_size=9.5, fill=PANEL)

    card(s, x2, y + 3.10, half, 1.28,
         heading="Cost",
         lines=["23 minutes wall-clock on an RTX 4050 laptop GPU. At imgsz 960 and "
                "batch 8 the same run wanted 9.05 GB and spilled to 10 s/iteration — "
                "an estimated 18 hours. Fitting the card was worth more than the "
                "resolution."],
         accent=AMBER, heading_size=11.5, body_size=9.5, fill=PANEL2)
    return s


def s08_curves(prs, page):
    s, y = chrome(prs, "Criterion 3 — Training & Evaluation",
                  "Loss and detection quality, epoch by epoch",
                  "Two charts rather than one: loss and mAP share an x-axis and "
                  "nothing else, and a second y-scale would invite a relationship "
                  "that is an artefact of scaling.", page, AMBER)

    half = (CW - 0.34) / 2
    b1 = picture(s, ASSETS / "train_loss.png", ML, y, half)
    b2 = picture(s, ASSETS / "train_metrics.png", ML + half + 0.34, y, half)

    yy = max(b1, b2) + 0.26
    third = (CW - 2 * 0.26) / 3
    notes = [
        ("Validation loss, not training loss", TEAL,
         "Training loss falls by construction. The question a reviewer is asking is "
         "whether it fell on images the model never saw, so that is what is plotted."),
        ("Early stopping did its job", GREEN,
         "Best mAP@50-95 at epoch 27; training ran to 42 and stopped on patience 15. "
         "The 15 epochs after the peak are the evidence that it had stopped learning."),
        ("The curve is noisy, and that is the split", AMBER,
         "58 validation images means one hard image moves mAP@50 by a visible amount. "
         "We report the peak, but the honest read is a plateau around 0.55-0.60."),
    ]
    for i, (title, accent, body) in enumerate(notes):
        x = ML + i * (third + 0.26)
        rect(s, x, yy, third, 1.30, fill=PANEL, line=BORDER)
        bar(s, x, yy, 0.045, 1.30, accent)
        add_text(s, x + 0.22, yy + 0.14, third - 0.40, 1.02,
                 [P(title, 11, True, INK, FONT_SB, line=1.10),
                  P(body, 9.5, False, BODY, line=1.20, space_before=5)])
    return s


def s09_perclass(prs, page):
    s, y = chrome(prs, "Criterion 3 — Training & Evaluation",
                  "Per-class results, re-validated from the weights",
                  "Not copied from the training log: re-run for this deck with "
                  "tools/model_facts.py --validate.", page, AMBER)

    data = [
        ["Class", "Precision", "Recall", "mAP@50", "mAP@50-95", "Train boxes"],
        ["write", "0.723", "0.681", "0.767", "0.331", "1,121"],
        ["using_device", "0.749", "0.700", "0.720", "0.336", "2,204"],
        ["sleep", "0.670", "0.495", "0.521", "0.263", "1,512"],
        ["read", "0.455", "0.469", "0.419", "0.143", "1,254"],
        ["all", "0.649", "0.586", "0.607", "0.268", "6,091"],
    ]
    tw = 7.35
    tbl = table(s, ML, y, tw, [1.70, 1.15, 1.02, 1.12, 1.32, 1.04], data,
                row_h=0.40, head_h=0.38, size=10, head_size=9.5, col_bold={0},
                head_fill=AMBER)
    for r in range(1, len(data)):
        for c in range(1, 6):
            run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
            run.font.name = MONO
            run.font.size = Pt(9.5)
        colour = (INK if data[r][0] == "all" else
                  GREEN if r <= 2 else (AMBER if r == 3 else RED))
        for c in (0, 3):
            run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
            run.font.color.rgb = colour
            run.font.bold = True

    yy = y + 0.38 + 5 * 0.40 + 0.26
    card(s, ML, yy, tw, 1.62,
         heading="'read' is the weakest class, and the weakness was predicted",
         lines=["0.419 against write at 0.767. Reading and writing differ only by what "
                "the hands are doing, and the closest published work reaches 57.8% on "
                "writing even with a strong temporal model.",
                "Note it is not a data-volume problem: read has more training boxes "
                "than write and scores far worse. More labels would not fix this; a "
                "temporal model or a hand-state cue might."],
         accent=RED, heading_size=11.5, body_size=9.5, fill=PANEL)

    x2 = ML + tw + 0.30
    w2 = SW - MR - x2
    card(s, x2, y, w2, 2.05,
         heading="How the system uses a 0.607 model safely",
         lines=["It is a second opinion, never the primary signal. Actions are decided "
                "first by geometry — object overlap, wrist position, head pitch — "
                "which is auditable and needs no training data.",
                "Where the two disagree, geometry wins and the record shows both. That "
                "is why a mid-0.6 mAP model is safe to ship inside this system."],
         accent=TEAL, heading_size=11.5, body_size=9.5, fill=PANEL2)

    card(s, x2, y + 2.23, w2, 2.10,
         heading="And how the pipeline hides the read/write confusion",
         lines=["When a book is visible but the hands are not, the reported action is "
                "“reading or writing” — one label covering both — instead of "
                "a coin-flip between them.",
                "When the hands are visible, writing is reported with confidence "
                "“inferred” rather than “direct”. The weakest "
                "number in the table is the one the output is most careful about."],
         accent=GREEN, heading_size=11.5, body_size=9.5, fill=PANEL)
    return s


def s10_comparison(prs, page):
    s, y = chrome(prs, "Criterion 4 — Comparison with Base Paper",
                  "Where we can compare, where we cannot, and why",
                  "The honest answer is partly a non-comparison, and that is the "
                  "finding rather than an excuse.", page, RED)

    data = [
        ["Base paper", "Their result", "Our comparable result", "Verdict"],
        ["ARG (Wu et al., CVPR 2019)",
         "Volleyball 92.3% /\nCollective 91.0% activity acc.",
         "None — the GCN is implemented\nbut untrained",
         "Cannot compare"],
        ["Robust Dynamic FER\n(Liu, Wang & Shen, 2025)",
         "DFEW / FERV39k\nexpression accuracy",
         "Not reproduced — we take the\nmethod, not the classifier",
         "Not applicable"],
        ["Dynamic Objectives Learning\n(Wen et al., 2020)",
         "Expression accuracy on\nposed + wild sets",
         "Not reproduced — same reason",
         "Not applicable"],
        ["Our fine-tuned detector\n(YOLO11m, 4 classes)",
         "Literature on writing\ndetection: 57.8%",
         "write mAP@50 0.767,\nread mAP@50 0.419",
         "Comparable, with care"],
    ]
    tbl = table(s, ML, y, CW, [3.30, 3.05, 3.55, 2.07], data,
                row_h=0.68, head_h=0.38, size=9.5, head_size=9.5, col_bold={0},
                head_fill=RED)
    for r in range(1, len(data)):
        run = tbl.cell(r, 3).text_frame.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.color.rgb = GREEN if r == 4 else (RED if r == 1 else MUTE)

    yy = y + 0.38 + 4 * 0.68 + 0.26
    half = (CW - 0.30) / 2
    card(s, ML, yy, half, 1.80,
         heading="Why the one comparison we can make still needs care",
         lines=["57.8% is an accuracy on a writing-recognition benchmark; 0.767 is "
                "mAP@50 on our own 58-image validation split. Different metric, "
                "different data, different task definition.",
                "What survives the comparison is the ORDERING: both find writing hard "
                "and both find it harder than gross-posture classes. Our read/write "
                "gap reproduces the difficulty the literature reports, which is a "
                "weaker claim than matching a number and a truer one."],
         accent=AMBER, heading_size=11.5, body_size=9.5, fill=PANEL)
    card(s, ML + half + 0.30, yy, half, 1.80,
         heading="What we would need to make the ARG comparison real",
         lines=["A labelled group-engagement set in the OUC-CGE mould: roughly 3,000 "
                "clips with an ordinal high/medium/low label agreed by multiple "
                "raters. We do not have one and could not build one in this project.",
                "The architecture, the input contract, the abstention rule and a "
                "14-test suite are all in place, so the missing piece is data, not "
                "engineering. That is a specific answer to “what is left”, "
                "which is worth more than a number we cannot defend."],
         accent=TEAL, heading_size=11.5, body_size=9.5, fill=PANEL2)
    return s


def s11_summary(prs, page):
    s, y = chrome(prs, "Summary",
                  "The four criteria, answered",
                  "Each row is a claim we can show the evidence for, in this "
                  "repository, in under a minute.", page)

    data = [
        ["Criterion", "What we did", "Evidence", "Honest gap"],
        ["1 · Base paper implementation",
         "ARG replicated stage for stage;\n2 FER papers taken as method",
         "backend/group_activity.py,\n14 passing tests",
         "GCN untrained —\nno labelled data"],
        ["2 · Dataset preparation",
         "8 classes to 4, 423 to 877 images,\n11 augmentation settings justified",
         "args.yaml, data.yaml,\nmAP@50 0.415 to 0.607",
         "imgsz/batch also\nchanged — confounded"],
        ["3 · Training & evaluation",
         "42 epochs, early stop at 27,\nper-class validation",
         "results.csv, re-validated\nfrom best.pt",
         "58-image val split\nis small and noisy"],
        ["4 · Comparison with base paper",
         "One real comparison; two N/A;\none impossible — each explained",
         "write 0.767 vs 57.8%\nreported difficulty",
         "Different metric —\nordering, not parity"],
    ]
    tbl = table(s, ML, y, CW, [2.85, 3.80, 2.86, 2.46], data,
                row_h=0.72, head_h=0.38, size=9.5, head_size=9.5, col_bold={0})
    for r in range(1, len(data)):
        run = tbl.cell(r, 3).text_frame.paragraphs[0].runs[0]
        run.font.color.rgb = AMBER
        run.font.bold = True

    yy = y + 0.38 + 4 * 0.72 + 0.28
    banner(s, yy,
           [R("The through-line:  ", 11.5, True, TEAL_D, FONT_SB),
            R("every number in this deck was produced by a command that can be re-run "
              "in front of you — tools/model_facts.py --validate for the metrics, "
              "tools/make_review_charts.py for the curves, tools/train_behaviour.py "
              "for the run itself. Where we could not measure something, the slide "
              "says so instead of estimating it.", 11.5, False, INK)],
           h=1.10)
    return s


def main():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    s01_title(prs)
    s02_papers(prs, 2)
    s03_arg(prs, 3)
    s04_dataset_prep(prs, 4)
    s05_augmentation(prs, 5)
    s06_ablation(prs, 6)
    s07_hyper(prs, 7)
    s08_curves(prs, 8)
    s09_perclass(prs, 9)
    s10_comparison(prs, 10)
    s11_summary(prs, 11)

    prs.save(str(OUT))
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
