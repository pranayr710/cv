"""Build the model-detail deck: what each model is, and what it actually does here.

Every number is measured from this machine by tools/model_facts.py -- parameter
counts read off the loaded graphs, file sizes off disk, and the fine-tuning
metrics re-validated from the trained weights rather than copied out of a log.
A panel asking "how many parameters" should get an answer that came from the
model, not from a datasheet.

Run:  python build_models_ppt.py
Out:  ClassGraph_Models.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "ppt_assets"
OUT = ROOT / "ClassGraph_Models.pptx"

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
RULE = RGBColor(0xE6, 0xEC, 0xF2)

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
        if blk.get("space_after"):
            para.space_after = Pt(blk["space_after"])
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


def chrome(prs, eyebrow, title, lead=None, page=None):
    s = new_slide(prs)
    bar(s, 0, 0, SW, 0.055, TEAL)
    add_text(s, ML, 0.40, CW, 0.26,
             [P(eyebrow.upper(), 9.5, True, TEAL, FONT_SB)])
    add_text(s, ML, 0.68, CW, 0.52, [P(title, 25, True, INK, FONT_SB, line=1.04)])
    y = 1.34
    if lead:
        add_text(s, ML, y, CW, 0.44, [P(lead, 11.5, False, MUTE, line=1.22)])
        y += 0.58
    if page:
        add_text(s, SW - MR - 1.0, SH - 0.46, 1.0, 0.22,
                 [P(str(page), 9, False, MUTE, align=PP_ALIGN.RIGHT)])
    return s, y + 0.06


def table(slide, x, y, w, col_w, data, row_h=0.34, head_h=0.36, size=10,
          head_size=10, col_bold=frozenset()):
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
                INK if r == 0 else (WHITE if r % 2 else PANEL))
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


# --------------------------------------------------------------------------- #
# Measured facts. Regenerate with: python tools/model_facts.py
# --------------------------------------------------------------------------- #

MODELS = [
    # name, params, file MB, purpose
    ("YOLO11m", "20,114,688", "40.7", "Find every person and held object"),
    ("YOLO11m fine-tuned", "20,056,092", "40.5", "Classroom behaviour, 4 classes"),
    ("SCRFD det_10g", "4,225,835", "16.9", "Find faces inside each person box"),
    ("ArcFace w600k_r50", "43,590,976", "174.4", "512-d identity embedding"),
    ("SixDRepNet", "39,323,398", "157.3", "Head pose: yaw, pitch, roll"),
    ("EfficientNet-B0 (emotion)", "3,996,789", "16.0", "Facial expression, 8 classes"),
    ("MediaPipe Pose (BlazePose)", "not exposed", "6.4", "33 body keypoints"),
    ("MediaPipe Face Mesh", "not exposed", "1.2", "468 face landmarks"),
]

PER_CLASS = [
    ["Class", "Precision", "Recall", "mAP@50", "mAP@50-95", "Train boxes"],
    ["write", "0.723", "0.681", "0.767", "0.331", "1,121"],
    ["using_device", "0.749", "0.700", "0.720", "0.336", "2,204"],
    ["sleep", "0.670", "0.495", "0.521", "0.263", "1,512"],
    ["read", "0.455", "0.469", "0.419", "0.143", "1,254"],
    ["all", "0.649", "0.586", "0.607", "0.268", "6,091"],
]


def s01_title(prs):
    s = new_slide(prs)
    bar(s, 0, 0, SW, 0.09, TEAL)
    add_text(s, ML, 2.05, CW, 0.3,
             [P("MODEL CARD DECK", 11, True, TEAL, FONT_SB)])
    add_text(s, ML, 2.44, CW, 1.1,
             [P("The eight models, and what each one is actually for",
                40, True, INK, FONT_SB, line=1.02)])
    add_text(s, ML, 3.80, CW * 0.78, 0.9,
             [P("Purpose, backbone, size, parameter count, training data, output, and "
                "the one place each model is allowed to be wrong. Every number here was "
                "read off the loaded model on this machine, not copied from a datasheet.",
                14, False, BODY, line=1.32)])
    bar(s, ML, 5.05, 1.5, 0.03, TEAL)
    add_text(s, ML, 5.30, CW, 0.9,
             [P("ClassGraph — classroom engagement from video", 13, True, INK, FONT_SB),
              P("Seven pre-trained models used as-is, one fine-tuned on classroom "
                "footage. Regenerate every figure with tools/model_facts.py",
                11, False, MUTE, line=1.26, space_before=6)])
    return s


def s02_overview(prs, page):
    s, y = chrome(prs, "The stack at a glance",
                  "Eight models, 131 M parameters, one of them ours",
                  "131.3 M is the six models whose parameters are countable; MediaPipe "
                  "ships TFLite graphs that do not expose a count. Each model does one "
                  "narrow job it was built for.", page)
    data = [["Model", "Parameters", "MB", "What it is for"]]
    for name, params, mb, purpose in MODELS:
        data.append([name, params, mb, purpose])
    tbl = table(s, ML, y, CW, [3.05, 2.05, 1.05, 5.82], data,
                row_h=0.36, head_h=0.36, size=10.5, head_size=10.5, col_bold={0})
    for r in range(1, len(data)):
        for c in (1, 2):
            run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
            run.font.name = MONO
            run.font.size = Pt(10)
        if data[r][0] == "YOLO11m fine-tuned":
            for c in range(4):
                run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
                run.font.color.rgb = TEAL_D
                run.font.bold = True

    yy = y + 0.36 + len(MODELS) * 0.36 + 0.24
    half = (CW - 0.30) / 2
    card(s, ML, yy, half, 1.22,
         heading="Why so many small models instead of one big one",
         lines=["A 4 M-parameter face detector and a 20 M-parameter object detector "
                "each solve a problem they were trained for. One network asked to do "
                "both would need far more data than we have, and would fail in ways we "
                "could not attribute to a stage."],
         accent=TEAL, heading_size=11, body_size=9.5, fill=PANEL)
    card(s, ML + half + 0.30, yy, half, 1.22,
         heading="Only one model was trained by us",
         lines=["Seven are used exactly as published. The eighth is YOLO11m fine-tuned "
                "on classroom behaviour, because no public model predicts "
                "read / write / sleep / using_device from a classroom camera. "
                "That is the next four slides."],
         accent=GREEN, heading_size=11, body_size=9.5, fill=PANEL2)
    return s


def s03_detection(prs, page):
    s, y = chrome(prs, "Model 1 of 8 — Detection",
                  "YOLO11m — find every person and every held object",
                  "The first stage. Everything downstream is cropped from the boxes "
                  "this model produces, so its failures are the only ones that cannot "
                  "be recovered later.", page)

    half = (CW - 0.30) / 2
    card(s, ML, y, half, 2.05,
         heading="What it is",
         lines=["Single-stage anchor-free detector, CSPDarknet-style backbone with a "
                "PAN-FPN neck and a decoupled detection head.",
                "20,114,688 parameters, 40.7 MB. Trained by Ultralytics on COCO — "
                "118,000 images, 80 object classes.",
                "Used exactly as published. No fine-tuning: COCO already contains "
                "person, cell phone, laptop, book and bottle, which is what we need."],
         accent=TEAL, heading_size=12, body_size=10, fill=PANEL)

    card(s, ML + half + 0.30, y, half, 2.05,
         heading="What it outputs, and how we use it",
         lines=["Per detection: a box, a class from the 80, and a confidence.",
                "Person boxes become the crop for every other model. Object boxes are "
                "assigned to exactly one student by largest overlap share — before that "
                "rule, 35% of phones were credited to more than one student.",
                "person_conf is 0.30, not the 0.40 default: a back-row student scores "
                "low, and missing one costs more than a spurious box we can filter."],
         accent=TEAL, heading_size=12, body_size=10, fill=PANEL2)

    yy = y + 2.25
    card(s, ML, yy, CW, 1.30,
         heading="The limitation a panel should know about — COCO has 80 classes, not 500",
         lines=["A spectacle case, a poster and a water bottle are not the same problem: "
                "the bottle is a COCO class and is detected, the other two are not and "
                "never will be by this model. We report what the model can name and stay "
                "silent otherwise, rather than guessing a label from shape.",
                "The classes we actually consume are person, cell phone, laptop, book, "
                "bottle, cup and keyboard. A held object outside that set is evidence "
                "that the hands are busy, which the action rules use without naming it."],
         accent=AMBER, heading_size=11.5, body_size=9.5, fill=PANEL)

    yy2 = yy + 1.50
    rect(s, ML, yy2, CW, 1.28, fill=PANEL2, line=TEAL, line_w=1.25)
    add_text(s, ML + 0.28, yy2 + 0.16, CW - 0.56, 1.00,
             [P("THE SETTING THAT MATTERED MOST", 9, True, TEAL_D, FONT_SB),
              P("Inference size is not a free parameter. imgsz 1920 suits a classroom "
                "camera, where students are small and enlarging the frame helps. On a "
                "640x480 webcam it enlarges 3x and the detector finds nobody at all — "
                "so the upscale is now capped at 1.5x, which costs nothing on classroom "
                "footage (962 -> 964 persons) and takes the webcam from 0 to 1.",
                11, False, INK, line=1.24, space_before=6)])
    return s


def s04_finetune_why(prs, page):
    s, y = chrome(prs, "Model 2 of 8 — Fine-tuning (1 / 3)",
                  "Why we had to train one model ourselves",
                  "Everything else in this deck is a model somebody else trained. This "
                  "is the one gap no public checkpoint filled.", page)

    third = (CW - 2 * 0.26) / 3
    blocks = [
        ("THE GAP", "COCO knows objects, not behaviour",
         "COCO can tell us a laptop is present. It cannot tell us whether the student "
         "is typing on it, reading from it, or asleep beside it. No public detector "
         "predicts classroom behaviour classes from a room camera."),
        ("THE CHOICE", "Fine-tune, do not train from scratch",
         "935 labelled images cannot train a 20 M-parameter detector from random "
         "weights. Starting from COCO weights means the backbone already knows edges, "
         "texture and human shape, and only the class semantics have to be learned."),
        ("THE SCOPE", "Four classes, deliberately few",
         "read, write, sleep, using_device. Each is a visible posture-and-object "
         "configuration. We did not add classes like 'bored' or 'confused' — those are "
         "interpretations, not things a camera can see."),
    ]
    for i, (tag, title, body) in enumerate(blocks):
        x = ML + i * (third + 0.26)
        rect(s, x, y, third, 2.55, fill=WHITE, line=BORDER)
        bar(s, x, y, third, 0.05, TEAL)
        add_text(s, x + 0.24, y + 0.22, third - 0.48, 0.24,
                 [P(tag, 9, True, TEAL, FONT_SB)])
        add_text(s, x + 0.24, y + 0.54, third - 0.48, 0.56,
                 [P(title, 13, True, INK, FONT_SB, line=1.10)])
        add_text(s, x + 0.24, y + 1.18, third - 0.48, 1.20,
                 [P(body, 10, False, BODY, line=1.22)])

    yy = y + 2.78
    card(s, ML, yy, CW, 1.50,
         heading="Transfer learning, stated precisely",
         lines=["A detector is a backbone that turns pixels into features, a neck that "
                "mixes them across scales, and a head that turns features into boxes and "
                "class scores. Only the head's meaning is task-specific; edges and human "
                "shape are the same in COCO and in a classroom.",
                "So we keep the COCO weights as the starting point and let gradient "
                "descent move all of them. The backbone barely moves because it is "
                "already right; the head moves a lot, because 80 COCO classes have to "
                "become 4 behaviour classes."],
         accent=GREEN, heading_size=12, body_size=10, fill=PANEL2)
    return s


def s05_finetune_how(prs, page):
    s, y = chrome(prs, "Model 2 of 8 — Fine-tuning (2 / 3)",
                  "Exactly what was trained, on what, with which settings",
                  "Read straight out of runs/behaviour/merged4_aug/args.yaml — the "
                  "configuration the run actually used, not one written down afterwards.",
                  page)

    half = (CW - 0.30) / 2
    cfg = [
        ["Setting", "Value", "Why"],
        ["Starting weights", "yolo11m.pt", "COCO-pretrained, not random"],
        ["Layers frozen", "none (freeze: null)", "all 20.1 M updated"],
        ["Dataset", "behaviour_merged", "877 train / 58 val images"],
        ["Labelled boxes", "6,091 train", "4 classes"],
        ["Epochs", "42 of 60", "early-stopped, patience 15"],
        ["Best epoch", "27", "later epochs overfit"],
        ["Image size", "640", "1600 needed 9.05 GB on a 6.4 GB card"],
        ["Batch", "8", "largest that fits in VRAM"],
        ["Optimiser / LR", "auto, lr0 0.01", "Ultralytics default"],
        ["Wall-clock", "23 min", "RTX 4050 laptop"],
    ]
    tbl = table(s, ML, y, half, [1.72, 1.85, 2.35], cfg,
                row_h=0.315, head_h=0.33, size=9.5, head_size=9.5, col_bold={0})
    for r in range(1, len(cfg)):
        run = tbl.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.name = MONO
        run.font.size = Pt(9)
        run.font.color.rgb = TEAL_D
        run.font.bold = True

    x2 = ML + half + 0.30
    card(s, x2, y, half, 2.10,
         heading="Which layers were fine-tuned? All of them.",
         lines=["args.yaml records freeze: null, so no layer was held fixed — every one "
                "of the 20.1 M parameters received gradients.",
                "This is the right choice at this scale. Freezing the backbone is for "
                "when the new data is tiny or very close to the original domain; a "
                "classroom seen from a high rear corner is neither. The COCO weights are "
                "the starting point, not a fixed feature extractor.",
                "The parameter count drops slightly, 20,114,688 to 20,056,092, because "
                "the detection head now predicts 4 classes instead of 80."],
         accent=GREEN, heading_size=12, body_size=9.5, fill=PANEL2)

    yy = y + 2.30
    card(s, x2, yy, half, 1.55,
         heading="The dataset, and its honest weakness",
         lines=["935 images total, merged from classroom sets and hand-checked. Train "
                "boxes per class: using_device 2,204, sleep 1,512, read 1,254, "
                "write 1,121.",
                "The validation split is 58 images. That is small enough that a single "
                "hard image moves the metric, and we say so rather than quoting the "
                "number as if it were a benchmark result."],
         accent=AMBER, heading_size=12, body_size=9.5, fill=PANEL)
    return s


def s06_finetune_result(prs, page):
    s, y = chrome(prs, "Model 2 of 8 — Fine-tuning (3 / 3)",
                  "What fine-tuning bought, and where it still fails",
                  "Re-validated from the trained weights for this deck, not copied from "
                  "the training log.", page)

    tw = 7.55
    tbl = table(s, ML, y, tw, [1.75, 1.20, 1.05, 1.15, 1.35, 1.05], PER_CLASS,
                row_h=0.40, head_h=0.38, size=10, head_size=9.5, col_bold={0})
    for r in range(1, len(PER_CLASS)):
        for c in range(1, 6):
            run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
            run.font.name = MONO
            run.font.size = Pt(9.5)
        name = PER_CLASS[r][0]
        colour = (INK if name == "all" else
                  GREEN if r <= 2 else (AMBER if r == 3 else RED))
        run = tbl.cell(r, 0).text_frame.paragraphs[0].runs[0]
        run.font.color.rgb = colour
        run.font.bold = True
        tbl.cell(r, 3).text_frame.paragraphs[0].runs[0].font.color.rgb = colour
        tbl.cell(r, 3).text_frame.paragraphs[0].runs[0].font.bold = True

    yy = y + 0.38 + len(PER_CLASS) * 0.40 + 0.24
    card(s, ML, yy, tw, 1.42,
         heading="Did fine-tuning improve accuracy? Yes — measurably.",
         lines=["mAP@50 on the validation split rose from 0.188 after the first epoch "
                "to 0.607 at the best epoch: a 3.2x improvement as the head learned the "
                "four classes. The COCO model scores zero on this task by construction, "
                "because it cannot emit these labels at all.",
                "Early stopping at epoch 42 with the best at 27 is the useful detail: "
                "the model stopped improving on unseen images long before it stopped "
                "improving on the training set."],
         accent=GREEN, heading_size=11.5, body_size=9.5, fill=PANEL2)

    x2 = ML + tw + 0.30
    w2 = SW - MR - x2
    card(s, x2, y, w2, 2.35,
         heading="'read' is the weakest class — and we predicted it",
         lines=["read scores mAP@50 0.419 against write at 0.767. That is not noise: "
                "reading and writing differ only by what the hands are doing, and the "
                "closest published work reaches 57.8% on writing even with a strong "
                "temporal model.",
                "So the pipeline does not present them as equals. When a book is visible "
                "but the hands are not, the reported action is 'reading or writing' — one "
                "label covering both — and writing is flagged inferred, never direct."],
         accent=RED, heading_size=12, body_size=9.5, fill=PANEL)

    card(s, x2, y + 2.55, w2, 2.05,
         heading="Where this model sits in the pipeline",
         lines=["It is a second opinion, not the primary signal. Actions are decided "
                "first by geometry — object overlap, wrist position, head pitch — which "
                "is auditable and needs no training data.",
                "The behaviour model supplies a label where geometry is silent. Keeping "
                "it subordinate is why a 0.607 mAP model is safe to include: its "
                "mistakes are visible as a disagreement, not as the only answer."],
         accent=TEAL, heading_size=12, body_size=9.5, fill=PANEL2)
    return s


def s07_identity(prs, page):
    s, y = chrome(prs, "Models 3 and 4 of 8 — Identity",
                  "SCRFD finds the face, ArcFace says whose it is",
                  "Identity is what makes a per-frame observation into a student's "
                  "history. Both models are used exactly as published.", page)

    half = (CW - 0.30) / 2
    card(s, ML, y, half, 2.55,
         heading="SCRFD det_10g — face detection",
         lines=["4,225,835 parameters, 16.9 MB, ONNX on GPU. Part of InsightFace's "
                "buffalo_l pack, trained on WIDER FACE.",
                "Sample-redistribution single-stage detector: it spends most of its "
                "compute on the small-face scales, which is exactly the classroom case.",
                "Output per face: a box, a confidence, and 5 keypoints (both eyes, nose, "
                "two mouth corners). The keypoints matter — they align the crop before "
                "recognition and before expression."],
         accent=TEAL, heading_size=12, body_size=9.5, fill=PANEL)

    card(s, ML + half + 0.30, y, half, 2.55,
         heading="ArcFace w600k_r50 — identity embedding",
         lines=["43,590,976 parameters, 174.4 MB — the largest model in the stack. "
                "ResNet-50 backbone, trained on WebFace600K.",
                "Output: one 512-dimensional unit vector per face. Not a name and not a "
                "class — a point on a hypersphere, where the same person lands close "
                "together and different people land far apart.",
                "The additive angular margin in its training loss is what forces that "
                "separation, which is why cosine distance between two embeddings is a "
                "meaningful identity score at all."],
         accent=TEAL, heading_size=12, body_size=9.5, fill=PANEL2)

    yy = y + 2.75
    card(s, ML, yy, CW, 1.85,
         heading="What we build on top — the part that is ours",
         lines=["A raw embedding match is not enough in a classroom. Two students can "
                "look alike, and the same student's embedding drifts as they turn. So "
                "identity is resolved by constrained agglomerative clustering over "
                "cosine distance, with a hard cannot-link between tracks that appear in "
                "the same frame — one body cannot be two people at once, and that "
                "constraint is free information the embedding does not carry.",
                "On registration each student is enrolled once and keeps a fixed ID. "
                "98.6% of detections receive an ID; where no face is good enough, the "
                "detection is reported as unidentified rather than attributed to the "
                "nearest match — 6 of 42 identities in the 60-clip run were refused on "
                "exactly this ground."],
         accent=GREEN, heading_size=12, body_size=9.5, fill=PANEL)
    return s


def s08_geometry(prs, page):
    s, y = chrome(prs, "Models 5, 6 and 7 of 8 — Geometry",
                  "Head pose and landmarks: the measurements behind every action",
                  "These three produce no labels at all. They produce numbers, and the "
                  "action rules read those numbers.", page)

    third = (CW - 2 * 0.26) / 3
    items = [
        ("SixDRepNet", TEAL,
         ["39,323,398 parameters, 157 MB. RepVGG backbone. Checkpoint 6DRepNet_300W_LP_AFLW2000 — trained on 300W-LP, evaluated on AFLW2000.",
          "Predicts head rotation as a 6-D continuous representation rather than three "
          "Euler angles — Euler angles are discontinuous at their wrap-around, which "
          "makes them a poor regression target.",
          "Output: yaw, pitch, roll in degrees. We use pitch to separate a bowed head "
          "from a raised one. Yaw is now a fallback only, because it is measured "
          "relative to the camera and every seat has a different angle to it."]),
        ("MediaPipe Pose", TEAL,
         ["BlazePose, 6.4 MB TFLite, CPU. model_complexity 1, so the \"full\" landmark model. Two-stage: a detector, then a landmark model.",
          "Output: 33 body keypoints with visibility scores — shoulders, hips, wrists, "
          "nose.",
          "Two things depend on it. Wrist position relative to the face and the desk "
          "gives raised hand, head-on-hand and hands-low. Shoulder direction gives "
          "facing_direction in image space, which is the input to the room-layout "
          "measurement and carries no camera constant."]),
        ("MediaPipe Face Mesh", TEAL,
         ["1.2 MB TFLite, CPU. Its own detector is bypassed — it runs on the crop SCRFD "
          "already found, so the two never disagree about where the face is.",
          "Output: 478 landmarks with refinement on; we keep the canonical 468 and drop the 10 iris points, because the frozen output schema says 468. Refinement still sharpens the eye points the EAR depends on.",
          "We derive two scalars: eye aspect ratio for closed eyes, and mouth opening "
          "ratio for a yawn. Both are ratios of distances between landmarks, so they are "
          "scale-invariant — a face near the camera and one far away give comparable "
          "numbers."]),
    ]
    for i, (title, accent, lines) in enumerate(items):
        x = ML + i * (third + 0.26)
        rect(s, x, y, third, 3.55, fill=WHITE, line=BORDER)
        bar(s, x, y, third, 0.05, accent)
        add_text(s, x + 0.24, y + 0.24, third - 0.48, 0.34,
                 [P(title, 14, True, INK, FONT_SB)])
        blocks = [P(ln, 9.5, False, BODY, line=1.22, space_before=7) for ln in lines]
        add_text(s, x + 0.24, y + 0.68, third - 0.48, 2.70, blocks)

    yy = y + 3.75
    rect(s, ML, yy, CW, 1.10, fill=PANEL2, line=TEAL, line_w=1.25)
    add_text(s, ML + 0.28, yy + 0.15, CW - 0.56, 0.85,
             [P("WHY THESE ARE THE MOST IMPORTANT MODELS IN THE STACK", 9, True,
                TEAL_D, FONT_SB),
              P("A classifier gives a label you must trust. These give a measurement you "
                "can check. When a threshold is wrong, the number shows it — which is how "
                "the yawn threshold was found to be unreachable (set at 0.55, the signal "
                "never exceeded 0.274) and how the eye ratio was found to be saturated.",
                11, False, INK, line=1.24, space_before=6)])
    return s


def s09_expression(prs, page):
    s, y = chrome(prs, "Model 8 of 8 — Expression",
                  "EfficientNet-B0, and why we report 3 labels from its 8",
                  "The smallest model in the stack, and the one where we most "
                  "deliberately discard information.", page)

    half = (CW - 0.30) / 2
    card(s, ML, y, half, 2.45,
         heading="What it is",
         lines=["EfficientNet-B0 backbone, 3,996,789 parameters, 16.0 MB ONNX. Run on "
                "CPU on purpose: YOLO and SixDRepNet already occupy the 6.4 GB card.",
                "Checkpoint enet_b0_8_best_vgaf from EmotiEffLib — trained on AffectNet "
                "and validated on VGAF, an in-the-wild video group-affect set, which is "
                "much closer to a classroom than a posed studio dataset.",
                "Input 224x224. Output: 8 AffectNet class probabilities."],
         accent=TEAL, heading_size=12, body_size=9.5, fill=PANEL)

    card(s, ML + half + 0.30, y, half, 2.45,
         heading="The 8 to 3 mapping, and why it is not lossy in the file",
         lines=["Reported: happy, sad, neutral. Anger, Contempt, Disgust, Fear and "
                "Surprise all map to neutral.",
                "They map to neutral rather than to sad on purpose. Folding anger into "
                "sadness would assert something the model never said; mapping it to "
                "neutral says only 'not one of the three we report', which is true.",
                "The full 8-class distribution is kept in the output and the mapping "
                "lives in config, so the collapse is auditable and reversible — a "
                "reviewer can ask for the other five and get them."],
         accent=GREEN, heading_size=12, body_size=9.5, fill=PANEL2)

    yy = y + 2.65
    card(s, ML, yy, CW, 1.65,
         heading="Alignment: a measured decision, not a default",
         lines=["AffectNet was trained on aligned faces, so feeding a raw box crop puts "
                "the model out of distribution. We align each crop using SCRFD's five "
                "keypoints before classifying.",
                "On the 60-clip external run this model produced 9,321 classifications — "
                "neutral 4,011, happy 707, sad 665 — and 3,468 came back 'uncertain'. "
                "Those are faces too small or too turned to classify, and they are "
                "reported as uncertain rather than defaulted to neutral, which would "
                "have silently inflated the neutral count by 37%."],
         accent=AMBER, heading_size=12, body_size=9.5, fill=PANEL)
    return s


def s10_summary(prs, page):
    s, y = chrome(prs, "Summary",
                  "What each model is allowed to be wrong about",
                  "The question a panel is really asking is not how big each model is, "
                  "but what happens when it fails.", page)

    data = [
        ["Model", "Output", "Failure mode", "How the system contains it"],
        ["YOLO11m", "boxes + 80 classes", "misses a small student",
         "conf floor 0.30; upscale capped at 1.5x"],
        ["YOLO11m fine-tuned", "4 behaviour classes", "confuses read with write",
         "reported as 'reading or writing', marked inferred"],
        ["SCRFD", "face box + 5 points", "no face when head is down",
         "body pose recovers the student; else unknown"],
        ["ArcFace", "512-d embedding", "two students look alike",
         "cannot-link between co-occurring tracks"],
        ["SixDRepNet", "yaw, pitch, roll", "yaw is camera-relative",
         "demoted to fallback; room layout used instead"],
        ["MediaPipe Pose", "33 keypoints", "fits a skeleton to furniture",
         "requires both shoulders before a lean is reported"],
        ["MediaPipe Face Mesh", "468 landmarks", "eye ratio saturates",
         "known open — needs open-eye calibration"],
        ["EfficientNet-B0", "8 emotion classes", "small or turned faces",
         "reported 'uncertain', never defaulted to neutral"],
    ]
    tbl = table(s, ML, y, CW, [2.35, 2.05, 2.55, 5.02], data,
                row_h=0.40, head_h=0.38, size=9.5, head_size=9.5, col_bold={0})
    for r in range(1, len(data)):
        tbl.cell(r, 2).text_frame.paragraphs[0].runs[0].font.color.rgb = RED
        run = tbl.cell(r, 3).text_frame.paragraphs[0].runs[0]
        run.font.color.rgb = GREEN
        run.font.bold = True

    yy = y + 0.38 + 8 * 0.40 + 0.26
    rect(s, ML, yy, CW, 1.10, fill=PANEL2, line=TEAL, line_w=1.25)
    add_text(s, ML + 0.28, yy + 0.15, CW - 0.56, 0.85,
             [P("THE ONE ROW WE WOULD DEFEND HARDEST", 9, True, TEAL_D, FONT_SB),
              P("Every row's containment strategy ends in the same place: when a model "
                "cannot answer, the system says so instead of substituting its best "
                "guess. 14 of 42 identities in the external run were refused on those "
                "grounds, each with the count behind the refusal recorded.",
                11, False, INK, line=1.24, space_before=6)])
    return s



# --------------------------------------------------------------------------- #
# Diagram primitives
# --------------------------------------------------------------------------- #

def node(slide, x, y, w, h, title, sub=None, fill=WHITE, accent=TEAL,
         title_size=10.5, sub_size=8.5):
    """One labelled box in a diagram."""
    rect(slide, x, y, w, h, fill=fill, line=BORDER)
    bar(slide, x, y, w, 0.045, accent)
    blocks = [P(title, title_size, True, INK, FONT_SB, line=1.08,
                align=PP_ALIGN.CENTER)]
    # A newline inside a single run lands in the XML as a raw newline, which
    # PowerPoint renders as a space -- so each line has to be its own paragraph.
    for i, part in enumerate((sub or "").splitlines()):
        if part:
            blocks.append(P(part, sub_size, False, MUTE, line=1.14,
                            align=PP_ALIGN.CENTER,
                            space_before=3 if i == 0 else 0))
    add_text(slide, x + 0.08, y + 0.14, w - 0.16, h - 0.22, blocks,
             anchor=MSO_ANCHOR.MIDDLE)


def arrow(slide, x1, y1, x2, y2, colour=TEAL, width=1.5, dashed=False):
    """A straight connector between two points, in inches."""
    from pptx.enum.shapes import MSO_CONNECTOR
    from pptx.oxml.ns import qn
    cx = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),
                                    Inches(y1), Inches(x2), Inches(y2))
    cx.line.color.rgb = colour
    cx.line.width = Pt(width)
    ln = cx.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle",
                                            "w": "med", "len": "med"})
    ln.append(tail)
    if dashed:
        dash = ln.makeelement(qn("a:prstDash"), {"val": "dash"})
        ln.insert(0, dash)
    return cx


def lane(slide, x, y, w, h, label, fill=PANEL, accent=TEAL):
    """A titled band grouping several nodes."""
    rect(slide, x, y, w, h, fill=fill, line=BORDER)
    add_text(slide, x + 0.14, y + 0.10, 2.6, 0.22,
             [P(label.upper(), 8.5, True, accent, FONT_SB)])


# --------------------------------------------------------------------------- #

def s_architecture(prs, page):
    """Every model, what feeds it, and what it feeds."""
    s, y = chrome(prs, "Architecture",
                  "The whole system, and what each stage hands the next",
                  "Read left to right. Nothing in a later stage can recover what an "
                  "earlier stage missed, which is why the detector's settings matter "
                  "more than any threshold downstream.", page)

    col_w, gap = 2.15, 0.30
    xs = [ML + i * (col_w + gap) for i in range(5)]
    top = y + 0.30

    # Stage headers
    heads = ["INPUT", "DETECT", "PER-PERSON MODELS", "DERIVE", "OUTPUT"]
    for i, htxt in enumerate(heads):
        add_text(s, xs[i], y, col_w, 0.24,
                 [P(htxt, 8.5, True, TEAL, FONT_SB, align=PP_ALIGN.CENTER)])

    # Column 1 - input
    node(s, xs[0], top, col_w, 0.86, "Frame",
         "webcam 640x480, or\nvideo at 1080p", PANEL2, TEAL)
    node(s, xs[0], top + 1.06, col_w, 0.86, "Enrolled gallery",
         "name -> 512-d vector\nregistered once", PANEL2, GREEN)

    # Column 2 - detection
    node(s, xs[1], top, col_w, 1.20, "YOLO11m",
         "20.1 M params\nboxes + 80 COCO classes", WHITE, TEAL)
    node(s, xs[1], top + 1.40, col_w, 1.00, "ByteTrack",
         "track_id across frames", WHITE, TEAL)

    # Column 3 - per-person models
    per = [("SCRFD det_10g", "4.2 M - face box + 5 kps"),
           ("ArcFace w600k_r50", "43.6 M - 512-d embedding"),
           ("MediaPipe Pose", "33 keypoints, facing ray"),
           ("MediaPipe Face Mesh", "468 landmarks, EAR"),
           ("SixDRepNet", "39.3 M - yaw/pitch/roll"),
           ("EfficientNet-B0", "4.0 M - 8 emotions")]
    ph = 0.60
    for i, (nm, sub) in enumerate(per):
        node(s, xs[2], top + i * (ph + 0.09), col_w, ph, nm, sub, WHITE, AMBER,
             title_size=9.5, sub_size=7.5)

    # Column 4 - derivation
    node(s, xs[3], top, col_w, 0.92, "Identity resolver",
         "clustering + cannot-link\n-> stable person_id", WHITE, GREEN)
    node(s, xs[3], top + 1.06, col_w, 0.92, "Scene layout",
         "facing rays -> focus\ngroup vs lecture", PANEL2, GREEN)
    node(s, xs[3], top + 2.12, col_w, 0.92, "Action rules",
         "geometry first,\n17 actions + evidence", WHITE, GREEN)
    node(s, xs[3], top + 3.18, col_w, 0.92, "Temporal tracker",
         "blink vs closure,\nrolling engagement", WHITE, GREEN)

    # Column 5 - outputs
    node(s, xs[4], top, col_w, 1.00, "raw.jsonl",
         "one record per frame\nevery model's output", PANEL2, TEAL)
    node(s, xs[4], top + 1.16, col_w, 1.00, "live_graph.jsonl",
         "nodes = students\nedges = relations", PANEL2, TEAL)
    node(s, xs[4], top + 2.32, col_w, 1.00, "profiles + report",
         "per-student history,\ntwo scores, graphs", PANEL2, TEAL)

    # Flow arrows between columns
    mid = top + 0.60
    arrow(s, xs[0] + col_w, mid, xs[1], mid)
    arrow(s, xs[1] + col_w, mid, xs[2], mid)
    arrow(s, xs[2] + col_w, top + 1.80, xs[3], top + 1.06)
    arrow(s, xs[3] + col_w, top + 1.80, xs[4], top + 1.16)
    # The gallery feeds identity directly, not the detector.
    arrow(s, xs[0] + col_w, top + 1.50, xs[3], top + 0.46, GREEN, 1.25, True)

    yy = SH - 1.02
    rect(s, ML, yy, CW, 0.80, fill=PANEL2, line=TEAL, line_w=1.25)
    add_text(s, ML + 0.24, yy + 0.13, CW - 0.48, 0.58,
             [PR([R("The dashed line is the point people miss:  ", 10, True,
                    TEAL_D, FONT_SB),
                  R("enrolment does not change detection. A registered student is "
                    "found the same way as anyone else; the gallery only supplies the "
                    "name once a face is good enough to match, and stays silent when "
                    "it is not.", 10, False, INK)], line=1.20)])
    return s


def s_workflow(prs, page):
    """The pipeline as it runs, including what happens when a stage says nothing."""
    s, y = chrome(prs, "Workflow",
                  "One frame, end to end - including every branch that gives up",
                  "The branches matter as much as the path. Most of this system is "
                  "decisions about what to do when a model cannot answer.", page)

    bw, bh, gap = 1.78, 0.72, 0.30
    row1 = y + 0.34
    steps = [
        ("1  Capture", "frame + timestamp"),
        ("2  Detect", "persons, objects"),
        ("3  Track", "track_id"),
        ("4  Face", "SCRFD box + kps"),
        ("5  Identify", "ArcFace -> person_id"),
        ("6  Measure", "pose, mesh, head"),
    ]
    for i, (t, sub) in enumerate(steps):
        x = ML + i * (bw + gap)
        node(s, x, row1, bw, bh, t, sub, WHITE, TEAL, 10, 8)
        if i:
            arrow(s, x - gap, row1 + bh / 2, x, row1 + bh / 2)

    row2 = row1 + bh + 0.92
    steps2 = [
        ("7  Layout", "rays -> group / lecture"),
        ("8  Classify", "17 actions + evidence"),
        ("9  Expression", "8 classes -> 3"),
        ("10  Temporal", "blink vs closure"),
        ("11  Graph", "nodes + 4 edge types"),
        ("12  Report", "profiles, scores, HTML"),
    ]
    for i, (t, sub) in enumerate(steps2):
        x = ML + i * (bw + gap)
        node(s, x, row2, bw, bh, t, sub, WHITE, GREEN, 10, 8)
        if i:
            arrow(s, x - gap, row2 + bh / 2, x, row2 + bh / 2, GREEN)
    # wrap from step 6 to step 7
    arrow(s, ML + 5 * (bw + gap) + bw / 2, row1 + bh,
          ML + bw / 2, row2, TEAL, 1.25, True)

    # The give-up branches, called out beneath the stage they belong to.
    row3 = row2 + bh + 0.46
    outs = [
        ("no person box", "nothing downstream runs", 1),
        ("no readable face", "body pose carries the student", 3),
        ("no identity match", "reported unidentified, not guessed", 4),
        ("fewer than 3 people", "layout unknown, second score blank", 6),
        ("no evidence at all", "action is unknown, never attentive", 7),
    ]
    ow = (CW - 4 * 0.22) / 5
    for i, (cond, act, _) in enumerate(outs):
        x = ML + i * (ow + 0.22)
        rect(s, x, row3, ow, 0.92, fill=PANEL, line=BORDER)
        bar(s, x, row3, 0.045, 0.92, AMBER)
        add_text(s, x + 0.18, row3 + 0.14, ow - 0.32, 0.68,
                 [P(cond, 9, True, AMBER, FONT_SB, line=1.10),
                  P(act, 8.5, False, BODY, line=1.16, space_before=4)])

    yy = row3 + 1.10
    rect(s, ML, yy, CW, 0.72, fill=PANEL2, line=TEAL, line_w=1.25)
    add_text(s, ML + 0.24, yy + 0.12, CW - 0.48, 0.50,
             [PR([R("Steps 1-6 are perception, 7-12 are interpretation.  ", 10,
                    True, TEAL_D, FONT_SB),
                  R("Everything in the top row is a measurement some model makes; "
                    "everything in the bottom row is a decision we make about those "
                    "measurements, and is the part we can defend line by line.",
                    10, False, INK)], line=1.20)])
    return s


def s_io_overview(prs, page):
    """The same frame, drawn by four models."""
    s, y = chrome(prs, "Model outputs",
                  "All eight models, on one frame",
                  "Seven panels are the same classroom frame. Expression is the exception, and the caption says why.", page)
    img = ASSETS / "model_io_grid.jpg"
    if img.exists():
        from PIL import Image
        with Image.open(img) as im:
            ratio = im.height / im.width
        w = 6.30
        s.shapes.add_picture(str(img), Inches(ML), Inches(y), width=Inches(w))
        h = w * ratio
    else:
        w, h = 6.30, 3.15

    x2 = ML + w + 0.34
    w2 = SW - MR - x2
    card(s, x2, y, w2, 1.62,
         heading="Why one frame, not eight demo images",
         lines=["The pipeline is a chain. SCRFD only searches inside YOLO's person "
                "boxes; Face Mesh, SixDRepNet and the expression model all run on "
                "SCRFD's crop. Separate demo images would hide the dependency that "
                "makes the detector's settings the most consequential in the system."],
         accent=TEAL, heading_size=11.5, body_size=9.5, fill=PANEL2)

    card(s, x2, y + 1.80, w2, 1.62,
         heading="Read the counts across the panels",
         lines=["13 persons, 12 faces, 12 head poses, 10 skeletons, 11 behaviour "
                "detections. Pose needs both shoulders in view, so the three it misses "
                "are the students whose torsos are behind a desk or a neighbour — not "
                "the smallest ones, which it gets.",
                "Expression ran on a different frame on purpose: on this one every face "
                "is 9-23 px against a 25 px minimum, so the model declined all twelve. "
                "That is the correct answer, and a blank panel would not have shown it."],
         accent=AMBER, heading_size=11.5, body_size=9.5, fill=PANEL)

    card(s, x2, y + 3.60, w2, 1.30,
         heading="Four kinds of answer",
         lines=["YOLO and SCRFD answer WHERE. Pose and Face Mesh answer HOW, as "
                "continuous numbers you can argue with. SixDRepNet answers WHICH WAY. "
                "ArcFace answers WHO. Only the behaviour and expression models emit a "
                "label you must simply trust — which is why both stay subordinate to "
                "geometry."],
         accent=GREEN, heading_size=11.5, body_size=9.5, fill=PANEL2)
    return s


def s_io_yolo(prs, page):
    """YOLO in, YOLO out - picture and JSON, side by side."""
    s, y = chrome(prs, "Input and output - YOLO11m",
                  "Give it an image; get boxes, classes and scores",
                  "The JSON on the right is copied from outputs/final2/raw.jsonl, a "
                  "real 60-clip run - not an illustration.", page)

    img = ASSETS / "model_io_yolo.jpg"
    w = 3.85
    if img.exists():
        s.shapes.add_picture(str(img), Inches(ML), Inches(y), width=Inches(w))

    x2 = ML + w + 0.32
    w2 = SW - MR - x2
    rect(s, x2, y, w2, 2.54, fill=RGBColor(0x0E, 0x1A, 0x26), line=None)
    add_text(s, x2 + 0.20, y + 0.12, w2 - 0.40, 2.30,
             [P("WHAT IT WRITES TO JSON", 8.5, True, TEAL, FONT_SB),
              P('"persons": [{', 9, False, WHITE, MONO, space_before=6),
              P('    "bbox": [765, 612, 301, 443],', 9, False, GREEN, MONO),
              P('    "confidence": 0.8949,', 9, False, GREEN, MONO),
              P('    "source": "yolo",  "track_id": 5', 9, False, GREEN, MONO),
              P('}],', 9, False, WHITE, MONO),
              P('"objects": [{', 9, False, WHITE, MONO, space_before=4),
              P('    "cls": "laptop",', 9, False, AMBER, MONO),
              P('    "bbox": [632, 475, 194, 235],', 9, False, AMBER, MONO),
              P('    "confidence": 0.9084', 9, False, AMBER, MONO),
              P('}]', 9, False, WHITE, MONO)])

    yy = y + 2.70
    card(s, x2, yy, w2, 1.10,
         heading="What the output image contains",
         lines=["A rectangle per detection with its class and score. Green boxes are "
                "people, amber are objects. Nothing is identified yet — at this stage "
                "every person is anonymous, and the box is all we have."],
         accent=TEAL, heading_size=11, body_size=9.5, fill=PANEL)

    card(s, x2, yy + 1.24, w2, 1.48,
         heading="How it is useful overall",
         lines=["The bbox is the crop every other model runs on, so it decides what the "
                "rest of the system can even see.",
                "Object boxes become evidence for an action: a phone overlapping a "
                "student is on_phone with confidence 'direct'. Each object is assigned "
                "to exactly one student by largest overlap share — without that rule, "
                "35% of phones were credited to more than one person."],
         accent=GREEN, heading_size=11, body_size=9.5, fill=PANEL2)
    return s


def s_io_rest(prs, page):
    """Every remaining model: output field, and what it unlocks."""
    s, y = chrome(prs, "Input and output - the other seven",
                  "What each one writes, and which decision depends on it",
                  "Field names are the real keys in raw.jsonl and live_graph.jsonl, so "
                  "a reviewer can open the file and find them.", page)

    data = [
        ["Model", "JSON it writes", "Example value", "What that makes possible"],
        ["SCRFD", 'face.bbox, kps',
         "[1099,231,63,82]", "crop + alignment for the next two models"],
        ["ArcFace", "512-d vector (internal)",
         "cosine 0.41 -> match", "person_id that survives the whole lecture"],
        ["MediaPipe Pose", "posture.facing_direction",
         "[-0.076, 0.997]", "room layout: group work or lecture"],
        ["MediaPipe Pose", "posture.vertical_lean",
         "-0.191", "slouching, leaning forward, head down"],
        ["MediaPipe Pose", "posture.left_wrist",
         "[1105.6, 458.0]", "raised hand, head-on-hand, writing"],
        ["Face Mesh", "face.ear",
         "0.2746", "eyes closed, once it lasts over 600 ms"],
        ["Face Mesh", "468 landmarks",
         "mouth ratio 0.04", "yawning (threshold still uncalibrated)"],
        ["SixDRepNet", "head_pose.pitch",
         "-3.74 deg", "bowed head vs raised head"],
        ["EfficientNet-B0", "expression.distribution",
         "8 probabilities", "happy / sad / neutral, rest kept for audit"],
        ["YOLO11m tuned", "behaviour.label",
         "using_device 0.72", "second opinion where geometry is silent"],
    ]
    tbl = table(s, ML, y, CW, [2.05, 2.55, 2.35, 5.02], data,
                row_h=0.335, head_h=0.35, size=9.5, head_size=9.5, col_bold={0})
    for r in range(1, len(data)):
        for c in (1, 2):
            run = tbl.cell(r, c).text_frame.paragraphs[0].runs[0]
            run.font.name = MONO
            run.font.size = Pt(9)
            run.font.color.rgb = TEAL_D if c == 1 else INK

    yy = y + 0.35 + (len(data) - 1) * 0.335 + 0.24
    half = (CW - 0.30) / 2
    card(s, ML, yy, half, 1.24,
         heading="The one output that is not a number",
         lines=["ArcFace's 512-d vector never reaches the JSON. It is consumed by the "
                "identity resolver and discarded, because a face embedding is "
                "biometric data and the files are meant to be shareable. What survives "
                "is an integer person_id."],
         accent=GREEN, heading_size=11, body_size=9.5, fill=PANEL2)
    card(s, ML + half + 0.30, yy, half, 1.24,
         heading="Why every row ends in a decision, not a label",
         lines=["A panel will ask what the models are for. The honest answer is that "
                "none of them decides anything: they produce measurements, and the "
                "action rules in backend/actions.py turn those into one of 17 actions "
                "with the evidence string that produced it."],
         accent=TEAL, heading_size=11, body_size=9.5, fill=PANEL)
    return s


def s_io_json(prs, page):
    """The full record for one student, annotated by which model wrote each part."""
    s, y = chrome(prs, "The record for one student, one frame",
                  "Every model's contribution, in the file a reviewer can open",
                  "Real values from outputs/final2. Colour shows which model wrote "
                  "each field.", page)

    x2 = ML
    w2 = CW * 0.545
    rect(s, x2, y, w2, 4.35, fill=RGBColor(0x0E, 0x1A, 0x26), line=None)
    rows = [
        ('{', WHITE), ('  "track_id": 5,  "person_id": 18,', GREEN),
        ('  "bbox": [765, 612, 301, 443],', GREEN),
        ('  "confidence": 0.8949,  "source": "yolo",', GREEN),
        ('  "face": {', WHITE),
        ('      "bbox": [817, 652, 67, 72],', CYAN_J := RGBColor(0x3C, 0xB8, 0xC8)),
        ('      "ear": 0.2746, "landmarks": [...468]', CYAN_J),
        ('  },', WHITE),
        ('  "head_pose": {', WHITE),
        ('      "yaw": 64.71, "pitch": -3.74,', AMBER),
        ('      "roll": 15.62, "gaze_label": "right"', AMBER),
        ('  },', WHITE),
        ('  "posture": {', WHITE),
        ('      "vertical_lean": -0.191,', MAGENTA_J := RGBColor(0xC8, 0x8C, 0xD8)),
        ('      "facing_direction": [-0.076, 0.997],', MAGENTA_J),
        ('      "left_wrist": [1105.6, 458.0], ...', MAGENTA_J),
        ('  },', WHITE),
        ('  "expression": {', WHITE),
        ('      "label": "sad", "confidence": 0.557,', RGBColor(0xE8, 0xB4, 0x5C)),
        ('      "distribution": { ...8 classes }', RGBColor(0xE8, 0xB4, 0x5C)),
        ('  },', WHITE),
        ('  "behaviour": null', MUTE), ('}', WHITE),
    ]
    add_text(s, x2 + 0.22, y + 0.14, w2 - 0.44, 4.07,
             [P("raw.jsonl - one person, one frame", 8.5, True, TEAL, FONT_SB)]
             + [P(t, 9, False, c, MONO, line=1.24) for t, c in rows])

    x3 = ML + w2 + 0.32
    w3 = SW - MR - x3
    rect(s, x3, y, w3, 2.30, fill=RGBColor(0x0E, 0x1A, 0x26), line=None)
    add_text(s, x3 + 0.22, y + 0.14, w3 - 0.44, 2.02,
             [P("live_graph.jsonl - what we derive from it", 8.5, True, GREEN,
                FONT_SB),
              P('"action": "on_phone",', 9.5, False, GREEN, MONO, space_before=6),
              P('"action_evidence": "cell phone overlap",', 9.5, False, GREEN, MONO),
              P('"action_confidence": "direct",', 9.5, False, GREEN, MONO),
              P('"object": "cell phone",', 9.5, False, GREEN, MONO),
              P('"layout": "unknown",  "oriented": false,', 9.5, False, WHITE, MONO),
              P('"focus_offset_deg": 129.8,', 9.5, False, WHITE, MONO),
              P('"engagement": "off",', 9.5, False, WHITE, MONO),
              P('"rolling_engagement_pct": 0.0', 9.5, False, WHITE, MONO)])

    card(s, x3, y + 2.48, w3, 1.87,
         heading="This is the slide to leave on screen during questions",
         lines=["Every field traces to a model on the left and a decision on the right. "
                "action_evidence is the field worth pointing at: it records the reason "
                "in words, so a wrong label can be argued with rather than merely "
                "disbelieved.",
                "behaviour is null here because the fine-tuned model had nothing to add "
                "once the phone overlap already settled it — geometry wins, and the "
                "record shows that it did."],
         accent=TEAL, heading_size=11.5, body_size=9.5, fill=PANEL2)
    return s


def main():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    s01_title(prs)
    s02_overview(prs, 2)
    s03_detection(prs, 3)
    s04_finetune_why(prs, 4)
    s05_finetune_how(prs, 5)
    s06_finetune_result(prs, 6)
    s07_identity(prs, 7)
    s08_geometry(prs, 8)
    s09_expression(prs, 9)
    s_architecture(prs, 10)
    s_workflow(prs, 11)
    s_io_overview(prs, 12)
    s_io_yolo(prs, 13)
    s_io_rest(prs, 14)
    s_io_json(prs, 15)
    s10_summary(prs, 16)

    prs.save(str(OUT))
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
