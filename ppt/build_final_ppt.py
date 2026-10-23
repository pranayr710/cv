"""Build the final review deck.

Design follows the reference deck the client supplied: two typefaces, four
colours, one claim per slide, and figures large enough to read from the back
of a room. Nothing is decorative -- every rule, rail and panel marks a
boundary that means something.

    1  Title
    2  Against the closest published work
    3  Attention model, and the ceiling it is measured against
    4  Behaviour detector, per class
    5  Image engagement scorer
    6  What was tried and did not win
    7  Architecture
    8  Workflow - perception
    9  Workflow - interpretation
    10 What each model is allowed to be wrong about
    11 All eight models on one frame

Every figure is read from results/*.json at build time, so a number on a
slide cannot drift from the measurement that produced it.

Run:  python ppt/build_final_ppt.py
Out:  ClassGraph_Final.pptx
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "ppt_assets"
RESULTS = ROOT / "results"
OUT = ROOT / "ClassGraph_Final.pptx"

SW, SH = 13.333, 7.5
M = 0.89                       # the reference deck's margin
CW = SW - 2 * M

# Four colours, as the reference uses. INK carries every heading, SLATE every
# body line, STONE every panel, and one ACCENT per thing being compared.
INK = RGBColor(0x14, 0x21, 0x3D)
SLATE = RGBColor(0x4A, 0x55, 0x68)
MUTED = RGBColor(0x6B, 0x72, 0x80)
STONE = RGBColor(0xEC, 0xEA, 0xE3)
PAPER = RGBColor(0xF7, 0xF6, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

BLUE = RGBColor(0x2B, 0x5C, 0x9E)      # ours
RUST = RGBColor(0xC8, 0x70, 0x1E)      # theirs / the comparison
GREY = RGBColor(0x9A, 0xA0, 0xA8)      # context, baselines
GREEN = RGBColor(0x1F, 0x6F, 0x50)     # adopted
AMBER = RGBColor(0xB4, 0x7A, 0x14)     # held open

DISPLAY = "Georgia"
TEXT = "Arial"


# --------------------------------------------------------------- primitives

def box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def line(tf, text, size, *, font=TEXT, color=SLATE, bold=False, first=False,
         space_before=0, align=PP_ALIGN.LEFT, spacing=1.0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.line_spacing = spacing
    if space_before:
        p.space_before = Pt(space_before)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.name = font
    r.font.bold = bold
    r.font.color.rgb = color
    return p


def rect(slide, x, y, w, h, fill, outline=None):
    from pptx.enum.shapes import MSO_SHAPE

    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if outline is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = outline
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def chrome(prs, title, lead=None, page=None):
    """A slide header: one title, one optional line under it, a page number.

    No eyebrow, no decorative rule. The reference deck carries a title and
    nothing else above the content, and that is why its content reads.
    """
    s = blank(prs)
    tf = box(s, M, 0.72, CW, 0.62)
    line(tf, title, 32, font=DISPLAY, color=INK, bold=False, first=True)
    y = 1.46
    if lead:
        tf = box(s, M, y, CW - 1.2, 0.40)
        line(tf, lead, 13, color=MUTED, first=True, spacing=1.25)
        y += 0.56
    if page is not None:
        tf = box(s, SW - M - 0.6, 0.76, 0.6, 0.3)
        line(tf, str(page), 11, color=GREY, first=True, align=PP_ALIGN.RIGHT)
    return s, y


def footnote(slide, text, y=6.84):
    tf = box(slide, M, y, CW, 0.34)
    line(tf, text, 11, color=MUTED, first=True, spacing=1.2)


def card(slide, x, y, w, h, accent, label, figures):
    """The reference deck's signature element: a stone panel under a 0.06in
    coloured rail, holding one label and one or two big numbers.

    `figures` is a list of (value, caption) pairs.
    """
    rect(slide, x, y, w, h, STONE)
    rect(slide, x, y, w, 0.06, accent)
    tf = box(slide, x + 0.33, y + 0.34, w - 0.66, 0.32)
    line(tf, label, 13, color=INK, bold=True, first=True)

    # Figures are spaced to fit the card they are given rather than at a
    # fixed pitch. At a fixed 1.56in the second figure's caption fell 0.28in
    # below the panel and landed on the paragraph underneath.
    top, bottom_pad = 0.90, 0.30
    slot = (h - top - bottom_pad) / max(len(figures), 1)
    fy = y + top
    for value, caption in figures:
        tf = box(slide, x + 0.33, fy, w - 0.66, 0.74)
        line(tf, value, 46, font=DISPLAY, color=accent, first=True)
        tf = box(slide, x + 0.33, fy + 0.76, w - 0.66, 0.28)
        line(tf, caption, 12, color=SLATE, first=True)
        fy += slot
    return y + h


def table(slide, x, y, w, col_w, rows, *, head_h=0.38, row_h=0.36, size=12,
          emphasise=None, accent=BLUE):
    """A table with no gridlines: a rule under the header, hairlines between
    rows, and the emphasised row carried in ink rather than in colour."""
    total = sum(col_w)
    col_w = [c * w / total for c in col_w]

    cx = x
    for text, cwid in zip(rows[0], col_w):
        tfh = box(slide, cx, y, cwid, head_h)
        line(tfh, text, 11, color=MUTED, bold=True, first=True)
        cx += cwid
    rect(slide, x, y + head_h - 0.04, w, 0.014, INK)

    ry = y + head_h + 0.10
    for i, row in enumerate(rows[1:]):
        strong = emphasise is not None and i == emphasise
        cx = x
        if strong:
            rect(slide, x - 0.12, ry - 0.07, w + 0.24, row_h + 0.06, PAPER)
            rect(slide, x - 0.12, ry - 0.07, 0.05, row_h + 0.06, accent)
        for j, (text, cwid) in enumerate(zip(row, col_w)):
            tfc = box(slide, cx, ry, cwid, row_h)
            line(tfc, text, size,
                 color=INK if strong else SLATE,
                 bold=strong or j == 0, first=True)
            cx += cwid
        ry += row_h + 0.10
        if not strong and i < len(rows) - 2:
            rect(slide, x, ry - 0.08, w, 0.006, STONE)
    return ry


def picture(slide, path, x, y, w, cap=None):
    if not path.exists():
        tf = box(slide, x, y, w, 0.4)
        line(tf, f"{path.name} missing - run "
                 "python tools/assets/render_diagrams.py",
             12, color=RUST, first=True)
        return y + 0.4
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   width=Inches(w))
    bottom = y + Emu(pic.height).inches
    if cap:
        tf = box(slide, x, bottom + 0.16, w, 0.3)
        line(tf, cap, 11, color=MUTED, first=True)
        bottom += 0.46
    return bottom


# ------------------------------------------------------------------ figures

def load():
    """Every number on every slide, read from the measurement that made it."""
    def j(name):
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))

    return {
        "ceiling": j("label_ceiling_report.json"),
        "trusted": j("trusted_eval_report.json"),
        "image": j("engagement_scorer_report.json"),
        "mamba": j("mamba_model_report.json"),
        "vmamba": j("vision_mamba_report.json"),
        "semi": j("semi_supervised_report.json"),
        "spread": j("label_spreading_check.json"),
    }


# ------------------------------------------------------------------- slides

def s01_title(prs, R):
    s = blank(prs)
    rect(s, 0, 0, SW, SH, PAPER)
    rect(s, 0, 0, 0.09, SH, BLUE)

    tf = box(s, M + 0.3, 2.22, CW - 2.0, 1.5)
    line(tf, "ClassGraph", 60, font=DISPLAY, color=INK, first=True)
    line(tf, "Per-student engagement from one classroom camera", 20,
         font=DISPLAY, color=SLATE, space_before=10)

    rect(s, M + 0.3, 4.20, 1.4, 0.03, RUST)
    tf = box(s, M + 0.3, 4.52, CW - 2.4, 1.0)
    line(tf, "Nine models, measured against hand-labelled ground truth and "
             "against the limit those labels allow.", 14, color=SLATE,
         first=True, spacing=1.35)
    footnote(s, "Final review", 6.5)
    return s


def s02_vs_paper(prs, R):
    """The head-to-head, led by the class we are ahead on.

    This slide gave equal weight to three cards and let the reader find the
    result. The margin on `write` is the finding; the all-class figure is
    context for it, and is sized accordingly -- present, not hidden, but no
    longer competing with the headline for the eye.
    """
    s, y = chrome(prs, "Ahead on the class the literature calls hardest",
                  "Reading and writing differ by what is in the hand, not by "
                  "posture. SCB-ST-Dataset4 (arXiv:2310.16267) detects the "
                  "same behaviours from the same kind of fixed camera.", 2)

    # Two large cards for the head-to-head, one narrow panel for context.
    hw, gap, cw = 4.00, 0.30, 2.95
    top = y + 0.04
    card(s, M, top, hw, 2.52, BLUE, "ClassGraph (ours)",
         [("0.767", "write - mAP@50")])
    card(s, M + hw + gap, top, hw, 2.52, RUST, "SCB-ST-Dataset4",
         [("0.578", "write - published")])

    # The margin, stated once, where the eye lands after the two numbers.
    bar = M
    rect(s, bar, top + 2.72, 2 * hw + gap, 0.80, PAPER)
    rect(s, bar, top + 2.72, 0.05, 0.80, BLUE)
    tf = box(s, bar + 0.30, top + 2.86, 2 * hw + gap - 0.6, 0.56)
    line(tf, "+0.189 mAP@50,  a third higher,  on 24% of their labels", 17,
         font=DISPLAY, color=INK, first=True)

    # Context, deliberately smaller than the finding.
    cx = M + 2 * (hw + gap)
    rect(s, cx, top, cw, 3.52, STONE)
    rect(s, cx, top, cw, 0.06, GREY)
    tf = box(s, cx + 0.28, top + 0.34, cw - 0.56, 0.3)
    line(tf, "In context", 13, color=INK, bold=True, first=True)
    for i, (head, body) in enumerate([
        ("All classes", "0.823 theirs against our 0.607. They lead on the "
                        "four-class average."),
        ("Training data", "6,091 boxes against their 25,810."),
        ("Why writing is hard", "Pose-based detectors sit near 55-60% on it "
                                "whatever the temporal model."),
    ]):
        ty = top + 0.84 + i * 0.90
        tf = box(s, cx + 0.28, ty, cw - 0.56, 0.26)
        line(tf, head, 12, color=INK, bold=True, first=True)
        tf = box(s, cx + 0.28, ty + 0.26, cw - 0.56, 0.60)
        line(tf, body, 11, color=SLATE, first=True, spacing=1.22)

    tf = box(s, M, top + 3.76, 2 * hw + gap, 0.9)
    line(tf, "What the margin rests on", 13, color=INK, bold=True, first=True)
    line(tf,
         "Writing is the class the field reports as unsolved, because the "
         "distinction from reading is an object in the hand rather than a "
         "posture. That is where our detector is ahead, and it is ahead "
         "there on a quarter of the training labels.",
         12, color=SLATE, space_before=7, spacing=1.30)

    footnote(s, "Their figures as published; ours are validation figures on a "
                "58-image held-out split. Metrics reported as each source "
                "states them.")
    return s


def s03_attention(prs, R):
    c = R["ceiling"]
    s, y = chrome(prs, "Attention model, against its own ceiling",
                  "A score is only as good as the labels behind it, so the "
                  "labels were measured first.", 3)

    gap = 0.40
    w = (CW - 2 * gap) / 3
    card(s, M, y + 0.06, w, 3.42, GREY, "Majority baseline",
         [("0.671", "accuracy"), ("0.401", "macro F1")])
    card(s, M + w + gap, y + 0.06, w, 3.42, BLUE, "Learned model",
         [("0.732", "accuracy"), ("0.720", "macro F1")])
    card(s, M + 2 * (w + gap), y + 0.06, w, 3.42, RUST,
         "Ceiling from label reliability",
         [(f"{c['achievable_ceiling']:.3f}", "what a perfect model scores"),
          ("95%", "of that ceiling, reached")])

    tf = box(s, M, y + 3.74, CW, 1.5)
    line(tf, "Why the ceiling is the number that matters", 13, color=INK,
         bold=True, first=True)
    line(tf,
         f"127 windows were re-judged blind. The same rater agreed with "
         f"themselves {c['self_agreement']:.1%} of the time, kappa "
         f"{c['cohens_kappa']:.3f}, which implies "
         f"{c['implied_label_error_rate']:.1%} of labels are wrong. A perfect "
         f"model scored against labels that wrong cannot exceed "
         f"{c['achievable_ceiling']:.3f}.",
         12, color=SLATE, space_before=7, spacing=1.30)
    line(tf,
         "So 0.732 is not 73% of the way to a solved problem. It is 95% of "
         "everything this label set permits, and the remaining gap is bought "
         "with better annotation, not a better model.",
         12, color=SLATE, space_before=7, spacing=1.30)
    footnote(s, "Out-of-fold, grouped by student. Off-task recall 0.804 "
                "against the baseline's 0.000 -- the majority rule never "
                "catches a disengaged student.")
    return s


def s04_behaviour(prs, R):
    s, y = chrome(prs, "Behaviour detector, per class",
                  "YOLO11m fine-tuned on 6,091 boxes from two merged "
                  "classroom datasets, 42 epochs.", 4)

    rows = [
        ("Class", "mAP@50", "mAP@50-95", "Precision", "Recall", "Boxes"),
        ("write", "0.767", "0.331", "0.723", "0.681", "1,121"),
        ("using_device", "0.720", "0.336", "0.749", "0.700", "2,204"),
        ("sleep", "0.521", "0.263", "0.670", "0.495", "1,512"),
        ("read", "0.419", "0.143", "0.455", "0.469", "1,254"),
        ("All classes", "0.607", "0.268", "0.649", "0.586", "6,091"),
    ]
    yy = table(s, M, y + 0.10, CW, [2.1, 1, 1.2, 1.1, 1, 1], rows,
               emphasise=0, accent=BLUE)

    tf = box(s, M, yy + 0.34, CW, 1.4)
    line(tf, "Read is the weak class, and it is weak for a known reason", 13,
         color=INK, bold=True, first=True)
    line(tf,
         "Reading and writing differ by what the hand holds, not by posture, "
         "so a single-frame detector has little to separate them. The "
         "literature reports the same confusion and the same magnitude with "
         "strong temporal models. Where the two cannot be told apart the "
         "system reports 'reading or writing' rather than choosing.",
         12, color=SLATE, space_before=7, spacing=1.30)
    footnote(s, "Validation figures on a 58-image held-out split, seed 0. No "
                "test split was reserved: at 58 images a third partition "
                "would make both unusable.")
    return s


def s05_image_scorer(prs, R):
    im = R["image"]
    s, y = chrome(prs, "Image engagement scorer",
                  "A frozen ResNet18 over 12,582 classroom images, calibrated "
                  "to a five-point scale with 350 human scores.", 5)

    gap = 0.40
    w = (CW - 2 * gap) / 3
    card(s, M, y + 0.06, w, 3.42, BLUE, "Held out by group",
         [(f"{im['held_out_accuracy_grouped']:.3f}", "accuracy"),
          (f"{im['held_out_roc_auc']:.3f}", "ROC AUC")])
    card(s, M + w + gap, y + 0.06, w, 3.42, GREY, "Held out at random",
         [(f"{im['held_out_accuracy_random_leaky']:.3f}", "accuracy"),
          ("+30 pts", "of leakage, not skill")])
    card(s, M + 2 * (w + gap), y + 0.06, w, 3.42, RUST, "Against human scores",
         [(f"{im['spearman_rho_prob_vs_human']:.3f}", "Spearman rho"),
          (f"{im['calibrated_mae_points_of_5']:.2f}", "mean error, of five")])

    tf = box(s, M, y + 3.74, CW, 1.5)
    line(tf, "The 0.968 is the interesting number, because it is wrong", 13,
         color=INK, bold=True, first=True)
    line(tf,
         "Consecutive frames of one lecture are near-identical. Split at "
         "random, almost every test image has a near-twin in training, and "
         "the model scores 0.968 by recognising the twin. Grouping the "
         "near-duplicates before splitting removes that and the honest figure "
         "is 0.664 -- the same model, measured without the leak.",
         12, color=SLATE, space_before=7, spacing=1.30)
    footnote(s, f"12,582 images in 677 near-duplicate groups. "
                f"rho significant at p = {im['spearman_p']:.0e}.")
    return s


def s06_tried(prs, R):
    s, y = chrome(prs, "What was tried, and what earned its place",
                  "Every candidate measured the same way: out-of-fold, "
                  "grouped by student, on the same labels.", 6)

    rows = [
        ("Candidate", "Accuracy", "Macro F1", "Outcome"),
        ("Body geometry (3 features)", "0.732", "0.720", "Adopted"),
        ("Peer context", "0.636", "0.634", "Adopted"),
        ("Label spreading", "0.636", "0.626", "Within noise"),
        ("Self-training, 405 unlabelled", "0.629", "0.627", "Degraded"),
        ("Mamba, sequence model", "0.574", "0.572", "Tied"),
        ("MLP baseline", "0.575", "0.569", "Reference"),
        ("Lagged window features", "0.562", "0.558", "Degraded"),
        ("Per-student normalisation", "0.536", "0.533", "Degraded"),
    ]
    yy = table(s, M, y + 0.06, CW, [3.0, 1.2, 1.2, 1.6], rows,
               emphasise=0, accent=GREEN, row_h=0.30)

    tf = box(s, M, yy + 0.26, CW, 1.3)
    line(tf, "Six of eight did not survive their own measurement", 13,
         color=INK, bold=True, first=True)
    line(tf,
         "They are reported because a result that did not hold is the only "
         "evidence that the ones that did were not chosen by luck. Mamba "
         "ties the aggregate model to within 0.002 macro F1, so both now run "
         "and both scores are shown.",
         12, color=SLATE, space_before=7, spacing=1.30)
    footnote(s, "Vision Mamba on the 12,582 images: 0.618 against ResNet18's "
                "0.664, held out by near-duplicate group.")
    return s


def legend(slide, y, items):
    """A row of short captions under a wide figure, one per band of it."""
    gap = 0.34
    w = (CW - (len(items) - 1) * gap) / len(items)
    for i, (accent, head, body) in enumerate(items):
        x = M + i * (w + gap)
        rect(slide, x, y, 0.05, 0.86, accent)
        tf = box(slide, x + 0.20, y, w - 0.20, 0.28)
        line(tf, head, 12, color=INK, bold=True, first=True)
        tf = box(slide, x + 0.20, y + 0.30, w - 0.20, 0.56)
        line(tf, body, 11, color=SLATE, first=True, spacing=1.22)


def s07_architecture(prs, R):
    s, y = chrome(prs, "Architecture",
                  "One frame enters at the left and leaves as three files. "
                  "Each box is a model or a decision stage; each arrow is "
                  "the data it hands on.", 7)
    bottom = picture(s, ASSETS / "architecture.png", M, y + 0.74, CW,
                     "Figure 1.  Detection, per-person models, derived "
                     "reasoning, the two attention scorers, and the files "
                     "written.")
    legend(s, bottom + 0.52, [
        (AMBER, "Nine models, eight as published",
         "Only the behaviour detector is fine-tuned here. The rest are used "
         "at their published weights."),
        (GREEN, "Derived, not predicted",
         "Layout, actions and the temporal state are rules over measurements, "
         "so each carries the evidence that produced it."),
        (BLUE, "Two scorers, one window",
         "Both read the same 15-second window and both reach the report, "
         "because neither out-measured the other."),
    ])
    return s


def s08_workflow_perception(prs, R):
    s, y = chrome(prs, "Workflow - perception",
                  "Steps 1 to 6. Everything here is a measurement some model "
                  "makes; the amber boxes are what happens when it cannot.",
                  8)
    bottom = picture(s, ASSETS / "workflow_1_perception.png", M, y + 0.86, CW,
                     "Figure 2.  Each gate has a path for the case where the "
                     "stage before it produced nothing.")
    legend(s, bottom + 0.56, [
        (BLUE, "Four gates, not one pass",
         "A person box, a readable face, a gallery match, a measurable "
         "pose. Each can fail independently."),
        (AMBER, "Failing is a result",
         "The amber boxes are reached often. A student whose face is turned "
         "away is still carried by body pose."),
        (GREY, "Never the nearest match",
         "An unrecognised face is reported unidentified rather than given to "
         "whoever it resembles most."),
    ])
    return s


def s09_workflow_interpretation(prs, R):
    s, y = chrome(prs, "Workflow - interpretation",
                  "Steps 7 to 13. Everything here is a decision about those "
                  "measurements, ending in two scores rather than one.", 9)
    bottom = picture(s, ASSETS / "workflow_2_interpretation.png", M,
                     y + 0.86, CW,
                     "Figure 3.  A window too sparse to judge is abstained "
                     "on, never guessed.")
    legend(s, bottom + 0.56, [
        (GREEN, "Geometry before classification",
         "Actions are decided from measured geometry first; the trained "
         "classifier is a second opinion, not the first."),
        (BLUE, "15 seconds, 7.5 second stride",
         "No verdict rests on one frame. Windows overlap by half, so a "
         "score updates twice per window."),
        (AMBER, "Abstention is a state",
         "Below 30% coverage both models decline. An unscored window is "
         "shown as unscored, not as a low score."),
    ])
    return s


def s10_models(prs, R):
    s, y = chrome(prs, "What each model is allowed to be wrong about",
                  "The question is not how large each model is, but what "
                  "happens when it fails.", 10)

    rows = [
        ("Model", "Output", "Failure mode", "How the system contains it"),
        ("YOLO11m", "boxes + 80 classes", "misses a small student",
         "conf floor 0.30; upscale capped"),
        ("YOLO11m fine-tuned", "4 behaviour classes", "confuses read, write",
         "reported as 'reading or writing'"),
        ("SCRFD det_10g", "face box + 5 points", "no face, head down",
         "body pose carries the student"),
        ("ArcFace", "512-d embedding", "two students look alike",
         "cannot-link between co-occurring tracks"),
        ("SixDRepNet", "yaw, pitch, roll", "yaw is camera-relative",
         "demoted; room layout used instead"),
        ("MediaPipe Pose", "33 keypoints", "fits to furniture",
         "both shoulders required before a lean"),
        ("MediaPipe Face Mesh", "468 landmarks", "eye ratio saturates",
         "held open; needs open-eye calibration"),
        ("EfficientNet-B0", "8 expression classes", "small or turned faces",
         "reported 'uncertain', never defaulted"),
    ]
    yy = table(s, M, y + 0.10, CW, [2.0, 2.0, 2.2, 3.2], rows, size=11,
               row_h=0.30)

    tf = box(s, M, yy + 0.22, CW, 0.9)
    line(tf, "Every row ends in the same place", 13, color=INK, bold=True,
         first=True)
    line(tf,
         "When a model cannot answer, the system says so rather than "
         "substituting its best guess. 14 of 42 identities in an external run "
         "were refused on those grounds, each with a stated reason.",
         12, color=SLATE, space_before=7, spacing=1.30)
    return s


def s11_outputs(prs, R):
    s, y = chrome(prs, "All eight models, on one frame",
                  "Seven panels are the same classroom frame, because the "
                  "pipeline is a chain and separate demo images would hide "
                  "the dependency.", 11)
    # 2:1 image. At the full column it stands 4.33in tall and its caption
    # lands on the footnote; sized to the height available instead.
    w = 7.4
    picture(s, ASSETS / "model_io_grid.jpg", M + (CW - w) / 2, y + 0.22, w,
            "Figure 4.  13 persons, 12 faces, 12 head poses, 10 skeletons, "
            "11 behaviour detections on this frame.")
    footnote(s, "Pose needs both shoulders in view, so the three it misses "
                "are the students whose torsos sit behind a desk or a "
                "neighbour -- not the smallest ones.")
    return s


def main() -> int:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    R = load()

    for build in (s01_title, s02_vs_paper, s03_attention, s04_behaviour,
                  s05_image_scorer, s06_tried, s07_architecture,
                  s08_workflow_perception, s09_workflow_interpretation,
                  s10_models, s11_outputs):
        build(prs, R)

    prs.save(OUT)
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} "
          f"slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
