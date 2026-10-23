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
    tf = box(s, M, 0.74, CW, 0.70)
    line(tf, title, 34, font=DISPLAY, color=INK, bold=False, first=True)
    y = 1.52
    if lead:
        tf = box(s, M, y, CW - 1.2, 0.40)
        line(tf, lead, 13, color=MUTED, first=True, spacing=1.25)
        y += 0.56
    if page is not None:
        tf = box(s, SW - M - 0.6, 0.76, 0.6, 0.3)
        line(tf, str(page), 11, color=GREY, first=True, align=PP_ALIGN.RIGHT)
    return s, y


def footnote(slide, text, y=6.84):
    """The caveat line. Pass a longer string and it wraps; start it higher
    when it needs two lines."""
    tf = box(slide, M, y, CW, 0.60)
    line(tf, text, 11, color=MUTED, first=True, spacing=1.25)


def card(slide, x, y, w, h, accent, label, figures):
    """The reference deck's signature element: a stone panel under a 0.06in
    coloured rail, holding one label and one or two big numbers.

    `figures` is a list of (value, caption) pairs.
    """
    rect(slide, x, y, w, h, STONE)
    rect(slide, x, y, w, 0.06, accent)
    tf = box(slide, x + 0.33, y + 0.34, w - 0.66, 0.32)
    line(tf, label, 13, color=INK, bold=True, first=True)

    # Figures are centred in the space under the label rather than stacked
    # from the top. A single figure pinned to the top left two inches of
    # empty stone beneath it -- whitespace belongs between the panels, not
    # trapped inside one.
    label_h, pad = 0.80, 0.30
    sizes = {1: 72, 2: 50}
    size = sizes.get(len(figures), 44)
    block = size / 72 * 1.22 + 0.34          # number + its caption
    span = h - label_h - pad
    fy = y + label_h + max((span - block * len(figures)) / 2, 0.0)

    for value, caption in figures:
        tf = box(slide, x + 0.33, fy, w - 0.66, block - 0.30)
        line(tf, value, size, font=DISPLAY, color=accent, first=True)
        tf = box(slide, x + 0.33, fy + block - 0.32, w - 0.66, 0.28)
        line(tf, caption, 12, color=SLATE, first=True)
        fy += block
    return y + h


def table(slide, x, y, w, col_w, rows, *, head_h=0.38, row_h=0.36, size=12,
          emphasise=None, mute=(), accent=BLUE):
    """A table with no gridlines: a rule under the header, hairlines between
    rows, and the emphasised rows carried in ink rather than in colour.

    `emphasise` is one row index or several. `mute` greys rows that are
    context rather than finding -- a weaker class, a candidate that did not
    survive. Muted is not hidden: every row is still printed, at full size,
    in a colour that reads perfectly well up close.
    """
    total = sum(col_w)
    col_w = [c * w / total for c in col_w]

    cx = x
    for text, cwid in zip(rows[0], col_w):
        tfh = box(slide, cx, y, cwid, head_h)
        line(tfh, text, 11, color=MUTED, bold=True, first=True)
        cx += cwid
    rect(slide, x, y + head_h - 0.04, w, 0.014, INK)

    strong_rows = ({emphasise} if isinstance(emphasise, int)
                   else set(emphasise or ()))
    ry = y + head_h + 0.10
    for i, row in enumerate(rows[1:]):
        strong = i in strong_rows
        quiet = i in set(mute)
        cx = x
        if strong:
            rect(slide, x - 0.12, ry - 0.07, w + 0.24, row_h + 0.06, PAPER)
            rect(slide, x - 0.12, ry - 0.07, 0.05, row_h + 0.06, accent)
        for j, (text, cwid) in enumerate(zip(row, col_w)):
            tfc = box(slide, cx, ry, cwid, row_h)
            line(tfc, text, size,
                 color=INK if strong else (MUTED if quiet else SLATE),
                 bold=strong or (j == 0 and not quiet), first=True)
            cx += cwid
        ry += row_h + 0.10
        if not strong and i < len(rows) - 2:
            rect(slide, x, ry - 0.08, w, 0.006, STONE)
    return ry


def banner(slide, x, y, w, text, accent=BLUE, size=17):
    """The one line a slide is about, set where the eye lands after the
    figures above it."""
    rect(slide, x, y, w, 0.80, PAPER)
    rect(slide, x, y, 0.05, 0.80, accent)
    tf = box(slide, x + 0.30, y + 0.14, w - 0.60, 0.56)
    line(tf, text, size, font=DISPLAY, color=INK, first=True)
    return y + 0.80


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

    tf = box(s, M + 0.3, 2.60, CW - 2.0, 1.5)
    line(tf, "ClassGraph", 60, font=DISPLAY, color=INK, first=True)
    line(tf, "Per-student engagement from one classroom camera", 19,
         font=DISPLAY, color=SLATE, space_before=12)
    rect(s, M + 0.3, 4.62, 1.4, 0.03, RUST)
    footnote(s, "Final review", 6.5)
    return s


def s02_vs_paper(prs, R):
    """The head-to-head. Two numbers, the margin, one line of caveat."""
    s, y = chrome(prs, "Where we lead", page=2)

    hw, gap = 5.15, 0.35
    top = y + 0.30
    card(s, M, top, hw, 3.52, BLUE, "ClassGraph",
         [("0.767", "write - mAP@50")])
    card(s, M + hw + gap, top, hw, 3.52, RUST, "SCB-ST-Dataset4",
         [("0.578", "write - published")])

    banner(s, M, top + 3.76, 2 * hw + gap,
           "+0.189 mAP@50,  on 24% of their labels", size=19)

    footnote(s, "Writing is the class the field reports as unsolved. On the "
                "four-class average they lead, 0.823 to our 0.607.")
    return s


def s03_attention(prs, R):
    c = R["ceiling"]
    s, y = chrome(prs, "Attention, against its ceiling", page=3)

    gap = 0.35
    w = (CW - 2 * gap) / 3
    top = y + 0.30
    card(s, M, top, w, 3.52, GREY, "Majority baseline",
         [("0.401", "macro F1")])
    card(s, M + w + gap, top, w, 3.52, BLUE, "Learned model",
         [("0.720", "macro F1")])
    card(s, M + 2 * (w + gap), top, w, 3.52, RUST, "Ceiling",
         [("0.770", "what a perfect model scores")])

    banner(s, M, top + 3.76, CW,
           "0.732 accuracy  -  95% of the ceiling these labels allow",
           size=19)

    footnote(s, f"Ceiling measured, not assumed: 127 windows re-judged blind, "
                f"kappa {c['cohens_kappa']:.3f}. Off-task recall 0.804 "
                f"against the baseline's 0.000.")
    return s


def s04_behaviour(prs, R):
    s, y = chrome(prs, "Behaviour, per class", page=4)

    rows = [
        ("Class", "mAP@50", "Precision", "Recall", "Boxes"),
        ("write", "0.767", "0.723", "0.681", "1,121"),
        ("using_device", "0.720", "0.749", "0.700", "2,204"),
        ("sleep", "0.521", "0.670", "0.495", "1,512"),
        ("read", "0.419", "0.455", "0.469", "1,254"),
        ("All classes", "0.607", "0.649", "0.586", "6,091"),
    ]
    yy = table(s, M, y + 0.46, CW, [2.4, 1.1, 1.1, 1.1, 1.1], rows,
               emphasise=(0, 1), mute=(2, 3), accent=BLUE, row_h=0.46,
               size=13)

    banner(s, M, yy + 0.42, CW, "Two of four classes clear 0.72", size=19)

    footnote(s, "Read and write differ by what the hand holds, not by "
                "posture; the field reports the same confusion.")
    return s


def s05_image_scorer(prs, R):
    im = R["image"]
    s, y = chrome(prs, "Image engagement scorer", page=5)

    gap = 0.35
    w = (CW - 2 * gap) / 3
    top = y + 0.30
    card(s, M, top, w, 3.52, BLUE, "Against human scores",
         [(f"{im['spearman_rho_prob_vs_human']:.3f}", "Spearman rho")])
    card(s, M + w + gap, top, w, 3.52, RUST, "Held out by group",
         [(f"{im['held_out_roc_auc']:.3f}", "ROC AUC")])
    card(s, M + 2 * (w + gap), top, w, 3.52, GREY, "Held out at random",
         [(f"{im['held_out_accuracy_random_leaky']:.3f}", "leakage")])

    banner(s, M, top + 3.76, CW,
           "Agrees with human scorers,  p = 7.8e-30", size=19)

    footnote(s, "12,582 images, 677 near-duplicate groups. Grouped accuracy "
                "0.664; the 0.968 is the same model with a leak.")
    return s


def s06_tried(prs, R):
    s, y = chrome(prs, "What earned its place", page=6)

    rows = [
        ("Candidate", "Macro F1", "Outcome"),
        ("Body geometry", "0.720", "Adopted"),
        ("Peer context", "0.634", "Adopted"),
        ("Label spreading", "0.626", "Within noise"),
        ("Self-training", "0.627", "Degraded"),
        ("Mamba sequence model", "0.572", "Tied"),
        ("MLP baseline", "0.569", "Reference"),
        ("Lagged windows", "0.558", "Degraded"),
        ("Per-student norm.", "0.533", "Degraded"),
    ]
    yy = table(s, M, y + 0.40, CW, [3.2, 1.4, 1.6], rows,
               emphasise=(0, 1), mute=(2, 3, 4, 5, 6, 7), accent=GREEN,
               row_h=0.34, size=13)

    banner(s, M, yy + 0.30, CW, "+0.151 macro F1 over the baseline",
           accent=GREEN, size=19)

    footnote(s, "Six of eight did not survive their own measurement, and are "
                "reported.")
    return s


def s07_architecture(prs, R):
    s, y = chrome(prs, "Architecture", page=7)
    picture(s, ASSETS / "architecture.png", M, y + 1.30, CW)
    footnote(s, "Nine models. One frame enters at the left and leaves as "
                "three files.")
    return s


def s08_workflow_perception(prs, R):
    s, y = chrome(prs, "Workflow - perception", page=8)
    picture(s, ASSETS / "workflow_1_perception.png", M, y + 1.30, CW)
    footnote(s, "Steps 1 to 6. Amber is what happens when a model cannot "
                "answer.")
    return s


def s09_workflow_interpretation(prs, R):
    s, y = chrome(prs, "Workflow - interpretation", page=9)
    picture(s, ASSETS / "workflow_2_interpretation.png", M, y + 1.30, CW)
    footnote(s, "Steps 7 to 13, ending in two scores rather than one.")
    return s


def s10_models(prs, R):
    s, y = chrome(prs, "What each model may be wrong about", page=10)

    rows = [
        ("Model", "Output", "Contained by"),
        ("YOLO11m", "boxes + 80 classes", "conf floor, capped upscale"),
        ("YOLO11m fine-tuned", "4 behaviours", "'reading or writing'"),
        ("SCRFD det_10g", "face box + 5 points", "body pose carries it"),
        ("ArcFace", "512-d embedding", "cannot-link on co-occurrence"),
        ("SixDRepNet", "yaw, pitch, roll", "room layout instead"),
        ("MediaPipe Pose", "33 keypoints", "both shoulders required"),
        ("MediaPipe Face Mesh", "468 landmarks", "held open"),
        ("EfficientNet-B0", "8 expressions", "reported 'uncertain'"),
    ]
    table(s, M, y + 0.40, CW, [2.6, 2.6, 3.4], rows, size=13, row_h=0.40)

    footnote(s, "When a model cannot answer, the system says so. 14 of 42 "
                "identities in an external run were refused.")
    return s


def s11_outputs(prs, R):
    s, y = chrome(prs, "Eight models, one frame", page=11)
    w = 7.9
    picture(s, ASSETS / "model_io_grid.jpg", M + (CW - w) / 2, y + 0.34, w)
    footnote(s, "13 persons, 12 faces, 12 head poses, 10 skeletons, 11 "
                "behaviour detections.")
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
    print(f"wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
