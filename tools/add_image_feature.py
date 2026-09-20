"""Feed the 12,582-image scorer into the window model as one more feature.

The two models have been unrelated: the image scorer learned engagement from
12,582 labelled portraits, while the attention score learned from 179 windows
of geometry. This connects them. Each frame of a window is scored by the image
model, the scores are summarised, and the summary joins the window's feature
vector -- so the large labelled set contributes to the attention score through
a feature rather than through a label.

Whether it helps is an open question and the reason this reports before and
after. The image model was trained on close-up portraits and is being given
small classroom crops taken from an elevated camera. That is a domain shift,
and the failure it produces is quiet: if the model cannot see enough to
discriminate, it returns nearly the same score for everyone and the new feature
is noise wearing the shape of information. The spread of its output on real
crops is printed for exactly that reason -- a near-zero spread means it should
not be used, whatever the accuracy does.

Crops are cut fresh rather than reused from labeling/images/, because those
carry a green box and dimmed surroundings drawn on for human labelling. Scoring
those would measure the annotation.

    python tools/add_image_feature.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLIPS = Path("C:/Users/prana/Downloads/high-2636(2191-2347)/high-2636(2191-2347)")
RAW = Path("outputs/final2/raw.jsonl")
LABELS = Path("labeling/labels.json")
DATA = Path("labeling/images/Final Dataset 256/Final Dataset 256")
OUT = Path("outputs/window_labels_with_image_feature.json")

SAMPLE = 10
FRAME_STRIDE = 3


def build_image_scorer():
    """Train the portrait model and return a callable scoring a BGR crop."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from torchvision import models, transforms

    from tools.train_engagement_scorer import list_images, load_or_build

    paths = list_images(DATA)
    labels = np.array([1 if "Engagement" in p.parts
                       and "Disengagement" not in p.parts else 0
                       for p in paths])
    features = load_or_build(paths, refresh=False)
    clf = LogisticRegression(max_iter=2000).fit(features, labels)
    print(f"  image scorer trained on {len(paths)} images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def score_batch(crops: list) -> list[float]:
        """Probability of engaged, per BGR crop."""
        from PIL import Image
        if not crops:
            return []
        batch = torch.stack([
            tf(Image.fromarray(c[:, :, ::-1])) for c in crops]).to(device)
        with torch.no_grad():
            embedded = net(batch).cpu().numpy()
        return clf.predict_proba(embedded)[:, 1].tolist()

    return score_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import cv2

    if not LABELS.exists():
        print(f"{LABELS} not found.")
        return 1
    if not CLIPS.is_dir():
        print(f"source clips not found at {CLIPS}")
        return 1

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    print(f"{len(labels)} labelled windows")
    score_batch = build_image_scorer()

    clips = sorted(CLIPS.glob("*.mp4"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    counts = []
    for c in clips:
        cap = cv2.VideoCapture(str(c))
        counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()

    boxes: dict[tuple[int, int], tuple] = {}
    times: dict[int, int] = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        fid = int(rec.get("frame_id", -1))
        times[fid] = int(rec.get("timestamp_ms") or 0)
        for person in rec.get("persons", []):
            pid = person.get("person_id")
            if pid and pid > 0 and person.get("bbox"):
                boxes[(fid, int(pid))] = tuple(int(v) for v in person["bbox"])

    def locate(frame_id: int):
        src = frame_id * SAMPLE
        for i, n in enumerate(counts):
            if src < n:
                return i, src
            src -= n
        return None, None

    caps: dict[int, object] = {}
    enriched: dict[str, dict] = {}
    all_scores: list[float] = []

    for n, (key, entry) in enumerate(labels.items(), 1):
        pid = entry["person_id"]
        fids = [f for f, p in boxes
                if p == pid and entry["start_ms"] <= times.get(f, -1)
                < entry["end_ms"]]
        crops = []
        for fid in sorted(fids)[::FRAME_STRIDE]:
            ci, local = locate(fid)
            if ci is None:
                continue
            cap = caps.get(ci)
            if cap is None:
                cap = caps[ci] = cv2.VideoCapture(str(clips[ci]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, local)
            ok, frame = cap.read()
            if not ok:
                continue
            x, y, w, h = boxes[(fid, pid)]
            pad = int(max(w, h) * 0.22)
            H, W = frame.shape[:2]
            crop = frame[max(0, y - pad):min(H, y + h + pad),
                         max(0, x - pad):min(W, x + w + pad)]
            if crop.size:
                crops.append(crop)

        scores = score_batch(crops)
        mean = statistics.fmean(scores) if scores else 0.5
        all_scores.extend(scores)
        enriched[key] = {**entry,
                         "image_engagement_mean": round(float(mean), 4),
                         "image_frames_scored": len(scores)}
        if n % 40 == 0:
            print(f"    {n}/{len(labels)} windows")

    for cap in caps.values():
        cap.release()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(enriched, indent=1), encoding="utf-8")

    means = [v["image_engagement_mean"] for v in enriched.values()]
    print(f"\n  {len(all_scores)} crops scored across {len(enriched)} windows")
    print(f"  per-window mean score: min {min(means):.3f}  "
          f"median {statistics.median(means):.3f}  max {max(means):.3f}")
    spread = max(means) - min(means)
    print(f"  spread: {spread:.3f}  (stdev {statistics.pstdev(means):.3f})")

    on = [v["image_engagement_mean"] for v in enriched.values()
          if v["label"] == "on"]
    off = [v["image_engagement_mean"] for v in enriched.values()
           if v["label"] == "off"]
    if on and off:
        gap = statistics.fmean(on) - statistics.fmean(off)
        print(f"\n  on-task windows  : {statistics.fmean(on):.3f}")
        print(f"  off-task windows : {statistics.fmean(off):.3f}")
        print(f"  separation       : {gap:+.3f}")
        if abs(gap) < 0.02:
            print("\n  The portrait model does not separate these crops. Adding"
                  " it as a\n  feature would add noise, not information.")
        else:
            print("\n  It separates them. Worth adding as a feature.")
    print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
