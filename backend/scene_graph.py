"""Stage 3: Scene Graph generator.

Consumes Stage 1+2 JSONL and produces a frame-by-frame scene graph matching
`graph_schema.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from itertools import combinations
from pathlib import Path

from backend.config import CONFIG, Config
from backend.engagement import classify_engagement
from backend.peer_interaction import _bbox_center, _within_conversational_distance, classify_pair_frame

logger = logging.getLogger(__name__)


def _phone_overlaps(person_bbox: list[float], objects: list[dict], cfg: Config) -> bool:
    """Whether a phone-class object overlaps this student's box at all."""
    if not cfg.engagement.use_object_fallback:
        return False
    px, py, pw, ph = person_bbox
    for obj in objects:
        if obj.get("cls") not in cfg.engagement.fallback_off_task_objects:
            continue
        ox, oy, ow, oh = obj["bbox"]
        if (
            max(0.0, min(px + pw, ox + ow) - max(px, ox)) > 0
            and max(0.0, min(py + ph, oy + oh) - max(py, oy)) > 0
        ):
            return True
    return False


def generate_scene_graph(record: dict, config: Config | None = None) -> dict:
    """Generate a single frame's scene graph from a Stage 1+2 record.

    Args:
        record: Stage 1+2 JSON record.
        config: Configuration instance. Defaults to CONFIG.

    Returns:
        A dict matching graph_schema.json.
    """
    cfg = config if config is not None else CONFIG
    frame_id = record["frame_id"]
    timestamp_ms = record["timestamp_ms"]
    persons = record.get("persons", [])
    objects = record.get("objects", [])

    nodes = []
    node_by_id = {}
    person_idx_map = {}  # maps assigned node id -> person dict

    for idx, person in enumerate(persons):
        # Determine unique node ID
        person_id = person.get("person_id")
        track_id = person.get("track_id")
        if person_id is not None:
            node_id = person_id
        elif track_id is not None:
            node_id = track_id
        else:
            node_id = -100 - idx

        # Determine node role
        if person_id is not None and person_id in cfg.profile.instructor_ids:
            role = "instructor"
        elif track_id is not None and track_id in cfg.profile.instructor_ids:
            role = "instructor"
        else:
            role = "student"

        # Compute engagement
        gaze_label = person.get("head_pose", {}).get("gaze_label") if person.get("head_pose") else None
        behaviour_label = person.get("behaviour", {}).get("label") if person.get("behaviour") else None
        face = person.get("face")
        eyes_closed = None
        if face and face.get("ear") is not None:
            eyes_closed = face["ear"] < cfg.face.ear_closed_threshold
        phone_nearby = _phone_overlaps(person["bbox"], objects, cfg)

        engagement = classify_engagement(
            gaze_label,
            behaviour_label,
            cfg.engagement,
            phone_nearby=phone_nearby,
            eyes_closed=eyes_closed,
        )

        node = {
            "id": node_id,
            "person_id": person_id,
            "role": role,
            "features": {
                "bbox": person.get("bbox"),
                "gaze_label": gaze_label,
                "posture": person.get("posture"),
                "expression": person.get("expression", {}).get("label") if person.get("expression") else None,
                "behaviour": behaviour_label,
                "engagement": engagement,
                "eyes_closed": eyes_closed,
                "rolling_engagement_pct": None,
                "is_sustained_distracted": None,
                "is_eyes_closed_sustained": None,
            }
        }
        nodes.append(node)
        node_by_id[node_id] = node
        person_idx_map[node_id] = person

    edges = []

    # Check relationships between every unique pair of nodes
    for node_a, node_b in combinations(nodes, 2):
        id_a, id_b = node_a["id"], node_b["id"]
        source, target = min(id_a, id_b), max(id_a, id_b)
        person_a, person_b = person_idx_map[id_a], person_idx_map[id_b]

        bbox_a = person_a["bbox"]
        bbox_b = person_b["bbox"]
        ca = _bbox_center(bbox_a)
        cb = _bbox_center(bbox_b)
        distance = math.hypot(ca[0] - cb[0], ca[1] - cb[1])

        # 1. Spatial Adjacency Edge
        wa = bbox_a[2]
        wb = bbox_b[2]
        gap = max(0.0, distance - (wa + wb) / 2.0)
        adjacency_threshold = min(wa, wb) * cfg.scene_graph.adjacency_gap_ratio
        if gap <= adjacency_threshold:
            edges.append({
                "type": "spatial_adjacency",
                "source": source,
                "target": target,
                "features": {
                    "distance_px": distance,
                    "oriented_fraction": None,
                    "shared_object_class": None,
                    "is_sustained_interaction": None,
                    "rolling_interaction_fraction": None,
                }
            })

        # 2. Mutual Orientation Edge
        if classify_pair_frame(person_a, person_b, cfg.peer_interaction):
            edges.append({
                "type": "mutual_orientation",
                "source": source,
                "target": target,
                "features": {
                    "distance_px": distance,
                    "oriented_fraction": None,
                    "shared_object_class": None,
                    "is_sustained_interaction": None,
                    "rolling_interaction_fraction": None,
                }
            })

        # 3. Shared Object Edges
        # Find if any book/phone/laptop lies perpendicular-wise between them
        for obj in objects:
            obj_bbox = obj.get("bbox")
            if not obj_bbox or not obj.get("cls"):
                continue
            co = _bbox_center(obj_bbox)

            # Project co onto the line segment ca -> cb
            vx, vy = cb[0] - ca[0], cb[1] - ca[1]
            v_len2 = vx**2 + vy**2
            if v_len2 == 0.0:
                continue

            wx, wy = co[0] - ca[0], co[1] - ca[1]
            dot = wx * vx + wy * vy
            t = dot / v_len2

            # Is the projection strictly between student centers?
            if 0.1 <= t <= 0.9:
                # Perpendicular projection point
                px = ca[0] + t * vx
                py = ca[1] + t * vy
                perp_dist = math.hypot(co[0] - px, co[1] - py)

                if perp_dist <= cfg.scene_graph.max_shared_object_distance_px:
                    edges.append({
                        "type": "shared_object",
                        "source": source,
                        "target": target,
                        "features": {
                            "distance_px": distance,
                            "oriented_fraction": None,
                            "shared_object_class": obj["cls"],
                            "is_sustained_interaction": None,
                            "rolling_interaction_fraction": None,
                        }
                    })

    return {
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "nodes": nodes,
        "edges": edges
    }


def process_jsonl(input_path: Path, output_path: Path, config: Config | None = None) -> int:
    """Read Stage 1+2 JSONL file and write Stage 3 Scene Graph JSONL file."""
    cfg = config if config is not None else CONFIG
    written = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            graph = generate_scene_graph(record, cfg)
            fout.write(json.dumps(graph) + "\n")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m backend.scene_graph",
        description="Convert Stage 1+2 JSONL into Stage 3 Scene Graph JSONL."
    )
    parser.add_argument("--jsonl", required=True, type=str, help="Path to input Stage 1+2 JSONL file.")
    parser.add_argument("--out", required=True, type=str, help="Path to output Stage 3 JSONL file.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    input_path = Path(args.jsonl)
    output_path = Path(args.out)

    if not input_path.is_file():
        logger.error("Input file not found: %s", input_path)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        count = process_jsonl(input_path, output_path)
        logger.info("Successfully processed %d frames and wrote to %s", count, output_path)
        return 0
    except Exception as exc:
        logger.exception("Failed to generate scene graph: %s", exc)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
