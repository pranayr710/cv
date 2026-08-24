"""Unit tests for Stage 3: Scene Graph generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.validators import Draft202012Validator

from backend.config import CONFIG, SceneGraphConfig
from backend.scene_graph import generate_scene_graph, process_jsonl


def _person(
    track_id: int | None,
    person_id: int | None = None,
    bbox=(0, 0, 50, 100),
    left_shoulder=None,
    right_shoulder=None,
    gaze_label=None,
    expression=None,
    behaviour=None,
    face_ear=None,
) -> dict:
    """Build a person dict matching the Stage 1+2 schema."""
    posture = None
    if left_shoulder is not None or right_shoulder is not None:
        posture = {
            "nose": None,
            "left_shoulder": list(left_shoulder) if left_shoulder else None,
            "right_shoulder": list(right_shoulder) if right_shoulder else None,
            "shoulder_mid": None,
            "hip_mid": None,
            "vertical_lean": None,
            "facing_direction": None,
        }
    face = None
    if face_ear is not None:
        face = {
            "bbox": [10, 10, 20, 20],
            "landmarks": None,
            "ear": face_ear,
        }
    return {
        "track_id": track_id,
        "person_id": person_id,
        "bbox": list(bbox),
        "confidence": 0.9,
        "face": face,
        "head_pose": {"gaze_label": gaze_label} if gaze_label is not None else None,
        "posture": posture,
        "source": "camera",
        "expression": {"label": expression, "score": 0.8} if expression is not None else None,
        "behaviour": {"label": behaviour, "score": 0.7, "reliability": "strong"} if behaviour is not None else None,
    }


@pytest.fixture
def graph_validator() -> Draft202012Validator:
    """Fixture returning a JSON Schema validator for graph_schema.json."""
    schema_path = Path(__file__).resolve().parent.parent / "graph_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def test_schema_compliance_empty_frame(graph_validator) -> None:
    record = {
        "frame_id": 0,
        "timestamp_ms": 0,
        "persons": [],
        "objects": []
    }
    graph = generate_scene_graph(record)
    assert graph["frame_id"] == 0
    assert graph["timestamp_ms"] == 0
    assert len(graph["nodes"]) == 0
    assert len(graph["edges"]) == 0
    graph_validator.validate(graph)


def test_node_role_and_features(graph_validator) -> None:
    # Student A is an ordinary student
    # Student B is declared as the instructor in CONFIG
    record = {
        "frame_id": 1,
        "timestamp_ms": 1000,
        "persons": [
            _person(track_id=1, person_id=1, bbox=(0, 0, 50, 100), gaze_label="teacher", expression="happy"),
            _person(track_id=2, person_id=2, bbox=(200, 0, 50, 100), gaze_label="student", behaviour="sleep"),
        ],
        "objects": []
    }
    # Temporary patch config to declare person_id=2 as instructor
    import copy
    custom_config = copy.deepcopy(CONFIG)
    # We have to patch profile because it's a frozen dataclass, so we re-create it
    from backend.config import ProfileConfig
    object.__setattr__(custom_config, "profile", ProfileConfig(instructor_ids=(2,)))

    graph = generate_scene_graph(record, custom_config)
    assert len(graph["nodes"]) == 2

    # Node 1: student
    node_1 = next(n for n in graph["nodes"] if n["id"] == 1)
    assert node_1["role"] == "student"
    assert node_1["features"]["gaze_label"] == "teacher"
    assert node_1["features"]["expression"] == "happy"
    assert node_1["features"]["engagement"] == "on"

    # Node 2: instructor
    node_2 = next(n for n in graph["nodes"] if n["id"] == 2)
    assert node_2["role"] == "instructor"
    assert node_2["features"]["behaviour"] == "sleep"
    assert node_2["features"]["engagement"] == "off"

    graph_validator.validate(graph)


def test_edges_adjacency_and_orientation(graph_validator) -> None:
    # A and B are close and oriented perpendicularly (facing each other/L-shape)
    # C is far away
    record = {
        "frame_id": 2,
        "timestamp_ms": 2000,
        "persons": [
            _person(track_id=1, person_id=1, bbox=(0, 0, 50, 100), left_shoulder=(25, 40), right_shoulder=(25, 60)),
            _person(track_id=2, person_id=2, bbox=(60, 0, 50, 100), left_shoulder=(85, 40), right_shoulder=(85, 60)),
            _person(track_id=3, person_id=3, bbox=(1000, 0, 50, 100), left_shoulder=(1025, 40), right_shoulder=(1025, 60)),
        ],
        "objects": []
    }
    graph = generate_scene_graph(record)
    graph_validator.validate(graph)

    # We expect 2 edges between 1 and 2: spatial_adjacency and mutual_orientation
    # No edges to 3
    edges_1_2 = [e for e in graph["edges"] if {e["source"], e["target"]} == {1, 2}]
    assert len(edges_1_2) == 2
    types = {e["type"] for e in edges_1_2}
    assert types == {"spatial_adjacency", "mutual_orientation"}

    edges_1_3 = [e for e in graph["edges"] if {e["source"], e["target"]} == {1, 3}]
    assert len(edges_1_3) == 0


def test_edges_shared_object(graph_validator) -> None:
    # A at (0, 0), B at (200, 0). Center distance is 200.
    # Object 1 (book) at (100, 10) -> projected t=0.5, perp_dist = 10 (lies between)
    # Object 2 (phone) at (50, 200) -> projected t=0.25, perp_dist = 200 (too far perp-wise)
    # Object 3 (book) at (250, 0) -> projected t=1.25 (not between)
    record = {
        "frame_id": 3,
        "timestamp_ms": 3000,
        "persons": [
            _person(track_id=1, person_id=1, bbox=(0, 0, 50, 100)),
            _person(track_id=2, person_id=2, bbox=(200, 0, 50, 100)),
        ],
        "objects": [
            {"cls": "book", "bbox": [75, -25, 50, 50]},       # center (100, 0), t=0.5, perp=0
            {"cls": "cell phone", "bbox": [25, 200, 50, 50]}, # center (50, 225), t=0.125, perp=175
            {"cls": "book", "bbox": [225, -25, 50, 50]},       # center (250, 0), t=1.25, perp=0
        ]
    }
    graph = generate_scene_graph(record)
    graph_validator.validate(graph)

    # Should find exactly 1 shared_object edge (Object 1: book)
    shared_edges = [e for e in graph["edges"] if e["type"] == "shared_object"]
    assert len(shared_edges) == 1
    assert shared_edges[0]["features"]["shared_object_class"] == "book"
    assert shared_edges[0]["source"] == 1
    assert shared_edges[0]["target"] == 2


def test_process_jsonl(tmp_path: Path, graph_validator) -> None:
    record = {
        "frame_id": 0,
        "timestamp_ms": 0,
        "persons": [
            _person(track_id=1, person_id=1, bbox=(0, 0, 50, 100))
        ],
        "objects": []
    }
    infile = tmp_path / "stage2.jsonl"
    outfile = tmp_path / "stage3.jsonl"
    infile.write_text(json.dumps(record) + "\n", encoding="utf-8")

    count = process_jsonl(infile, outfile)
    assert count == 1
    assert outfile.is_file()

    content = outfile.read_text(encoding="utf-8")
    graph = json.loads(content)
    graph_validator.validate(graph)
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["id"] == 1
