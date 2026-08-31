"""The pipeline's real output must satisfy the schemas it claims to.

Every other schema test in this repo validates hand-built records. That is
useful and it is not the same thing: a hand-built record contains exactly the
fields the test author remembered, so a field the CODE emits and the schema does
not declare is invisible to it.

An audit found four such violations at once, every one silent:

* Stage 1 records had gained a ``scene`` id from ``tools/batch_session``
* graph nodes emitted ``action_confidence``, never declared
* ``shared_action`` edges omitted the five features every edge must carry, and
  added an undeclared ``action``
* the object-class enum still listed three classes after the detector whitelist
  was widened to seventeen, so every frame containing a tv, bottle or keyboard
  was invalid

0 of 500 Stage 1 records and 0 of 500 graph records validated at that point,
while every schema test passed. These tests close that gap by driving the real
functions and validating what actually comes out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def stage1_validator():
    schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


@pytest.fixture(scope="module")
def graph_validator():
    schema = json.loads((ROOT / "graph_schema.json").read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def _record(objects=(), scene=None):
    """A Stage 1 record built the way the pipeline builds one."""
    from backend.integrate import _assemble_frame
    from backend.posture import PostureResult

    class _Person:
        bbox = (100, 100, 80, 160)
        confidence = 0.9

    class _Obj:
        def __init__(self, cls):
            self.cls = cls
            self.bbox = (110, 200, 40, 30)
            self.confidence = 0.6

    posture = PostureResult(
        keypoints_detected=True, nose=(140.0, 130.0),
        left_shoulder=(120.0, 160.0), right_shoulder=(160.0, 160.0),
        shoulder_mid=(140.0, 160.0), hip_mid=(140.0, 230.0),
        vertical_lean=-0.25, facing_direction=(0.0, -1.0),
    )
    record = _assemble_frame(
        0, 0, [_Person()], [None], [None], [posture], [None], [None],
        [1], [1], [_Obj(c) for c in objects],
    )
    if scene is not None:
        record["scene"] = scene
    return record


class TestStage1:
    def test_a_real_assembled_frame_validates(self, stage1_validator):
        stage1_validator.validate(_record())

    def test_the_scene_id_batch_session_adds_is_declared(self, stage1_validator):
        """tools/batch_session tags every record with its scene."""
        stage1_validator.validate(_record(scene=3))

    def test_every_whitelisted_object_class_is_allowed(self, stage1_validator):
        """The enum and the detector whitelist must not drift apart.

        They did: the whitelist grew from three classes to seventeen and the
        schema was not updated, so any frame containing a bottle or a keyboard
        was invalid while every test passed.
        """
        from backend.config import CONFIG

        for cls in CONFIG.detection.object_whitelist:
            stage1_validator.validate(_record(objects=[cls]))


class TestSceneGraph:
    @staticmethod
    def _graph(record):
        from backend.actions import annotate_graph
        from backend.config import CONFIG
        from backend.scene_graph import generate_scene_graph
        from backend.scene_layout import annotate as annotate_layout

        graph = generate_scene_graph(record, CONFIG)
        return annotate_graph(annotate_layout(graph, record), record, CONFIG)

    def test_a_real_graph_validates(self, graph_validator):
        graph_validator.validate(self._graph(_record()))

    def test_a_graph_with_objects_validates(self, graph_validator):
        graph_validator.validate(self._graph(_record(objects=["cell phone"])))

    def test_shared_action_edges_carry_every_required_feature(self, graph_validator):
        """An edge type added later must still satisfy the edge contract.

        shared_action edges shipped with only ``action`` in their features,
        which silently invalidated every graph frame containing one. The
        pair-interaction fields are null rather than absent: a consumer should
        not have to branch on edge type to read a field.
        """
        from backend.actions import annotate_graph
        from backend.config import CONFIG
        from backend.scene_graph import generate_scene_graph

        record = _record()
        # Two students, same posture, both looking away -> a shared action.
        record["persons"].append(json.loads(json.dumps(record["persons"][0])))
        record["persons"][1]["person_id"] = 2
        record["persons"][1]["bbox"] = [400, 100, 80, 160]
        for person in record["persons"]:
            person["head_pose"] = {"yaw": 60.0, "pitch": 0.0, "roll": 0.0,
                                   "gaze_label": "right"}

        graph = annotate_graph(generate_scene_graph(record, CONFIG), record, CONFIG)
        shared = [e for e in graph["edges"] if e["type"] == "shared_action"]
        assert shared, "expected a shared_action edge between two students"

        required = json.loads((ROOT / "graph_schema.json").read_text(encoding="utf-8"))
        required = required["$defs"]["edge"]["properties"]["features"]["required"]
        for edge in shared:
            assert set(required) <= set(edge["features"]), (
                f"shared_action edge missing {set(required) - set(edge['features'])}")
        graph_validator.validate(graph)

    def test_every_node_feature_the_code_writes_is_declared(self):
        """Catch a new feature key before it invalidates a whole run."""
        schema = json.loads((ROOT / "graph_schema.json").read_text(encoding="utf-8"))
        declared = set(
            schema["$defs"]["node"]["properties"]["features"]["properties"])
        graph = self._graph(_record(objects=["book"]))
        for node in graph["nodes"]:
            undeclared = set(node.get("features", {})) - declared
            assert not undeclared, f"undeclared node features: {undeclared}"
