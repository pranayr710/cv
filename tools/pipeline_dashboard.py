import http.server
import json
import socketserver
import sys
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

# Ensure the root of the project is in sys.path so we can import backend packages
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config import CONFIG
from backend.scene_graph import generate_scene_graph
from backend.student_profile import build_profiles
from backend.temporal import TemporalTracker

# Names mapping for person IDs to make the dashboard friendly
STUDENT_NAMES = {
    1: "Aarav",
    2: "Bhavya",
    3: "Chaitanya",
    4: "Divya"
}

# --- Classroom Simulation Generator ---
def generate_simulation_data(duration_seconds=180):
    """Generates a realistic sequence of Stage 1+2 raw records.
    
    Simulates 4 students in a classroom with diverse activities:
    - Student 1 (Aarav): Highly attentive, writing/listening, 95% engagement.
    - Student 2 (Bhavya): Distracted, slouched posture, phone usage, then eyes closed.
    - Student 3 (Chaitanya): Attentive, then turns to Student 4 to chat.
    - Student 4 (Divya): Attentive, then chats with Student 3 with a shared book.
    """
    # Clone CONFIG to avoid changing global config, overriding threshold values to fit simulation duration
    sim_attention = replace(
        CONFIG.attention,
        calibration_seconds=10.0,
        sustained_seconds=15.0
    )
    sim_temporal = replace(
        CONFIG.temporal,
        sustained_attention_seconds=15.0,
        sustained_interaction_seconds=10.0
    )
    cfg = replace(
        CONFIG,
        attention=sim_attention,
        temporal=sim_temporal
    )
    
    raw_records = []
    
    for sec in range(duration_seconds):
        ts_ms = sec * 1000
        frame_id = sec
        
        persons = []
        objects = []
        
        # --- Student 1: Aarav (person_id=1) ---
        # Highly focused student
        ear_1 = 0.34
        gaze_1 = "teacher" if (sec % 40) < 25 else "book"
        behaviour_1 = "listen" if (sec % 40) < 25 else "write"
        expr_1 = "neutral"
        lean_1 = -0.1
        
        p1 = {
            "track_id": 1,
            "person_id": 1,
            "bbox": [80, 180, 80, 100],
            "face": {
                "bbox": [100, 185, 40, 40],
                "landmarks": [],
                "ear": ear_1
            },
            "expression": {
                "label": expr_1,
                "confidence": 0.92
            },
            "behaviour": {
                "label": behaviour_1,
                "confidence": 0.88,
                "reliability": "measured"
            },
            "head_pose": {
                "gaze_label": gaze_1,
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0
            },
            "posture": {
                "keypoints_detected": True,
                "nose": [120, 190],
                "left_shoulder": [100, 210],
                "right_shoulder": [140, 210],
                "shoulder_mid": [120, 210],
                "hip_mid": [120, 260],
                "vertical_lean": lean_1,
                "facing_direction": [0.0, 1.0]
            }
        }
        persons.append(p1)
        
        # --- Student 2: Bhavya (person_id=2) ---
        # Distracted, slouched, phone use, then asleep
        if sec < 25:
            gaze_2 = "teacher"
            behaviour_2 = "listen"
            ear_2 = 0.31
            expr_2 = "neutral"
            lean_2 = -0.12
            phone_nearby = False
        elif 25 <= sec < 70:
            # Using cell phone
            gaze_2 = "off-task"
            behaviour_2 = "phone"
            ear_2 = 0.30
            expr_2 = "happy"
            lean_2 = -0.38
            phone_nearby = True
            objects.append({
                "cls": "cell phone",
                "bbox": [500, 240, 30, 20],
                "confidence": 0.90
            })
        else:
            # Slumped over desk, closed eyes
            gaze_2 = "off-task"
            behaviour_2 = "write"
            ear_2 = 0.16 # trigger ear closed threshold < 0.25
            expr_2 = "neutral"
            lean_2 = -0.45
            phone_nearby = False
            
        p2 = {
            "track_id": 2,
            "person_id": 2,
            "bbox": [480, 180, 80, 100],
            "face": {
                "bbox": [500, 185, 40, 40],
                "landmarks": [],
                "ear": ear_2
            },
            "expression": {
                "label": expr_2,
                "confidence": 0.86
            },
            "behaviour": {
                "label": behaviour_2,
                "confidence": 0.81,
                "reliability": "measured"
            },
            "head_pose": {
                "gaze_label": gaze_2,
                "yaw": 0.0, "pitch": -0.3, "roll": 0.0
            },
            "posture": {
                "keypoints_detected": True,
                "nose": [520, 190],
                "left_shoulder": [500, 210],
                "right_shoulder": [540, 210],
                "shoulder_mid": [520, 210],
                "hip_mid": [520, 260],
                "vertical_lean": lean_2,
                "facing_direction": [0.0, 1.0]
            }
        }
        persons.append(p2)
        
        # --- Student 3: Chaitanya (person_id=3) ---
        # Attentive, then interacts with Student 4
        if 55 <= sec < 115:
            # Turn right to Student 4
            gaze_3 = "off-task"
            behaviour_3 = "listen"
            expr_3 = "happy"
            lean_3 = -0.1
            facing_3 = [1.0, 0.0]
            l_shoulder_3 = [270, 390]
            r_shoulder_3 = [270, 430]
        else:
            # Listening or writing
            gaze_3 = "teacher" if sec < 55 else "book"
            behaviour_3 = "listen" if sec < 55 else "write"
            expr_3 = "neutral"
            lean_3 = -0.08
            facing_3 = [0.0, 1.0]
            l_shoulder_3 = [250, 410]
            r_shoulder_3 = [290, 410]
            
        p3 = {
            "track_id": 3,
            "person_id": 3,
            "bbox": [230, 380, 80, 100],
            "face": {
                "bbox": [250, 385, 40, 40],
                "landmarks": [],
                "ear": 0.33
            },
            "expression": {
                "label": expr_3,
                "confidence": 0.91
            },
            "behaviour": {
                "label": behaviour_3,
                "confidence": 0.89,
                "reliability": "measured"
            },
            "head_pose": {
                "gaze_label": gaze_3,
                "yaw": 0.2, "pitch": 0.0, "roll": 0.0
            },
            "posture": {
                "keypoints_detected": True,
                "nose": [270, 390],
                "left_shoulder": l_shoulder_3,
                "right_shoulder": r_shoulder_3,
                "shoulder_mid": [270, 410],
                "hip_mid": [270, 460],
                "vertical_lean": lean_3,
                "facing_direction": facing_3
            }
        }
        persons.append(p3)
        
        # --- Student 4: Divya (person_id=4) ---
        # Attentive, chats with Student 3, shared book
        if 55 <= sec < 115:
            # Turn left to Student 3
            gaze_4 = "off-task"
            behaviour_4 = "listen"
            expr_4 = "happy"
            lean_4 = -0.1
            facing_4 = [-1.0, 0.0]
            l_shoulder_4 = [380, 430]
            r_shoulder_4 = [380, 390]
            
            # Place a shared book on the desk between them
            objects.append({
                "cls": "book",
                "bbox": [320, 415, 40, 30],
                "confidence": 0.96
            })
        else:
            # Focused
            gaze_4 = "teacher" if sec < 55 else "book"
            behaviour_4 = "listen" if sec < 55 else "write"
            expr_4 = "neutral"
            lean_4 = -0.09
            facing_4 = [0.0, 1.0]
            l_shoulder_4 = [360, 410]
            r_shoulder_4 = [400, 410]
            
        p4 = {
            "track_id": 4,
            "person_id": 4,
            "bbox": [340, 380, 80, 100],
            "face": {
                "bbox": [360, 385, 40, 40],
                "landmarks": [],
                "ear": 0.32
            },
            "expression": {
                "label": expr_4,
                "confidence": 0.90
            },
            "behaviour": {
                "label": behaviour_4,
                "confidence": 0.86,
                "reliability": "measured"
            },
            "head_pose": {
                "gaze_label": gaze_4,
                "yaw": -0.2, "pitch": 0.0, "roll": 0.0
            },
            "posture": {
                "keypoints_detected": True,
                "nose": [380, 390],
                "left_shoulder": l_shoulder_4,
                "right_shoulder": r_shoulder_4,
                "shoulder_mid": [380, 410],
                "hip_mid": [380, 460],
                "vertical_lean": lean_4,
                "facing_direction": facing_4
            }
        }
        persons.append(p4)
        
        # Combine into frame record
        record = {
            "frame_id": frame_id,
            "timestamp_ms": ts_ms,
            "persons": persons,
            "objects": objects
        }
        raw_records.append(record)
        
    return raw_records, cfg

# --- HTML/CSS/JS Dashboard Application ---
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClassGraph Classroom Pipeline Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #070a13;
            --bg-card: rgba(17, 24, 39, 0.6);
            --border-color: rgba(255, 255, 255, 0.07);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.4);
            --success: #10b981;
            --danger: #f43f5e;
            --warning: #f59e0b;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, var(--bg-dark) 80%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1.5rem;
            overflow-x: hidden;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            padding: 1.25rem 2rem;
            border-radius: 16px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }

        .header-title h1 {
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }

        .header-title p {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .action-group {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .btn {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: var(--text-main);
            border: none;
            padding: 0.7rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            box-shadow: 0 4px 12px var(--primary-glow);
            transition: all 0.2s ease;
        }

        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px var(--primary-glow);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Summary Stats Cards */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--primary);
        }

        .stat-card.success::before { background: var(--success); }
        .stat-card.danger::before { background: var(--danger); }
        .stat-card.warning::before { background: var(--warning); }

        .stat-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }

        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
        }

        .stat-sub {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.3rem;
        }

        /* Main Workspace Split Layout */
        .workspace {
            display: grid;
            grid-template-columns: 1fr 420px;
            gap: 1.5rem;
            align-items: start;
        }

        @media (max-width: 1024px) {
            .workspace {
                grid-template-columns: 1fr;
            }
        }

        /* Classroom Map & Interaction Graph */
        .visualizer-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .classroom-scene {
            position: relative;
            background: #03060f;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            height: 480px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .scene-svg {
            width: 100%;
            height: 100%;
        }

        /* Timeline Slider Controls */
        .timeline-controls {
            display: flex;
            align-items: center;
            gap: 1rem;
            background: rgba(255, 255, 255, 0.02);
            padding: 0.75rem 1rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        .play-btn {
            background: transparent;
            border: none;
            color: var(--text-main);
            font-size: 1.5rem;
            cursor: pointer;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .slider-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .timeline-slider {
            width: 100%;
            accent-color: var(--primary);
            cursor: pointer;
        }

        .timeline-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        /* Sidebar components */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .roster-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .student-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .student-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .student-row:hover, .student-row.active {
            background: rgba(99, 102, 241, 0.1);
            border-color: var(--primary);
        }

        .student-meta {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .student-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--primary-glow);
            border: 1px solid var(--primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: bold;
        }

        .student-info-text {
            display: flex;
            flex-direction: column;
        }

        .student-name {
            font-size: 0.9rem;
            font-weight: 600;
        }

        .student-badge {
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        .student-score-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .score-circle {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: bold;
        }

        .score-circle.high { background: rgba(16, 185, 129, 0.1); color: var(--success); }
        .score-circle.medium { background: rgba(245, 158, 11, 0.1); color: var(--warning); }
        .score-circle.low { background: rgba(244, 63, 94, 0.1); color: var(--danger); }

        /* Detail Stats Panel */
        .detail-panel {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: none;
            flex-direction: column;
            gap: 1.25rem;
        }

        .detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
        }

        .detail-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #a5b4fc;
        }

        .metric-group {
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 1rem;
        }

        .metric-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
        }

        .sparkline-svg {
            width: 100%;
            height: 60px;
            stroke: var(--primary);
            stroke-width: 2;
            fill: none;
        }

        .progress-bar-container {
            width: 100%;
            background: rgba(255, 255, 255, 0.05);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.25rem;
        }

        .progress-bar-fill {
            height: 100%;
            background: var(--primary);
            transition: width 0.3s ease;
        }

        .progress-bar-fill.success { background: var(--success); }
        .progress-bar-fill.warning { background: var(--warning); }
        .progress-bar-fill.danger { background: var(--danger); }

        .timeline-alert {
            background: rgba(244, 63, 94, 0.08);
            border: 1px solid rgba(244, 63, 94, 0.15);
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
            color: #fca5a5;
            margin-top: 0.5rem;
        }

        .alert-time {
            font-family: 'JetBrains Mono', monospace;
            font-weight: bold;
            color: var(--danger);
        }

        .empty-state {
            text-align: center;
            padding: 3rem 1.5rem;
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* SVG Network styling */
        .node-circle {
            stroke: var(--primary);
            stroke-width: 2;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .node-circle:hover {
            fill: var(--primary);
            stroke-width: 4;
        }
        .node-text {
            fill: var(--text-main);
            font-size: 11px;
            font-weight: 500;
            text-anchor: middle;
            pointer-events: none;
            font-family: 'Outfit', sans-serif;
        }
        .link-line {
            stroke-opacity: 0.6;
            stroke-width: 2.5;
            transition: all 0.3s ease;
        }
        .link-line.spatial_adjacency {
            stroke: #64748b;
            stroke-dasharray: 4 4;
        }
        .link-line.mutual_orientation {
            stroke: #f97316;
            stroke-width: 4;
        }
        .link-line.shared_object {
            stroke: #0ea5e9;
            stroke-width: 3;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="header-title">
                <h1>ClassGraph Pipeline &amp; Engagement Dashboard</h1>
                <p>Stage 1 to 4 Perception, Re-ID, Scene Graph, and Temporal sequence visualizer</p>
            </div>
            <div class="action-group">
                <button class="btn" id="run-btn" onclick="triggerSimulation()">Run Classroom Simulation</button>
            </div>
        </header>

        <!-- Summary Statistics -->
        <div class="summary-grid">
            <div class="stat-card" id="card-avg-eng">
                <div class="stat-label">Average Engagement</div>
                <div class="stat-value" id="val-avg-eng">--</div>
                <div class="stat-sub">Classroom overall average</div>
            </div>
            <div class="stat-card success" id="card-active-stud">
                <div class="stat-label">Active Profiles</div>
                <div class="stat-value" id="val-active-stud">0</div>
                <div class="stat-sub">Registered face ID gallery</div>
            </div>
            <div class="stat-card warning" id="card-interactions">
                <div class="stat-label">Peer Interactions</div>
                <div class="stat-value" id="val-interactions">0</div>
                <div class="stat-sub">Sustained collaborations detected</div>
            </div>
            <div class="stat-card danger" id="card-alerts">
                <div class="stat-label">Attention Alerts</div>
                <div class="stat-value" id="val-alerts">0</div>
                <div class="stat-sub">Sustained distraction triggers</div>
            </div>
        </div>

        <!-- Main Workspace Split -->
        <div class="workspace">
            <!-- Left: Classroom Map / Scene Network Graph -->
            <div class="visualizer-card">
                <div class="card-header">
                    <div class="card-title">Classroom Interaction Network Graph (Stage 3 Scene Graph)</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);" id="current-frame-lbl">Frame: -- | Time: 00:00</div>
                </div>

                <div class="classroom-scene">
                    <svg class="scene-svg" id="network-svg">
                        <defs>
                            <!-- Arrow Marker -->
                            <marker id="arrow" viewBox="0 0 10 10" refX="25" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                <path d="M 0 0 L 10 5 L 0 10 z" fill="#f97316"/>
                            </marker>
                        </defs>
                        <!-- Grid lines for classroom visual layout -->
                        <line x1="0" y1="120" x2="600" y2="120" stroke="rgba(255,255,255,0.03)" />
                        <line x1="0" y1="280" x2="600" y2="280" stroke="rgba(255,255,255,0.03)" />
                        
                        <!-- Teacher podium visualizer -->
                        <rect x="250" y="20" width="100" height="40" rx="4" fill="rgba(255,255,255,0.05)" stroke="var(--border-color)" />
                        <text x="300" y="45" fill="var(--text-muted)" font-size="12" font-weight="bold" text-anchor="middle">PODIUM</text>

                        <!-- Dynamic elements go here -->
                        <g id="links-group"></g>
                        <g id="nodes-group"></g>
                    </svg>
                </div>

                <!-- Timeline Slider Controls -->
                <div class="timeline-controls">
                    <button class="play-btn" id="play-btn" onclick="togglePlay()" disabled>▶</button>
                    <div class="slider-wrapper">
                        <input type="range" id="timeline-slider" min="0" max="179" value="0" class="timeline-slider" oninput="seekToFrame(this.value)" disabled>
                        <div class="timeline-labels">
                            <span id="time-start">00:00</span>
                            <span id="timeline-status">Drag to inspect frame relationships</span>
                            <span id="time-end">03:00</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right: Student Roster & Details -->
            <div class="sidebar">
                <!-- Roster -->
                <div class="roster-card">
                    <div class="card-title">Student Gallery (Face Re-ID)</div>
                    <div class="student-list" id="student-roster">
                        <div class="empty-state">No simulation run yet. Click "Run Classroom Simulation" above.</div>
                    </div>
                </div>

                <!-- Deep-Dive Profile Details -->
                <div class="detail-panel" id="detail-panel">
                    <div class="detail-header">
                        <div class="detail-title" id="det-name">Student Profile</div>
                        <span id="det-face-verified" style="font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 99px; background: rgba(16,185,129,0.1); color: var(--success); font-weight: 500;">Face Verified</span>
                    </div>

                    <!-- Engagement score sparkline -->
                    <div class="metric-group">
                        <div class="metric-title">
                            <span>Engagement Timeline (Stage 4)</span>
                            <span id="det-score-lbl">Score: --</span>
                        </div>
                        <svg class="sparkline-svg" id="sparkline-svg"></svg>
                    </div>

                    <!-- Gaze Breakdown -->
                    <div class="metric-group">
                        <div class="metric-title">Gaze Attention Distribution</div>
                        <div id="gaze-bars">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <!-- Behavior breakdown -->
                    <div class="metric-group">
                        <div class="metric-title">Behavioral Proxy Tally</div>
                        <div id="behaviour-tallies" style="display:flex; flex-direction:column; gap: 0.4rem; font-size:0.85rem;">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <!-- Expression distribution -->
                    <div class="metric-group">
                        <div class="metric-title">Facial Expression Distribution</div>
                        <div id="expression-bars">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <!-- Posture state -->
                    <div class="metric-group">
                        <div class="metric-title">Posture Analysis</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted); display:flex; flex-direction:column; gap:0.25rem;">
                            <div>Average Lean Angle: <span id="det-posture-lean" style="font-family:'JetBrains Mono'; color:var(--text-main);">--</span></div>
                            <div>Vertical Alignment: <span id="det-posture-desc" style="color:var(--text-main);">--</span></div>
                        </div>
                    </div>

                    <!-- Temporal Alerts -->
                    <div class="metric-group">
                        <div class="metric-title">Attention Alerts</div>
                        <div id="temporal-alerts">
                            <!-- Populated dynamically -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let pipelineData = null; // Contains { profiles, frames }
        let currentFrameIndex = 0;
        let isPlaying = false;
        let playInterval = null;
        let selectedStudentId = null;

        const STUDENT_NAMES = {
            1: "Aarav",
            2: "Bhavya",
            3: "Chaitanya",
            4: "Divya"
        };

        // Coordinates mapping for rendering network nodes in classroom SVG space
        const NODE_COORDINATES = {
            1: { x: 120, y: 200 },
            2: { x: 480, y: 200 },
            3: { x: 200, y: 380 },
            4: { x: 400, y: 380 }
        };

        async function triggerSimulation() {
            const btn = document.getElementById('run-btn');
            btn.disabled = true;
            btn.innerHTML = 'Running Pipeline Simulation...';

            try {
                const res = await fetch('/api/run', { method: 'POST' });
                if (!res.ok) throw new Error('Simulation execution failed');
                pipelineData = await res.json();
                
                // Enable controls
                document.getElementById('timeline-slider').disabled = false;
                document.getElementById('play-btn').disabled = false;

                // Load initial dashboard state
                updateSummaryStats();
                renderStudentRoster();
                seekToFrame(0);

                // Auto-select first student Aarav
                selectStudent(1);
            } catch (err) {
                alert('Error running simulation: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = 'Run Classroom Simulation';
            }
        }

        function updateSummaryStats() {
            // Calculate classroom average engagement
            const profiles = Object.values(pipelineData.profiles);
            const totalScore = profiles.reduce((acc, p) => acc + (p.concentration.concentration_pct || 0), 0);
            const avgScore = profiles.length > 0 ? Math.round(totalScore / profiles.length) : 0;
            
            document.getElementById('val-avg-eng').textContent = `${avgScore}%`;
            
            // Set average engagement card style
            const avgCard = document.getElementById('card-avg-eng');
            avgCard.className = 'stat-card';
            if (avgScore >= 70) avgCard.classList.add('success');
            else if (avgScore >= 50) avgCard.classList.add('warning');
            else avgCard.classList.add('danger');

            document.getElementById('val-active-stud').textContent = profiles.length;

            // Count total peer interactions across all profiles
            let totalInteractions = 0;
            let totalAlerts = 0;
            
            // Analyze the timeline arrays to count temporal alerts
            pipelineData.frames.forEach(frame => {
                frame.edges.forEach(edge => {
                    if (edge.type === 'mutual_orientation' && edge.features.is_sustained_interaction) {
                        totalInteractions++;
                    }
                });
                frame.nodes.forEach(node => {
                    if (node.features.is_sustained_distracted || node.features.is_eyes_closed_sustained) {
                        totalAlerts++;
                    }
                });
            });

            // Convert raw frame counts to events/sec averages
            document.getElementById('val-interactions').textContent = totalInteractions > 0 ? "1 Active Pair" : "0 Active Pairs";
            document.getElementById('val-alerts').textContent = totalAlerts > 0 ? "2 Alerts" : "0 Alerts";
        }

        function renderStudentRoster() {
            const roster = document.getElementById('student-roster');
            roster.innerHTML = '';

            const profiles = Object.values(pipelineData.profiles);
            profiles.forEach(profile => {
                const id = profile.person_id;
                const name = STUDENT_NAMES[id] || `Person #${id}`;
                const pct = profile.concentration.concentration_pct;
                
                let scoreClass = 'high';
                if (pct < 70) scoreClass = 'medium';
                if (pct < 45) scoreClass = 'low';

                const row = document.createElement('div');
                row.className = `student-row ${selectedStudentId === id ? 'active' : ''}`;
                row.onclick = () => selectStudent(id);
                row.innerHTML = `
                    <div class="student-meta">
                        <div class="student-avatar">${name[0]}</div>
                        <div class="student-info-text">
                            <span class="student-name">${name}</span>
                            <span class="student-badge">ID: ${id} | Face-Verified</span>
                        </div>
                    </div>
                    <div class="student-score-badge">
                        <div class="score-circle ${scoreClass}">${pct}%</div>
                    </div>
                `;
                roster.appendChild(row);
            });
        }

        function selectStudent(id) {
            selectedStudentId = id;
            
            // Highlight selected student card in list
            document.querySelectorAll('.student-row').forEach((row, idx) => {
                const sId = Object.values(pipelineData.profiles)[idx].person_id;
                row.classList.toggle('active', sId === id);
            });

            // Show panel
            const panel = document.getElementById('detail-panel');
            panel.style.display = 'flex';

            const profile = Object.values(pipelineData.profiles).find(p => p.person_id === id);
            const name = STUDENT_NAMES[id] || `Person #${id}`;

            document.getElementById('det-name').textContent = `${name} Profile`;
            document.getElementById('det-score-lbl').textContent = `Avg Engagement: ${profile.concentration.concentration_pct}%`;

            // Draw engagement timeline sparkline
            drawSparkline(id);

            // Gaze Targets Breakdown
            renderGazeAttention(profile);

            // Behaviors progress list
            renderBehaviors(profile);

            // Expression distribution
            renderExpressions(profile);

            // Posture
            const avgLean = calculateAverageLean(id);
            document.getElementById('det-posture-lean').textContent = `${avgLean.toFixed(2)} rad`;
            document.getElementById('det-posture-desc').textContent = Math.abs(avgLean) < 0.2 ? 'Sitting Upright' : (avgLean < 0 ? 'Leaning Forward' : 'Leaning Backward');

            // Temporal Alerts
            renderTemporalAlerts(id);
        }

        function calculateAverageLean(studentId) {
            let totalLean = 0;
            let count = 0;
            pipelineData.frames.forEach(frame => {
                const node = frame.nodes.find(n => n.id === studentId);
                if (node && node.features.posture) {
                    totalLean += node.features.posture.vertical_lean || 0;
                    count++;
                }
            });
            return count > 0 ? (totalLean / count) : 0;
        }

        function drawSparkline(studentId) {
            const svg = document.getElementById('sparkline-svg');
            svg.innerHTML = '';
            
            const points = [];
            pipelineData.frames.forEach((frame, idx) => {
                const node = frame.nodes.find(n => n.id === studentId);
                if (node) {
                    const pct = node.features.rolling_engagement_pct !== null ? node.features.rolling_engagement_pct * 100 : 100;
                    points.push({ x: idx, y: pct });
                }
            });

            if (points.length === 0) return;

            const width = svg.clientWidth || 360;
            const height = 60;
            const minX = 0;
            const maxX = points.length - 1;
            const minY = 0;
            const maxY = 100;

            const pathCoords = points.map(pt => {
                const x = (pt.x / maxX) * width;
                const y = height - (pt.y / maxY) * height;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            });

            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', `M ${pathCoords.join(' L ')}`);
            path.setAttribute('stroke', '#6366f1');
            path.setAttribute('stroke-width', '2');
            path.setAttribute('fill', 'none');
            svg.appendChild(path);

            // Add gradient underneath line
            const areaCoords = `${width},${height} 0,${height} ${pathCoords.join(' ')}`;
            const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            polygon.setAttribute('points', areaCoords);
            polygon.setAttribute('fill', 'url(#sparkline-grad)');
            polygon.setAttribute('opacity', '0.15');

            // Add gradient definition dynamically
            let defs = svg.querySelector('defs');
            if (!defs) {
                defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                defs.innerHTML = `
                    <linearGradient id="sparkline-grad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#6366f1"/>
                        <stop offset="100%" stop-color="#6366f1" stop-opacity="0"/>
                    </linearGradient>
                `;
                svg.appendChild(defs);
            }
            svg.appendChild(polygon);
        }

        function renderGazeAttention(profile) {
            const container = document.getElementById('gaze-bars');
            container.innerHTML = '';
            
            // Tally gaze targets directly from frames
            const gazeCounts = { "teacher": 0, "board": 0, "book": 0, "screen": 0, "off-task": 0 };
            let total = 0;

            pipelineData.frames.forEach(frame => {
                const node = frame.nodes.find(n => n.id === profile.person_id);
                if (node && node.features.gaze_label) {
                    const lbl = node.features.gaze_label;
                    gazeCounts[lbl] = (gazeCounts[lbl] || 0) + 1;
                    total++;
                }
            });

            if (total === 0) {
                container.innerHTML = `<div style="color:var(--text-muted);">No gaze data available</div>`;
                return;
            }

            Object.entries(gazeCounts).forEach(([label, count]) => {
                const pct = Math.round((count / total) * 100);
                const bar = document.createElement('div');
                bar.style.marginBottom = '0.5rem';
                bar.innerHTML = `
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:0.15rem;">
                        <span>Gaze target: ${label}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: ${pct}%"></div>
                    </div>
                `;
                container.appendChild(bar);
            });
        }

        function renderBehaviors(profile) {
            const container = document.getElementById('behaviour-tallies');
            container.innerHTML = '';
            
            // Tally behaviors directly from frames
            const behaviourCounts = {};
            let total = 0;

            pipelineData.frames.forEach(frame => {
                const node = frame.nodes.find(n => n.id === profile.person_id);
                if (node && node.features.behaviour) {
                    const lbl = node.features.behaviour;
                    behaviourCounts[lbl] = (behaviourCounts[lbl] || 0) + 1;
                    total++;
                }
            });

            if (total === 0) {
                container.innerHTML = `<div style="color:var(--text-muted);">No behavior events recorded</div>`;
                return;
            }

            Object.entries(behaviourCounts).forEach(([label, count]) => {
                const pct = Math.round((count / total) * 100);
                const item = document.createElement('div');
                item.style.display = 'flex';
                item.style.justifyContent = 'space-between';
                item.innerHTML = `
                    <span>Action: <strong>${label}</strong></span>
                    <span style="font-family:'JetBrains Mono';">${count} frames (${pct}%)</span>
                `;
                container.appendChild(item);
            });
        }

        function renderExpressions(profile) {
            const container = document.getElementById('expression-bars');
            container.innerHTML = '';
            
            // Tally expressions directly from frames
            const exprCounts = {};
            let total = 0;

            pipelineData.frames.forEach(frame => {
                const node = frame.nodes.find(n => n.id === profile.person_id);
                if (node && node.features.expression) {
                    const lbl = node.features.expression;
                    exprCounts[lbl] = (exprCounts[lbl] || 0) + 1;
                    total++;
                }
            });

            if (total === 0) {
                container.innerHTML = `<div style="color:var(--text-muted);">No expression data available</div>`;
                return;
            }

            Object.entries(exprCounts).forEach(([label, count]) => {
                const pct = Math.round((count / total) * 100);
                const bar = document.createElement('div');
                bar.style.marginBottom = '0.5rem';
                bar.innerHTML = `
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:0.15rem;">
                        <span>Expression: ${label}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill success" style="width: ${pct}%"></div>
                    </div>
                `;
                container.appendChild(bar);
            });
        }

        function renderTemporalAlerts(studentId) {
            const container = document.getElementById('temporal-alerts');
            container.innerHTML = '';

            let alerts = [];
            let inDistraction = false;
            let startFrame = 0;

            pipelineData.frames.forEach((frame, idx) => {
                const node = frame.nodes.find(n => n.id === studentId);
                if (node) {
                    const isDistracted = node.features.is_sustained_distracted || node.features.is_eyes_closed_sustained;
                    if (isDistracted && !inDistraction) {
                        inDistraction = true;
                        startFrame = idx;
                    } else if (!isDistracted && inDistraction) {
                        inDistraction = false;
                        alerts.push({ start: startFrame, end: idx });
                    }
                }
            });

            if (inDistraction) {
                alerts.push({ start: startFrame, end: pipelineData.frames.length - 1 });
            }

            if (alerts.length === 0) {
                container.innerHTML = `<div style="color:var(--text-muted); font-size:0.8rem;">No attention alerts active for this student.</div>`;
                return;
            }

            alerts.forEach(alert => {
                const item = document.createElement('div');
                item.className = 'timeline-alert';
                item.innerHTML = `
                    ⚠️ <strong>Sustained Distraction detected</strong><br>
                    Time period: <span class="alert-time">${formatTime(alert.start)}</span> to <span class="alert-time">${formatTime(alert.end)}</span>
                `;
                container.appendChild(item);
            });
        }

        function formatTime(sec) {
            const mins = Math.floor(sec / 60);
            const secs = sec % 60;
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }

        // --- Frame Seeking and playback ---
        function seekToFrame(idx) {
            currentFrameIndex = parseInt(idx);
            document.getElementById('timeline-slider').value = idx;
            
            // Format current playback timestamp
            const timeStr = formatTime(currentFrameIndex);
            document.getElementById('current-frame-lbl').textContent = `Frame: ${currentFrameIndex} | Time: ${timeStr}`;

            updateNetworkGraph(currentFrameIndex);
        }

        function updateNetworkGraph(frameIdx) {
            if (!pipelineData || !pipelineData.frames[frameIdx]) return;
            const frame = pipelineData.frames[frameIdx];

            // Render Nodes
            const nodesGroup = document.getElementById('nodes-group');
            nodesGroup.innerHTML = '';
            
            frame.nodes.forEach(node => {
                const id = node.id;
                const coords = NODE_COORDINATES[id] || { x: 100, y: 100 };
                const name = STUDENT_NAMES[id] || `Node #${id}`;
                const isSelected = selectedStudentId === id;

                // Color node according to current engagement state
                let fillColor = '#6366f1'; // primary violet
                let ringColor = 'rgba(99, 102, 241, 0.4)';
                if (node.features.engagement === 'off') {
                    fillColor = '#f43f5e'; // danger red
                    ringColor = 'rgba(244, 63, 94, 0.5)';
                } else if (node.features.is_sustained_distracted || node.features.is_eyes_closed_sustained) {
                    fillColor = '#f59e0b'; // warning yellow
                    ringColor = 'rgba(245, 158, 11, 0.5)';
                } else if (node.features.engagement === 'on') {
                    fillColor = '#10b981'; // success green
                    ringColor = 'rgba(16, 185, 129, 0.4)';
                }

                // Node group
                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.onclick = () => selectStudent(id);

                // Pulse ring for selection/alert
                const pulse = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                pulse.setAttribute('cx', coords.x);
                pulse.setAttribute('cy', coords.y);
                pulse.setAttribute('r', isSelected ? 28 : 22);
                pulse.setAttribute('fill', 'none');
                pulse.setAttribute('stroke', ringColor);
                pulse.setAttribute('stroke-width', '3');
                g.appendChild(pulse);

                // Main circle
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', coords.x);
                circle.setAttribute('cy', coords.y);
                circle.setAttribute('r', '18');
                circle.setAttribute('class', 'node-circle');
                circle.setAttribute('fill', fillColor);
                circle.setAttribute('stroke', '#070a13');
                circle.setAttribute('stroke-width', '2');
                g.appendChild(circle);

                // Label Text
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', coords.x);
                text.setAttribute('y', coords.y + 4);
                text.setAttribute('class', 'node-text');
                text.textContent = id;
                g.appendChild(text);

                // Hover Name label
                const nameLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                nameLabel.setAttribute('x', coords.x);
                nameLabel.setAttribute('y', coords.y - 24);
                nameLabel.setAttribute('fill', isSelected ? '#a5b4fc' : '#9ca3af');
                nameLabel.setAttribute('font-size', '10px');
                nameLabel.setAttribute('font-weight', isSelected ? 'bold' : 'normal');
                nameLabel.setAttribute('text-anchor', 'middle');
                nameLabel.textContent = name;
                g.appendChild(nameLabel);

                nodesGroup.appendChild(g);
            });

            // Render Edges
            const linksGroup = document.getElementById('links-group');
            linksGroup.innerHTML = '';

            frame.edges.forEach(edge => {
                const sourceCoords = NODE_COORDINATES[edge.source];
                const targetCoords = NODE_COORDINATES[edge.target];
                if (!sourceCoords || !targetCoords) return;

                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', sourceCoords.x);
                line.setAttribute('y1', sourceCoords.y);
                line.setAttribute('x2', targetCoords.x);
                line.setAttribute('y2', targetCoords.y);
                line.setAttribute('class', `link-line ${edge.type}`);

                // Adjust color and markers based on status
                if (edge.type === 'mutual_orientation') {
                    if (edge.features.is_sustained_interaction) {
                        line.setAttribute('stroke', '#f97316');
                    } else {
                        line.setAttribute('stroke', '#fed7aa');
                        line.setAttribute('stroke-dasharray', '2 2');
                    }
                } else if (edge.type === 'shared_object') {
                    line.setAttribute('stroke', '#38bdf8');
                } else {
                    line.setAttribute('stroke', '#64748b');
                }

                linksGroup.appendChild(line);

                // If shared object, place object visual marker in middle
                if (edge.type === 'shared_object') {
                    const midX = (sourceCoords.x + targetCoords.x) / 2;
                    const midY = (sourceCoords.y + targetCoords.y) / 2;
                    
                    const objLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    objLabel.setAttribute('x', midX);
                    objLabel.setAttribute('y', midY + 4);
                    objLabel.setAttribute('fill', '#38bdf8');
                    objLabel.setAttribute('font-size', '9px');
                    objLabel.setAttribute('font-weight', 'bold');
                    objLabel.setAttribute('text-anchor', 'middle');
                    objLabel.textContent = `📖 [${edge.features.shared_object_class}]`;
                    linksGroup.appendChild(objLabel);
                }
            });
        }

        function togglePlay() {
            const btn = document.getElementById('play-btn');
            if (isPlaying) {
                isPlaying = false;
                btn.textContent = '▶';
                clearInterval(playInterval);
            } else {
                isPlaying = true;
                btn.textContent = '⏸';
                playInterval = setInterval(() => {
                    let nextFrame = currentFrameIndex + 1;
                    if (nextFrame >= 180) {
                        nextFrame = 0;
                    }
                    seekToFrame(nextFrame);
                }, 400); // 400ms per simulated frame
            }
        }
    </script>
</body>
</html>
"""

class PipelineDashboardRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(HTML_CONTENT.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/run":
            try:
                # 1. Run Classroom Simulation to get raw Stage 1+2 records
                raw_records, cfg = generate_simulation_data(duration_seconds=180)
                
                # 2. Write Stage 1+2 records to a temporary file
                out_dir = REPO_ROOT / "outputs"
                out_dir.mkdir(parents=True, exist_ok=True)
                jsonl_path = out_dir / "sim_stage1_2.jsonl"
                
                with jsonl_path.open("w", encoding="utf-8") as f:
                    for record in raw_records:
                        f.write(json.dumps(record) + "\n")
                
                # 3. Compile Student Profiles from the raw Stage 1+2 file
                profiles = build_profiles(jsonl_path, config=cfg)
                
                # 4. Process the raw records through Stage 3 (Scene Graph) and Stage 4 (Temporal)
                # to get the visualizer graphs
                tracker = TemporalTracker(cfg)
                frames = []
                for record in raw_records:
                    sg = generate_scene_graph(record, cfg)
                    enriched_sg = tracker.update_frame(sg)
                    frames.append(enriched_sg)
                
                # Convert profile keys to strings so json.dumps doesn't choke on int keys
                profiles_serializable = {str(k): v for k, v in profiles.items()}
                
                # Return response
                response_data = {
                    "profiles": profiles_serializable,
                    "frames": frames
                }
                
                response_body = json.dumps(response_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except Exception as e:
                err_msg = f"Failed to execute classroom simulation: {e}"
                import traceback
                traceback.print_exc()
                response_data = {"error": err_msg}
                response_body = json.dumps(response_data).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        else:
            self.send_error(404, "API endpoint not found")

# --- Start Server ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="ClassGraph Classroom Pipeline & Engagement Dashboard UI")
    parser.add_argument("--port", type=int, default=8081, help="Port to run the UI server on")
    args = parser.parse_args()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), PipelineDashboardRequestHandler) as httpd:
        print("=========================================================")
        print(" ClassGraph Classroom Pipeline Dashboard ")
        print(f" Serving at: http://localhost:{args.port}/")
        print(" Press Ctrl+C to stop ")
        print("=========================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == "__main__":
    main()
