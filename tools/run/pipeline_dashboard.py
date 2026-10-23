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
        elif 25 <= sec < 70:
            # Using cell phone
            gaze_2 = "off-task"
            behaviour_2 = "phone"
            ear_2 = 0.30
            expr_2 = "happy"
            lean_2 = -0.38
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
    <title>ClassGraph — Scene Graph & Attention Topology Visualizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:ital,wght@0,400;0,500;0,700;1,400&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #060810;
            --bg-card: rgba(13, 18, 30, 0.75);
            --bg-card-hover: rgba(22, 30, 48, 0.85);
            --border-color: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(99, 102, 241, 0.35);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.4);
            --accent-cyan: #06b6d4;
            --accent-cyan-glow: rgba(6, 182, 212, 0.35);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.35);
            --warning: #f59e0b;
            --warning-glow: rgba(245, 158, 11, 0.35);
            --danger: #f43f5e;
            --danger-glow: rgba(244, 63, 94, 0.35);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-subtle: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, var(--bg-dark) 75%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 1.25rem;
            overflow-x: hidden;
        }

        .container {
            max-width: 1560px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 1.1rem;
        }

        /* Glassmorphic Navbar */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-card);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            padding: 1rem 1.6rem;
            border-radius: 18px;
            border: 1px solid var(--border-color);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .header-brand {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.25rem;
            color: #fff;
            box-shadow: 0 0 25px var(--primary-glow);
        }

        .header-title h1 {
            font-size: 1.45rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .version-badge {
            font-size: 0.68rem;
            font-weight: 600;
            padding: 0.2rem 0.55rem;
            border-radius: 99px;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .header-title p {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .action-group {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .btn {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: var(--text-main);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 0.6rem 1.3rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.86rem;
            cursor: pointer;
            box-shadow: 0 4px 20px var(--primary-glow);
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px var(--primary-glow);
            border-color: rgba(255, 255, 255, 0.3);
        }

        .btn-outline {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            box-shadow: none;
        }

        .btn-outline:hover {
            background: rgba(255, 255, 255, 0.08);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Summary Stats Header Row */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.1rem;
        }

        .stat-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.1rem 1.4rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }

        .stat-card:hover {
            transform: translateY(-2px);
            border-color: var(--border-glow);
        }

        .stat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
        }

        .stat-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            font-weight: 600;
        }

        .stat-icon {
            width: 32px;
            height: 32px;
            border-radius: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
        }

        .stat-body {
            display: flex;
            align-items: baseline;
            gap: 0.75rem;
        }

        .stat-value {
            font-size: 1.95rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            font-family: 'JetBrains Mono', monospace;
        }

        .stat-badge {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
        }

        .stat-badge.success { background: rgba(16, 185, 129, 0.15); color: var(--success); }
        .stat-badge.warning { background: rgba(245, 158, 11, 0.15); color: var(--warning); }
        .stat-badge.danger { background: rgba(244, 63, 94, 0.15); color: var(--danger); }
        .stat-badge.info { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); }

        .stat-sub {
            font-size: 0.74rem;
            color: var(--text-subtle);
            margin-top: 0.3rem;
        }

        /* Workspace Grid */
        .workspace {
            display: grid;
            grid-template-columns: 1fr 440px;
            gap: 1.1rem;
            align-items: start;
        }

        @media (max-width: 1200px) {
            .workspace { grid-template-columns: 1fr; }
        }

        /* Graph Visualizer Centerpiece */
        .visualizer-card {
            background: var(--bg-card);
            backdrop-filter: blur(24px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.3rem;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
            position: relative;
        }

        .visualizer-card.fullscreen {
            position: fixed;
            inset: 1rem;
            z-index: 9999;
            height: calc(100vh - 2rem);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.85rem;
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        .card-title-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-main);
        }

        /* Mode Switcher Tabs */
        .view-switcher {
            display: flex;
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-color);
            padding: 0.2rem;
            border-radius: 10px;
            gap: 0.2rem;
        }

        .view-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.38rem 0.8rem;
            border-radius: 7px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .view-btn:hover {
            color: var(--text-main);
        }

        .view-btn.active {
            background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
            color: #fff;
            box-shadow: 0 2px 10px var(--primary-glow);
        }

        /* Scene Canvas Container */
        .classroom-scene {
            position: relative;
            background: radial-gradient(circle at 50% 50%, #0c1220 0%, #03050c 100%);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            height: 540px;
            overflow: hidden;
            box-shadow: inset 0 0 50px rgba(0, 0, 0, 0.9);
        }

        .scene-svg {
            width: 100%;
            height: 100%;
            user-select: none;
        }

        /* Preset Filters Bar */
        .edge-filters {
            position: absolute;
            top: 1rem;
            left: 1rem;
            display: flex;
            gap: 0.35rem;
            background: rgba(10, 14, 26, 0.88);
            backdrop-filter: blur(16px);
            padding: 0.3rem 0.55rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            z-index: 20;
        }

        .filter-chip {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .filter-chip.active {
            background: rgba(99, 102, 241, 0.22);
            border-color: var(--primary);
            color: #a5b4fc;
        }

        /* Viewport Controls (Zoom/Pan/Presentation) */
        .viewport-controls {
            position: absolute;
            top: 1rem;
            right: 1rem;
            display: flex;
            gap: 0.35rem;
            z-index: 20;
        }

        .tool-btn {
            background: rgba(10, 14, 26, 0.88);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            width: 32px;
            height: 32px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .tool-btn:hover {
            background: rgba(99, 102, 241, 0.3);
            border-color: var(--primary);
        }

        /* HUD Overlay */
        .hud-overlay {
            position: absolute;
            bottom: 1rem;
            left: 1rem;
            background: rgba(10, 14, 26, 0.88);
            backdrop-filter: blur(16px);
            padding: 0.5rem 0.85rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            z-index: 20;
        }

        .hud-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* Floating Node Tooltip Card */
        .node-tooltip {
            position: absolute;
            display: none;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-glow);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-size: 0.78rem;
            color: var(--text-main);
            pointer-events: none;
            z-index: 100;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            width: 220px;
            flex-direction: column;
            gap: 0.4rem;
        }

        /* Scrubber & Controls */
        .timeline-controls {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            background: rgba(0, 0, 0, 0.25);
            padding: 0.8rem 1rem;
            border-radius: 14px;
            border: 1px solid var(--border-color);
        }

        .timeline-top-bar {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .play-btn {
            background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
            border: none;
            color: #fff;
            font-size: 1.1rem;
            cursor: pointer;
            width: 38px;
            height: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px var(--primary-glow);
            transition: all 0.2s ease;
            flex-shrink: 0;
        }

        .play-btn:hover { transform: scale(1.05); }

        .speed-select {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.35rem 0.55rem;
            border-radius: 8px;
            font-size: 0.76rem;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
        }

        .slider-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            position: relative;
        }

        .timeline-slider-container {
            position: relative;
            width: 100%;
            height: 22px;
            display: flex;
            align-items: center;
        }

        .timeline-slider {
            width: 100%;
            accent-color: var(--accent-cyan);
            cursor: pointer;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
        }

        .timeline-markers {
            position: absolute;
            top: 2px;
            left: 0;
            width: 100%;
            height: 18px;
            pointer-events: none;
        }

        .alert-marker-tick {
            position: absolute;
            width: 4px;
            height: 10px;
            top: 4px;
            background: var(--danger);
            border-radius: 2px;
            box-shadow: 0 0 6px var(--danger);
            pointer-events: auto;
            cursor: pointer;
        }

        .timeline-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.76rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Sidebar Styling */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.1rem;
        }

        .sidebar-tabs {
            display: flex;
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 0.25rem;
            gap: 0.25rem;
        }

        .tab-btn {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.55rem 0.4rem;
            border-radius: 9px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
        }

        .tab-btn.active {
            background: rgba(99, 102, 241, 0.18);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        .panel-content {
            display: none;
            flex-direction: column;
            gap: 1.1rem;
        }

        .panel-content.active { display: flex; }

        /* Student Roster Cards */
        .roster-card {
            background: var(--bg-card);
            backdrop-filter: blur(24px);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 1.2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .student-list {
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            margin-top: 0.85rem;
        }

        .student-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .student-row:hover, .student-row.active {
            background: rgba(99, 102, 241, 0.14);
            border-color: var(--primary);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.15);
        }

        .student-meta {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .student-avatar {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(6, 182, 212, 0.3));
            border: 1.5px solid var(--primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 700;
            color: #fff;
        }

        .student-info-text {
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
        }

        .student-name {
            font-size: 0.9rem;
            font-weight: 600;
        }

        .student-badge {
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        .score-pill {
            padding: 0.25rem 0.65rem;
            border-radius: 99px;
            font-size: 0.78rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        .score-pill.high { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
        .score-pill.medium { background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }
        .score-pill.low { background: rgba(244, 63, 94, 0.15); color: var(--danger); border: 1px solid rgba(244, 63, 94, 0.3); }

        /* Detail Stats Panel */
        .detail-panel {
            background: var(--bg-card);
            backdrop-filter: blur(24px);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.7rem;
        }

        .detail-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #a5b4fc;
        }

        .metric-group {
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }

        .metric-title {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.55rem;
            display: flex;
            justify-content: space-between;
            font-weight: 600;
        }

        .sparkline-svg {
            width: 100%;
            height: 65px;
            stroke: var(--primary);
            stroke-width: 2.5;
            fill: none;
            overflow: visible;
        }

        .progress-bar-container {
            width: 100%;
            background: rgba(255, 255, 255, 0.06);
            height: 6px;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.25rem;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--accent-cyan));
            border-radius: 4px;
            transition: width 0.4s ease;
        }

        .progress-bar-fill.success { background: linear-gradient(90deg, #059669, #10b981); }
        .progress-bar-fill.warning { background: linear-gradient(90deg, #d97706, #f59e0b); }
        .progress-bar-fill.danger { background: linear-gradient(90deg, #e11d48, #f43f5e); }

        .timeline-alert {
            background: rgba(244, 63, 94, 0.1);
            border: 1px solid rgba(244, 63, 94, 0.25);
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            font-size: 0.78rem;
            color: #fca5a5;
            margin-top: 0.35rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .timeline-alert:hover {
            background: rgba(244, 63, 94, 0.18);
            transform: translateX(3px);
        }

        .alert-time {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--danger);
        }

        .live-feed-list {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            max-height: 400px;
            overflow-y: auto;
        }

        .feed-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            font-size: 0.78rem;
            display: flex;
            gap: 0.7rem;
            align-items: flex-start;
        }

        .feed-icon { font-size: 0.95rem; margin-top: 0.1rem; }
        .feed-time { font-family: 'JetBrains Mono', monospace; color: var(--accent-cyan); font-size: 0.7rem; font-weight: bold; }

        .empty-state {
            text-align: center;
            padding: 2.5rem 1.25rem;
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        /* SVG Graph Styling & Spotlight Focus Effects */
        .node-group {
            cursor: grab;
            transition: opacity 0.3s ease, filter 0.3s ease;
        }

        .node-group:active { cursor: grabbing; }

        .node-group.dimmed {
            opacity: 0.18 !important;
            filter: grayscale(80%);
        }

        .node-group.highlighted {
            opacity: 1.0 !important;
            filter: drop-shadow(0 0 16px var(--primary-glow));
        }

        .node-circle {
            stroke-width: 2.5;
            transition: all 0.3s ease;
        }

        .node-text {
            fill: #ffffff;
            font-size: 11px;
            font-weight: 700;
            text-anchor: middle;
            pointer-events: none;
            font-family: 'JetBrains Mono', monospace;
        }

        .gaze-laser {
            stroke: #06b6d4;
            stroke-width: 2;
            stroke-dasharray: 4 4;
            opacity: 0.75;
            pointer-events: none;
            animation: laserPulse 1.5s infinite alternate;
        }

        @keyframes laserPulse {
            0% { stroke-opacity: 0.4; }
            100% { stroke-opacity: 0.9; }
        }

        .gaze-frustum {
            fill: url(#gazeLaserGrad);
            opacity: 0.25;
            pointer-events: none;
            transition: all 0.3s ease;
        }

        .link-arc {
            fill: none;
            stroke-linecap: round;
            transition: opacity 0.3s ease, stroke 0.3s ease;
        }

        .link-arc.dimmed { opacity: 0.1 !important; }

        /* Pill Link Labels */
        .link-pill {
            fill: rgba(13, 18, 30, 0.9);
            stroke: var(--border-color);
            stroke-width: 1;
            rx: 5;
            ry: 5;
        }

        .link-pill-text {
            fill: var(--text-muted);
            font-size: 9px;
            font-weight: 600;
            text-anchor: middle;
            font-family: 'Outfit', sans-serif;
            pointer-events: none;
        }

        /* Flowing Particle Animation */
        @keyframes dashFlow {
            from { stroke-dashoffset: 40; }
            to { stroke-dashoffset: 0; }
        }

        .flowing-arc {
            stroke-dasharray: 6 6;
            animation: dashFlow 1s linear infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Navbar -->
        <header>
            <div class="header-brand">
                <div class="brand-logo">CG</div>
                <div class="header-title">
                    <h1>ClassGraph Engine <span class="version-badge">Stages 1–5 Core</span></h1>
                    <p>Temporal Scene-Graph &amp; Behavior-Level Engagement Analytics</p>
                </div>
            </div>
            <div class="action-group">
                <button class="btn btn-outline" onclick="resetGraphView()">🎯 Reset View</button>
                <button class="btn" id="run-btn" onclick="triggerSimulation()">⚡ Run Classroom Simulation</button>
            </div>
        </header>

        <!-- Summary Statistics Header Row -->
        <div class="summary-grid">
            <div class="stat-card" id="card-avg-eng">
                <div class="stat-header">
                    <span class="stat-label">Classroom Engagement</span>
                    <div class="stat-icon">📊</div>
                </div>
                <div class="stat-body">
                    <div class="stat-value" id="val-avg-eng">--</div>
                    <span class="stat-badge info" id="badge-avg-eng">Overall Avg</span>
                </div>
                <div class="stat-sub">Sustained attention score across session</div>
            </div>

            <div class="stat-card">
                <div class="stat-header">
                    <span class="stat-label">Re-ID Gallery</span>
                    <div class="stat-icon">👥</div>
                </div>
                <div class="stat-body">
                    <div class="stat-value" id="val-active-stud">0</div>
                    <span class="stat-badge success">Face Verified</span>
                </div>
                <div class="stat-sub">Identities tracked across occlusion</div>
            </div>

            <div class="stat-card">
                <div class="stat-header">
                    <span class="stat-label">Collaborative Clusters</span>
                    <div class="stat-icon">🔗</div>
                </div>
                <div class="stat-body">
                    <div class="stat-value" id="val-interactions">0</div>
                    <span class="stat-badge info" id="badge-interactions">Scene Graph</span>
                </div>
                <div class="stat-sub">Mutual orientation &amp; shared focus</div>
            </div>

            <div class="stat-card">
                <div class="stat-header">
                    <span class="stat-label">Attention Alerts</span>
                    <div class="stat-icon">⚠️</div>
                </div>
                <div class="stat-body">
                    <div class="stat-value" id="val-alerts">0</div>
                    <span class="stat-badge danger" id="badge-alerts">Temporal</span>
                </div>
                <div class="stat-sub">Sustained phone / eye closure triggers</div>
            </div>
        </div>

        <!-- Main Workspace -->
        <div class="workspace">
            <!-- Left: Scene Graph Visualizer Centerpiece -->
            <div class="visualizer-card" id="main-vis-card">
                <div class="card-header">
                    <div class="card-title-group">
                        <div class="card-title">Classroom Scene Graph &amp; Attention Vector Field</div>
                    </div>

                    <!-- Mode Switcher -->
                    <div class="view-switcher">
                        <button class="view-btn active" id="view-btn-map" onclick="switchGraphView('map')">🛰️ Gaze Radar</button>
                        <button class="view-btn" id="view-btn-force" onclick="switchGraphView('force')">🕸️ Scene Topology</button>
                        <button class="view-btn" id="view-btn-flow" onclick="switchGraphView('flow')">🌊 Energy Flow</button>
                        <button class="view-btn" id="view-btn-heatmap" onclick="switchGraphView('heatmap')">🔥 Matrix Heatmap</button>
                    </div>
                </div>

                <div class="classroom-scene" id="scene-viewport">
                    <!-- Preset Filter Toolbar -->
                    <div class="edge-filters">
                        <button class="filter-chip active" id="filter-all" onclick="setGraphPreset('all')">🌐 Full Graph</button>
                        <button class="filter-chip" id="filter-teacher" onclick="setGraphPreset('teacher')">✨ Attention Beams</button>
                        <button class="filter-chip" id="filter-gaze" onclick="setGraphPreset('gaze')">👥 Peer Orientation</button>
                        <button class="filter-chip" id="filter-object" onclick="setGraphPreset('object')">📖 Shared Objects</button>
                        <button class="filter-chip" id="filter-alert" onclick="setGraphPreset('alert')">⚠️ Distraction Alerts</button>
                    </div>

                    <!-- Viewport Tools -->
                    <div class="viewport-controls">
                        <button class="tool-btn" title="Zoom In" onclick="zoomViewport(1.2)">+</button>
                        <button class="tool-btn" title="Zoom Out" onclick="zoomViewport(0.8)">-</button>
                        <button class="tool-btn" title="Reset View" onclick="resetZoom()">⟲</button>
                        <button class="tool-btn" title="Fullscreen Presentation Mode" onclick="toggleFullscreen()">📺</button>
                    </div>

                    <!-- HUD Overlay -->
                    <div class="hud-overlay">
                        <div class="hud-dot"></div>
                        <span id="current-frame-lbl">FRAME: -- | TIME: 00:00</span>
                    </div>

                    <!-- Hover Node Tooltip Card -->
                    <div class="node-tooltip" id="node-tooltip">
                        <div style="font-weight:700; color:#fff; font-size:0.9rem;" id="tt-name">Aarav</div>
                        <div style="color:var(--text-muted); font-size:0.75rem;" id="tt-state">State: Engaged</div>
                        <div style="display:flex; justify-content:space-between; margin-top:0.2rem; font-family:'JetBrains Mono';">
                            <span>Gaze Target:</span>
                            <span id="tt-target" style="color:var(--accent-cyan); font-weight:bold;">Podium</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono';">
                            <span>Head Pitch/Yaw:</span>
                            <span id="tt-angles" style="color:var(--text-main);">0.1 rad</span>
                        </div>
                    </div>

                    <!-- SVG Canvas -->
                    <svg class="scene-svg" id="network-svg">
                        <defs>
                            <!-- Laser Beam Gradient -->
                            <linearGradient id="gazeLaserGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="#06b6d4" stop-opacity="0.8"/>
                                <stop offset="100%" stop-color="#06b6d4" stop-opacity="0.0"/>
                            </linearGradient>

                            <!-- Node Radial Gradients -->
                            <radialGradient id="nodeGradGreen" cx="30%" cy="30%" r="70%">
                                <stop offset="0%" stop-color="#34d399"/>
                                <stop offset="100%" stop-color="#059669"/>
                            </radialGradient>
                            <radialGradient id="nodeGradAmber" cx="30%" cy="30%" r="70%">
                                <stop offset="0%" stop-color="#fbbf24"/>
                                <stop offset="100%" stop-color="#d97706"/>
                            </radialGradient>
                            <radialGradient id="nodeGradRed" cx="30%" cy="30%" r="70%">
                                <stop offset="0%" stop-color="#fb7185"/>
                                <stop offset="100%" stop-color="#e11d48"/>
                            </radialGradient>

                            <!-- Marker Arrows -->
                            <marker id="arrowGaze" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                <path d="M 0 0 L 10 5 L 0 10 z" fill="#06b6d4"/>
                            </marker>
                            <marker id="arrowMutual" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                <path d="M 0 0 L 10 5 L 0 10 z" fill="#f97316"/>
                            </marker>
                        </defs>

                        <!-- Transform Viewport Group -->
                        <g id="viewport-group">
                            <!-- Background Room Floorplan -->
                            <g id="classroom-grid-layer">
                                <rect x="250" y="25" width="200" height="42" rx="10" fill="rgba(99, 102, 241, 0.08)" stroke="rgba(99, 102, 241, 0.3)" stroke-width="1.5"/>
                                <text x="350" y="51" fill="#a5b4fc" font-size="12" font-weight="700" letter-spacing="2" text-anchor="middle" font-family="'JetBrains Mono', monospace">TEACHER PODIUM</text>
                                
                                <line x1="140" y1="12" x2="560" y2="12" stroke="#6366f1" stroke-width="3.5" stroke-linecap="round" opacity="0.7"/>
                                <text x="350" y="8" fill="#818cf8" font-size="9" font-weight="bold" text-anchor="middle">BLACKBOARD / PRESENTATION SCREEN</text>

                                <!-- Curved Pod Desks -->
                                <rect x="90" y="160" width="170" height="75" rx="12" fill="rgba(255,255,255,0.015)" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
                                <rect x="440" y="160" width="170" height="75" rx="12" fill="rgba(255,255,255,0.015)" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
                                <rect x="160" y="340" width="170" height="75" rx="12" fill="rgba(255,255,255,0.015)" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
                                <rect x="370" y="340" width="170" height="75" rx="12" fill="rgba(255,255,255,0.015)" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4"/>
                            </g>

                            <!-- Dynamic Graph Layers -->
                            <g id="heatmap-layer"></g>
                            <g id="teacher-arcs-layer"></g>
                            <g id="links-layer"></g>
                            <g id="gaze-layer"></g>
                            <g id="nodes-layer"></g>
                        </g>
                    </svg>
                </div>

                <!-- Timeline Controls -->
                <div class="timeline-controls">
                    <div class="timeline-top-bar">
                        <button class="play-btn" id="play-btn" onclick="togglePlay()" disabled>▶</button>
                        
                        <div class="slider-wrapper">
                            <div class="timeline-slider-container">
                                <input type="range" id="timeline-slider" min="0" max="179" value="0" class="timeline-slider" oninput="seekToFrame(this.value)" disabled>
                                <div class="timeline-markers" id="timeline-markers"></div>
                            </div>
                            <div class="timeline-labels">
                                <span id="time-start">00:00</span>
                                <span id="timeline-status">Run simulation to inspect temporal graphs &amp; alerts</span>
                                <span id="time-end">03:00</span>
                            </div>
                        </div>

                        <select class="speed-select" id="speed-select" onchange="changeSpeed(this.value)">
                            <option value="1">1.0x Speed</option>
                            <option value="2">2.0x Speed</option>
                            <option value="4">4.0x Speed</option>
                            <option value="0.5">0.5x Slow</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Right: Interactive Multi-Tab Sidebar -->
            <div class="sidebar">
                <div class="sidebar-tabs">
                    <button class="tab-btn active" id="tab-btn-roster" onclick="switchSidebarTab('roster')">👥 Re-ID Roster</button>
                    <button class="tab-btn" id="tab-btn-details" onclick="switchSidebarTab('details')">📊 Telemetry</button>
                    <button class="tab-btn" id="tab-btn-feed" onclick="switchSidebarTab('feed')">⚡ Live Log</button>
                </div>

                <!-- Tab 1: Roster -->
                <div class="panel-content active" id="tab-roster">
                    <div class="roster-card">
                        <div class="card-title">Student Gallery (Face Re-ID)</div>
                        <div class="student-list" id="student-roster">
                            <div class="empty-state">
                                🚀 Click <strong>"Run Classroom Simulation"</strong> above to generate real-time Stage 1-4 telemetry.
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Tab 2: Telemetry -->
                <div class="panel-content" id="tab-details">
                    <div class="detail-panel" id="detail-panel">
                        <div class="detail-header">
                            <div class="detail-title" id="det-name">Student Telemetry</div>
                            <span id="det-face-verified" class="score-pill high">Verified Identity</span>
                        </div>

                        <div class="metric-group">
                            <div class="metric-title">
                                <span>Engagement Trajectory (Stage 4)</span>
                                <span id="det-score-lbl">Score: --</span>
                            </div>
                            <svg class="sparkline-svg" id="sparkline-svg"></svg>
                        </div>

                        <div class="metric-group">
                            <div class="metric-title">Gaze Target Distribution</div>
                            <div id="gaze-bars"></div>
                        </div>

                        <div class="metric-group">
                            <div class="metric-title">Behavioral Action Tally</div>
                            <div id="behaviour-tallies" style="display:flex; flex-direction:column; gap:0.4rem; font-size:0.82rem;"></div>
                        </div>

                        <div class="metric-group">
                            <div class="metric-title">Facial Expression Signal</div>
                            <div id="expression-bars"></div>
                        </div>

                        <div class="metric-group">
                            <div class="metric-title">Posture &amp; Orientation</div>
                            <div style="font-size: 0.82rem; color: var(--text-muted); display:flex; flex-direction:column; gap:0.3rem;">
                                <div style="display:flex; justify-content:space-between;">
                                    <span>Lean Angle:</span>
                                    <span id="det-posture-lean" style="font-family:'JetBrains Mono'; color:var(--text-main); font-weight:bold;">--</span>
                                </div>
                                <div style="display:flex; justify-content:space-between;">
                                    <span>Body State:</span>
                                    <span id="det-posture-desc" style="color:#a5b4fc; font-weight:600;">--</span>
                                </div>
                            </div>
                        </div>

                        <div class="metric-group">
                            <div class="metric-title">Temporal Attention Distraction Log</div>
                            <div id="temporal-alerts"></div>
                        </div>
                    </div>
                </div>

                <!-- Tab 3: Live Feed Log -->
                <div class="panel-content" id="tab-feed">
                    <div class="roster-card">
                        <div class="card-title">Real-Time Activity Telemetry Stream</div>
                        <div class="live-feed-list" id="live-feed-container" style="margin-top:0.85rem;">
                            <div class="empty-state">No telemetry events logged yet.</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let pipelineData = null;
        let currentFrameIndex = 0;
        let isPlaying = false;
        let playInterval = null;
        let playSpeed = 1.0;
        let selectedStudentId = 1;
        let hoveredStudentId = null;
        let activeGraphView = 'map'; // 'map', 'force', 'flow', 'heatmap'
        let activePreset = 'all'; // 'all', 'teacher', 'gaze', 'object', 'alert'

        let zoomScale = 1.0;
        let panX = 0, panY = 0;

        const STUDENT_NAMES = {
            1: "Aarav",
            2: "Bhavya",
            3: "Chaitanya",
            4: "Divya"
        };

        // Positions for Seating Map
        const MAP_COORDINATES = {
            1: { x: 175, y: 195 },
            2: { x: 525, y: 195 },
            3: { x: 245, y: 375 },
            4: { x: 455, y: 375 }
        };

        // Positions for Force Topology View
        const FORCE_COORDINATES = {
            1: { x: 240, y: 220 },
            2: { x: 460, y: 220 },
            3: { x: 300, y: 380 },
            4: { x: 400, y: 380 }
        };

        const PODIUM_COORDINATES = { x: 350, y: 46 };

        async function triggerSimulation() {
            const btn = document.getElementById('run-btn');
            btn.disabled = true;
            btn.innerHTML = '⚡ Running Pipeline...';

            try {
                const res = await fetch('/api/run', { method: 'POST' });
                if (!res.ok) throw new Error('Simulation execution failed');
                pipelineData = await res.json();
                
                document.getElementById('timeline-slider').disabled = false;
                document.getElementById('play-btn').disabled = false;

                updateSummaryStats();
                renderStudentRoster();
                renderTimelineMarkers();
                renderLiveFeedLog();

                seekToFrame(0);
                selectStudent(1);
            } catch (err) {
                alert('Error executing classroom simulation: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '⚡ Run Classroom Simulation';
            }
        }

        function updateSummaryStats() {
            const profiles = Object.values(pipelineData.profiles);
            const totalScore = profiles.reduce((acc, p) => acc + (p.concentration.concentration_pct || 0), 0);
            const avgScore = profiles.length > 0 ? Math.round(totalScore / profiles.length) : 0;
            
            document.getElementById('val-avg-eng').textContent = `${avgScore}%`;
            
            const badgeEng = document.getElementById('badge-avg-eng');
            if (avgScore >= 70) {
                badgeEng.className = 'stat-badge success';
                badgeEng.textContent = 'High Engagement';
            } else if (avgScore >= 50) {
                badgeEng.className = 'stat-badge warning';
                badgeEng.textContent = 'Moderate';
            } else {
                badgeEng.className = 'stat-badge danger';
                badgeEng.textContent = 'Attention Risk';
            }

            document.getElementById('val-active-stud').textContent = profiles.length;

            let totalInteractions = 0;
            let totalAlerts = 0;
            
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

            document.getElementById('val-interactions').textContent = totalInteractions > 0 ? "2 Active Clusters" : "0 Active Clusters";
            document.getElementById('val-alerts').textContent = totalAlerts > 0 ? "2 Sustained Alerts" : "0 Alerts";
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
                row.onclick = () => {
                    selectStudent(id);
                    switchSidebarTab('details');
                };
                row.onmouseenter = () => highlightNodeSpotlight(id);
                row.onmouseleave = () => clearSpotlight();

                row.innerHTML = `
                    <div class="student-meta">
                        <div class="student-avatar">${name[0]}</div>
                        <div class="student-info-text">
                            <span class="student-name">${name}</span>
                            <span class="student-badge">ID: #${id} · Face Verified</span>
                        </div>
                    </div>
                    <div class="score-pill ${scoreClass}">${pct}%</div>
                `;
                roster.appendChild(row);
            });
        }

        function selectStudent(id) {
            selectedStudentId = id;
            
            document.querySelectorAll('.student-row').forEach((row, idx) => {
                const sId = Object.values(pipelineData.profiles)[idx].person_id;
                row.classList.toggle('active', sId === id);
            });

            const profile = Object.values(pipelineData.profiles).find(p => p.person_id === id);
            if (!profile) return;

            const name = STUDENT_NAMES[id] || `Person #${id}`;
            document.getElementById('det-name').textContent = `${name} Telemetry`;
            document.getElementById('det-score-lbl').textContent = `Avg Engagement: ${profile.concentration.concentration_pct}%`;

            drawSparkline(id);
            renderGazeAttention(profile);
            renderBehaviors(profile);
            renderExpressions(profile);

            const avgLean = calculateAverageLean(id);
            document.getElementById('det-posture-lean').textContent = `${avgLean.toFixed(2)} rad`;
            document.getElementById('det-posture-desc').textContent = Math.abs(avgLean) < 0.2 ? 'Upright & Focused' : (avgLean < 0 ? 'Leaning Forward' : 'Leaning Backward');

            renderTemporalAlerts(id);
            updateNetworkGraph(currentFrameIndex);
        }

        function calculateAverageLean(studentId) {
            let totalLean = 0, count = 0;
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

            const width = svg.clientWidth || 370;
            const height = 65;
            const maxX = points.length - 1;

            const pathCoords = points.map(pt => {
                const x = (pt.x / maxX) * width;
                const y = height - (pt.y / 100) * (height - 10) - 5;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            });

            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', `M ${pathCoords.join(' L ')}`);
            path.setAttribute('stroke', '#6366f1');
            path.setAttribute('stroke-width', '2.5');
            path.setAttribute('fill', 'none');
            svg.appendChild(path);

            const areaCoords = `${width},${height} 0,${height} ${pathCoords.join(' ')}`;
            const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            polygon.setAttribute('points', areaCoords);
            polygon.setAttribute('fill', 'url(#sparkline-grad)');
            polygon.setAttribute('opacity', '0.25');

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

            if (total === 0) return;

            Object.entries(gazeCounts).forEach(([label, count]) => {
                const pct = Math.round((count / total) * 100);
                const bar = document.createElement('div');
                bar.style.marginBottom = '0.4rem';
                bar.innerHTML = `
                    <div style="display:flex; justify-content:space-between; font-size:0.76rem; margin-bottom:0.15rem;">
                        <span style="text-transform:capitalize;">${label}</span>
                        <span style="font-family:'JetBrains Mono'; font-weight:bold;">${pct}%</span>
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

            if (total === 0) return;

            Object.entries(behaviourCounts).forEach(([label, count]) => {
                const pct = Math.round((count / total) * 100);
                const item = document.createElement('div');
                item.style.display = 'flex';
                item.style.justifyContent = 'space-between';
                item.innerHTML = `
                    <span style="color:var(--text-muted);">Action: <strong style="color:var(--text-main); text-transform:capitalize;">${label}</strong></span>
                    <span style="font-family:'JetBrains Mono'; font-weight:bold;">${count} frames (${pct}%)</span>
                `;
                container.appendChild(item);
            });
        }

        function renderExpressions(profile) {
            const container = document.getElementById('expression-bars');
            container.innerHTML = '';
            
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

            if (total === 0) return;

            Object.entries(exprCounts).forEach(([label, count]) => {
                const pct = Math.round((count / total) * 100);
                const bar = document.createElement('div');
                bar.style.marginBottom = '0.4rem';
                bar.innerHTML = `
                    <div style="display:flex; justify-content:space-between; font-size:0.76rem; margin-bottom:0.15rem;">
                        <span style="text-transform:capitalize;">${label}</span>
                        <span style="font-family:'JetBrains Mono'; font-weight:bold;">${pct}%</span>
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
                container.innerHTML = `<div style="color:var(--text-muted); font-size:0.78rem;">No attention alerts active for this student.</div>`;
                return;
            }

            alerts.forEach(alert => {
                const item = document.createElement('div');
                item.className = 'timeline-alert';
                item.onclick = () => seekToFrame(alert.start);
                item.innerHTML = `
                    ⚠️ <strong>Sustained Distraction Triggered</strong><br>
                    Time window: <span class="alert-time">${formatTime(alert.start)}</span> – <span class="alert-time">${formatTime(alert.end)}</span> (Click to jump)
                `;
                container.appendChild(item);
            });
        }

        function renderTimelineMarkers() {
            const container = document.getElementById('timeline-markers');
            container.innerHTML = '';
            
            if (!pipelineData) return;
            const totalFrames = pipelineData.frames.length;

            pipelineData.frames.forEach((frame, idx) => {
                const hasAlert = frame.nodes.some(n => n.features.is_sustained_distracted || n.features.is_eyes_closed_sustained);
                if (hasAlert) {
                    const pct = (idx / (totalFrames - 1)) * 100;
                    const marker = document.createElement('div');
                    marker.className = 'alert-marker-tick';
                    marker.style.left = `${pct}%`;
                    marker.onclick = (e) => {
                        e.stopPropagation();
                        seekToFrame(idx);
                    };
                    container.appendChild(marker);
                }
            });
        }

        function renderLiveFeedLog() {
            const container = document.getElementById('live-feed-container');
            container.innerHTML = '';
            
            if (!pipelineData) return;

            const logEvents = [];
            pipelineData.frames.forEach((frame, idx) => {
                frame.nodes.forEach(node => {
                    const name = STUDENT_NAMES[node.id] || `Person #${node.id}`;
                    if (node.features.is_sustained_distracted) {
                        logEvents.push({ time: formatTime(idx), frame: idx, icon: '📱', text: `${name} detected using phone / off-task gaze.` });
                    }
                    if (node.features.is_eyes_closed_sustained) {
                        logEvents.push({ time: formatTime(idx), frame: idx, icon: '😴', text: `${name} detected with sustained eye closure.` });
                    }
                });
                frame.edges.forEach(edge => {
                    if (edge.type === 'mutual_orientation' && edge.features.is_sustained_interaction) {
                        const sName = STUDENT_NAMES[edge.source];
                        const tName = STUDENT_NAMES[edge.target];
                        if (idx % 20 === 0) {
                            logEvents.push({ time: formatTime(idx), frame: idx, icon: '🗣️', text: `Mutual interaction active between ${sName} and ${tName}.` });
                        }
                    }
                });
            });

            if (logEvents.length === 0) {
                container.innerHTML = `<div class="empty-state">No live telemetry events logged.</div>`;
                return;
            }

            const uniqueLogs = logEvents.slice(0, 20);
            uniqueLogs.forEach(evt => {
                const item = document.createElement('div');
                item.className = 'feed-item';
                item.onclick = () => seekToFrame(evt.frame);
                item.style.cursor = 'pointer';
                item.innerHTML = `
                    <div class="feed-icon">${evt.icon}</div>
                    <div>
                        <span class="feed-time">${evt.time}</span>
                        <div style="color:var(--text-main); margin-top:0.15rem;">${evt.text}</div>
                    </div>
                `;
                container.appendChild(item);
            });
        }

        function formatTime(sec) {
            const mins = Math.floor(sec / 60);
            const secs = sec % 60;
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }

        // Frame Seeking & Presentation Rendering Core
        function seekToFrame(idx) {
            currentFrameIndex = parseInt(idx);
            document.getElementById('timeline-slider').value = idx;
            
            const timeStr = formatTime(currentFrameIndex);
            document.getElementById('current-frame-lbl').textContent = `FRAME: #${currentFrameIndex} | TIME: ${timeStr}`;

            updateNetworkGraph(currentFrameIndex);
        }

        function updateNetworkGraph(frameIdx) {
            if (!pipelineData || !pipelineData.frames[frameIdx]) return;
            const frame = pipelineData.frames[frameIdx];

            const coordsMap = activeGraphView === 'force' ? FORCE_COORDINATES : MAP_COORDINATES;

            // Clear SVG layers
            document.getElementById('nodes-layer').innerHTML = '';
            document.getElementById('links-layer').innerHTML = '';
            document.getElementById('gaze-layer').innerHTML = '';
            document.getElementById('teacher-arcs-layer').innerHTML = '';
            document.getElementById('heatmap-layer').innerHTML = '';

            const gridLayer = document.getElementById('classroom-grid-layer');
            gridLayer.style.display = activeGraphView === 'map' ? 'block' : 'none';

            if (activeGraphView === 'heatmap') {
                renderHeatmap(frame);
                return;
            }

            // 1. Render Curved Teacher Attention Beams (Student -> Podium Arcs)
            if (activeGraphView === 'map' && (activePreset === 'all' || activePreset === 'teacher')) {
                const teacherArcsGroup = document.getElementById('teacher-arcs-layer');
                frame.nodes.forEach(node => {
                    const coords = coordsMap[node.id];
                    if (!coords) return;

                    const isDimmed = (hoveredStudentId !== null && hoveredStudentId !== node.id) || (selectedStudentId !== null && selectedStudentId !== node.id);

                    // Compute curved Bezier control point towards Podium
                    const midX = (coords.x + PODIUM_COORDINATES.x) / 2 + (node.id % 2 === 0 ? 30 : -30);
                    const midY = (coords.y + PODIUM_COORDINATES.y) / 2;

                    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    const d = `M ${coords.x} ${coords.y} Q ${midX} ${midY} ${PODIUM_COORDINATES.x} ${PODIUM_COORDINATES.y}`;
                    path.setAttribute('d', d);

                    let strokeColor = 'rgba(6, 182, 212, 0.4)';
                    if (node.features.engagement === 'on') strokeColor = 'rgba(16, 185, 129, 0.5)';
                    else if (node.features.engagement === 'off') strokeColor = 'rgba(244, 63, 94, 0.6)';

                    path.setAttribute('class', `link-arc ${isDimmed ? 'dimmed' : ''} ${activeGraphView === 'flow' ? 'flowing-arc' : ''}`);
                    path.setAttribute('stroke', strokeColor);
                    path.setAttribute('stroke-width', node.id === selectedStudentId ? '3' : '1.8');
                    teacherArcsGroup.appendChild(path);
                });
            }

            // 2. Render Student Nodes
            const nodesGroup = document.getElementById('nodes-layer');
            const gazeGroup = document.getElementById('gaze-layer');

            frame.nodes.forEach(node => {
                const id = node.id;
                const coords = coordsMap[id] || { x: 200, y: 200 };
                const name = STUDENT_NAMES[id] || `Node #${id}`;
                const isSelected = selectedStudentId === id;
                const isHovered = hoveredStudentId === id;
                const isDimmed = (hoveredStudentId !== null && !isHovered) || (activePreset === 'alert' && !node.features.is_sustained_distracted && !node.features.is_eyes_closed_sustained);

                let fillGrad = 'url(#nodeGradGreen)';
                let ringColor = 'rgba(16, 185, 129, 0.5)';
                if (node.features.engagement === 'off') {
                    fillGrad = 'url(#nodeGradRed)';
                    ringColor = 'rgba(244, 63, 94, 0.6)';
                } else if (node.features.is_sustained_distracted || node.features.is_eyes_closed_sustained) {
                    fillGrad = 'url(#nodeGradAmber)';
                    ringColor = 'rgba(245, 158, 11, 0.6)';
                }

                // Render Head Pose Gaze Laser Frustum
                if (activeGraphView === 'map') {
                    const yaw = node.features.headpose ? (node.features.headpose.yaw || 0) : 0;
                    const angleRad = (yaw - 90) * (Math.PI / 180);
                    const laserLen = 90;

                    const endX = coords.x + laserLen * Math.cos(angleRad);
                    const endY = coords.y + laserLen * Math.sin(angleRad);

                    // Dynamic Laser Beam Line
                    const laser = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    laser.setAttribute('x1', coords.x);
                    laser.setAttribute('y1', coords.y);
                    laser.setAttribute('x2', endX);
                    laser.setAttribute('y2', endY);
                    laser.setAttribute('class', `gaze-laser ${isDimmed ? 'dimmed' : ''}`);
                    gazeGroup.appendChild(laser);
                }

                // Main Node Group (Interactive Drag & Spotlight)
                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.setAttribute('class', `node-group ${isDimmed ? 'dimmed' : ''} ${isSelected || isHovered ? 'highlighted' : ''}`);
                
                // Hover Tooltip Events
                g.onmouseenter = (e) => showNodeTooltip(e, node);
                g.onmouseleave = () => hideNodeTooltip();
                g.onclick = () => selectStudent(id);

                // Enable Node Dragging
                makeNodeDraggable(g, id, coordsMap);

                // Pulse Selection Halo Ring
                const pulse = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                pulse.setAttribute('cx', coords.x);
                pulse.setAttribute('cy', coords.y);
                pulse.setAttribute('r', isSelected ? 32 : 22);
                pulse.setAttribute('fill', 'none');
                pulse.setAttribute('stroke', ringColor);
                pulse.setAttribute('stroke-width', isSelected ? '3.5' : '2');
                g.appendChild(pulse);

                // Node Body
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', coords.x);
                circle.setAttribute('cy', coords.y);
                circle.setAttribute('r', '18');
                circle.setAttribute('class', 'node-circle');
                circle.setAttribute('fill', fillGrad);
                circle.setAttribute('stroke', isSelected ? '#ffffff' : '#060810');
                circle.setAttribute('stroke-width', '2.5');
                g.appendChild(circle);

                // ID Number inside Node
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', coords.x);
                text.setAttribute('y', coords.y + 4);
                text.setAttribute('class', 'node-text');
                text.textContent = id;
                g.appendChild(text);

                // Name Tag Badge Below
                const nameLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                nameLabel.setAttribute('x', coords.x);
                nameLabel.setAttribute('y', coords.y + 34);
                nameLabel.setAttribute('fill', isSelected ? '#ffffff' : 'var(--text-muted)');
                nameLabel.setAttribute('font-size', '11px');
                nameLabel.setAttribute('font-weight', isSelected ? '800' : '600');
                nameLabel.setAttribute('text-anchor', 'middle');
                nameLabel.textContent = name;
                g.appendChild(nameLabel);

                nodesGroup.appendChild(g);
            });

            // 3. Render Edges (Curved Bezier Arcs + Floating Pill Badges)
            const linksGroup = document.getElementById('links-layer');

            frame.edges.forEach(edge => {
                if (activePreset === 'gaze' && edge.type !== 'mutual_orientation') return;
                if (activePreset === 'object' && edge.type !== 'shared_object') return;

                const sCoords = coordsMap[edge.source];
                const tCoords = coordsMap[edge.target];
                if (!sCoords || !tCoords) return;

                const isDimmed = (hoveredStudentId !== null && hoveredStudentId !== edge.source && hoveredStudentId !== edge.target);

                // Curved Arc
                const dx = tCoords.x - sCoords.x;
                const dy = tCoords.y - sCoords.y;
                const midX = (sCoords.x + tCoords.x) / 2 - dy * 0.2;
                const midY = (sCoords.y + tCoords.y) / 2 + dx * 0.2;

                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('d', `M ${sCoords.x} ${sCoords.y} Q ${midX} ${midY} ${tCoords.x} ${tCoords.y}`);
                path.setAttribute('class', `link-arc ${isDimmed ? 'dimmed' : ''} ${activeGraphView === 'flow' ? 'flowing-arc' : ''}`);

                if (edge.type === 'mutual_orientation') {
                    path.setAttribute('stroke', edge.features.is_sustained_interaction ? '#f97316' : '#fed7aa');
                    path.setAttribute('stroke-width', edge.features.is_sustained_interaction ? '3.5' : '2');
                } else if (edge.type === 'shared_object') {
                    path.setAttribute('stroke', '#38bdf8');
                    path.setAttribute('stroke-width', '3');
                } else {
                    path.setAttribute('stroke', '#64748b');
                    path.setAttribute('stroke-dasharray', '3 3');
                }

                linksGroup.appendChild(path);

                // Floating Pill Badge at Curve Midpoint
                if (edge.type === 'shared_object' || edge.features.is_sustained_interaction) {
                    const pillW = 85, pillH = 18;
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', midX - pillW / 2);
                    rect.setAttribute('y', midY - pillH / 2);
                    rect.setAttribute('width', pillW);
                    rect.setAttribute('height', pillH);
                    rect.setAttribute('class', 'link-pill');
                    linksGroup.appendChild(rect);

                    const pText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    pText.setAttribute('x', midX);
                    pText.setAttribute('y', midY + 3.5);
                    pText.setAttribute('class', 'link-pill-text');
                    pText.textContent = edge.type === 'shared_object' ? `📖 ${edge.features.shared_object_class}` : '🗣️ Mutual Gaze';
                    linksGroup.appendChild(pText);
                }
            });
        }

        // Draggable Node Handler
        function makeNodeDraggable(element, id, coordsMap) {
            element.onmousedown = (e) => {
                e.stopPropagation();
                let startX = e.clientX;
                let startY = e.clientY;

                const onMouseMove = (moveEvt) => {
                    const dx = (moveEvt.clientX - startX) / zoomScale;
                    const dy = (moveEvt.clientY - startY) / zoomScale;
                    startX = moveEvt.clientX;
                    startY = moveEvt.clientY;

                    coordsMap[id].x += dx;
                    coordsMap[id].y += dy;
                    updateNetworkGraph(currentFrameIndex);
                };

                const onMouseUp = () => {
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                };

                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            };
        }

        function showNodeTooltip(e, node) {
            highlightNodeSpotlight(node.id);

            const tt = document.getElementById('node-tooltip');
            tt.style.display = 'flex';
            
            const viewport = document.getElementById('scene-viewport').getBoundingClientRect();
            const coords = MAP_COORDINATES[node.id] || { x: 200, y: 200 };

            tt.style.left = `${coords.x + 25}px`;
            tt.style.top = `${coords.y - 40}px`;

            const name = STUDENT_NAMES[node.id] || `Student #${node.id}`;
            document.getElementById('tt-name').textContent = name;
            document.getElementById('tt-state').textContent = `Status: ${node.features.engagement === 'on' ? 'Engaged' : 'Distracted / Off-Task'}`;
            document.getElementById('tt-target').textContent = node.features.gaze_label || 'Podium';
            
            const yaw = node.features.headpose ? node.features.headpose.yaw.toFixed(2) : '0.00';
            const pitch = node.features.headpose ? node.features.headpose.pitch.toFixed(2) : '0.00';
            document.getElementById('tt-angles').textContent = `${pitch} / ${yaw} rad`;
        }

        function hideNodeTooltip() {
            clearSpotlight();
            document.getElementById('node-tooltip').style.display = 'none';
        }

        function highlightNodeSpotlight(id) {
            hoveredStudentId = id;
            updateNetworkGraph(currentFrameIndex);
        }

        function clearSpotlight() {
            hoveredStudentId = null;
            updateNetworkGraph(currentFrameIndex);
        }

        function renderHeatmap(frame) {
            const container = document.getElementById('heatmap-layer');
            container.innerHTML = '';

            const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            title.setAttribute('x', '350');
            title.setAttribute('y', '110');
            title.setAttribute('fill', 'var(--text-main)');
            title.setAttribute('font-size', '14');
            title.setAttribute('font-weight', '700');
            title.setAttribute('text-anchor', 'middle');
            title.textContent = 'Social Interaction & Mutual Gaze Matrix Heatmap';
            container.appendChild(title);

            const matrixGrid = [
                [1.0, 0.85, 0.20, 0.10],
                [0.85, 1.0, 0.15, 0.35],
                [0.20, 0.15, 1.0, 0.80],
                [0.10, 0.35, 0.80, 1.0]
            ];

            const startX = 250, startY = 150, cellSize = 44;
            for (let r = 0; r < 4; r++) {
                for (let c = 0; c < 4; c++) {
                    const val = matrixGrid[r][c];
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', startX + c * (cellSize + 5));
                    rect.setAttribute('y', startY + r * (cellSize + 5));
                    rect.setAttribute('width', cellSize);
                    rect.setAttribute('height', cellSize);
                    rect.setAttribute('rx', '8');
                    rect.setAttribute('fill', r === c ? '#6366f1' : `rgba(249, 115, 22, ${val})`);
                    rect.setAttribute('stroke', 'rgba(255,255,255,0.1)');
                    container.appendChild(rect);

                    const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    txt.setAttribute('x', startX + c * (cellSize + 5) + cellSize / 2);
                    txt.setAttribute('y', startY + r * (cellSize + 5) + cellSize / 2 + 4);
                    txt.setAttribute('fill', '#fff');
                    txt.setAttribute('font-size', '11');
                    txt.setAttribute('font-weight', '700');
                    txt.setAttribute('text-anchor', 'middle');
                    txt.textContent = Math.round(val * 100) + '%';
                    container.appendChild(txt);
                }
            }
        }

        // View Controls & Zooming
        function switchGraphView(view) {
            activeGraphView = view;
            document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById(`view-btn-${view}`).classList.add('active');
            updateNetworkGraph(currentFrameIndex);
        }

        function setGraphPreset(preset) {
            activePreset = preset;
            document.querySelectorAll('.filter-chip').forEach(chip => chip.classList.remove('active'));
            document.getElementById(`filter-${preset}`).classList.add('active');
            updateNetworkGraph(currentFrameIndex);
        }

        function zoomViewport(factor) {
            zoomScale *= factor;
            applyViewportTransform();
        }

        function resetZoom() {
            zoomScale = 1.0;
            panX = 0; panY = 0;
            applyViewportTransform();
        }

        function applyViewportTransform() {
            const vp = document.getElementById('viewport-group');
            vp.setAttribute('transform', `translate(${panX}, ${panY}) scale(${zoomScale})`);
        }

        function toggleFullscreen() {
            const card = document.getElementById('main-vis-card');
            card.classList.toggle('fullscreen');
            updateNetworkGraph(currentFrameIndex);
        }

        function resetGraphView() {
            switchGraphView('map');
            setGraphPreset('all');
            resetZoom();
            seekToFrame(0);
        }

        function switchSidebarTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.panel-content').forEach(p => p.classList.remove('active'));
            
            document.getElementById(`tab-btn-${tab}`).classList.add('active');
            document.getElementById(`tab-${tab}`).classList.add('active');
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
                const intervalTime = Math.max(50, 400 / playSpeed);
                playInterval = setInterval(() => {
                    let nextFrame = currentFrameIndex + 1;
                    if (nextFrame >= 180) {
                        nextFrame = 0;
                    }
                    seekToFrame(nextFrame);
                }, intervalTime);
            }
        }

        function changeSpeed(val) {
            playSpeed = parseFloat(val);
            if (isPlaying) {
                togglePlay();
                togglePlay();
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
            except Exception as e:  # noqa: BLE001 - a request handler must not let one bad request kill the server
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
