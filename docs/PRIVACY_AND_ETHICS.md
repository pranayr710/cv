# Privacy, Regulatory & Ethical Framework

This document outlines the ethical design choices, regulatory alignment, and privacy-preservation mechanics of the ClassGraph system. 

---

## 1. Framing: Behavior/Posture Classification vs. Emotion Inference
ClassGraph is designed strictly to classify observable physical behavior, body posture, and facial configuration (e.g., eye closure, gaze direction, head orientation). It deliberately **excludes** and avoids inferring internal psychological states, feelings, or emotions.

### EU AI Act Alignment
* **Regulation**: Article 5(1)(f) of the European Union Artificial Intelligence (AI) Act (in force since February 2025) prohibits the use of AI systems that detect emotions in educational institutions.
* **Rationale**: The European Union's prohibition is grounded in a deep scientific consensus (e.g., Barrett et al., 2019) that facial movements do not map reliably to internal emotional states (a smile can signal submission, politeness, or discomfort as often as happiness). Furthermore, emotion inference carries significant risks of bias, lack of generalizability, and student harm.
* **Implementation Choice**: Although ClassGraph is not currently deployed in the EU, it cites and adopts this standard globally. All output fields, codebases, and documentation are framed strictly around **facial expression classification** (e.g., "happy", "sad", "neutral" facial configurations) and **behavioral proxies** rather than emotion recognition.

---

## 2. Child Data Protection & India's DPDP Act 2023
Classroom video recordings capture minor children, which places ClassGraph under the highest level of regulatory scrutiny.

### DPDP Act 2023 Compliance
* **Regulation**: The Digital Personal Data Protection (DPDP) Act of India (2023) enforces strict purpose limitation, parental/guardian consent, and enhanced protection for child data. 
* **Data Re-identifiability Risk**: Raw video footage of children constitutes sensitive personal data and carries inherent re-identifiability risk, regardless of whether the pipeline extracts anonymous indices.

### Concrete Retention & Deletion Policy
* **Raw Video**: Raw input video files must be deleted immediately after local feature extraction completes. They must never be persisted on long-term storage or transmitted over external networks.
* **Derived Records**: JSONL logs and classroom reports are anonymous. However, if any raw bounding-box coordinate data is deemed sensitive, derived records must be purged within 30 days of generation.

---

## 3. Anonymization Guarantee & The Virginia Tech Precedent
ClassGraph features built-in privacy protection designed directly into the code and verified via automated regression tests.

### Technical Safeguards
* **Session-Scoped IDs**: All track IDs and person IDs are generated dynamically per session/video. There is no central database of student faces, and no face embeddings are persisted on disk.
* **Identity Leakage Prevention**: We enforce that identity cannot leak across sessions. Two dedicated regression tests verify that if a tracker is reused without an explicit reset, or across separate runs, person identities are completely fresh and non-reidentifiable.
* **Precedent**: This matches the Virginia Tech classroom-analytics system (arXiv:2604.03401), which discards raw video immediately after feature extraction to comply with the Family Educational Rights and Privacy Act (FERPA).
* **Project-wide Raw Crop Discarding**: ClassGraph adopts a strict policy of discarding all raw image crops of faces/bodies immediately after computing features (embeddings, facial expression, posture). Only derived alphanumeric measurements and bounding box positions are outputted in the stage JSONL.

---

## 4. Access Control & Parental Consent Recommendations
To prevent surveillance abuse and minimize harm, the deployment of ClassGraph must adhere to the following governance recommendations:

* **Teacher-Only Dashboard Access**: Class-level aggregates and individual drill-downs must be visible only to the classroom instructor. Student profiles must never be displayed publicly or used to shame or penalize students.
* **Guardian Consent & Opt-Out**: Schools deploying the system must obtain explicit, informed consent from parents or guardians prior to processing any classroom video. An opt-out mechanism must be provided (e.g., placing opt-out students in video-masked regions or excluding their bounding boxes from downstream analysis).
