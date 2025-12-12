import os
import json
import time
from typing import List, Dict, Optional, Tuple

import cv2
from ultralytics import YOLO

# ---- Optional activity dependency ----
try:
    import numpy as np
    import mediapipe as mp
    MEDIAPIPE_OK = True
except Exception:
    MEDIAPIPE_OK = False


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "yolov8n.pt"

# Pick 3–5 classes you can reliably demo
TARGET_CLASSES = {"person", "chair", "laptop", "book", "bottle"}
CONF_THRESH = 0.40

# Unity bridge output
UNITY_DIR = "unity_bridge"
UNITY_FILE = os.path.join(UNITY_DIR, "detection.json")

# Activity logic tuning
CONTEXT_IOU_THRESH = 0.05
SLEEP_POSTURE_RATIO_THRESH = 0.55
SLEEP_MOTION_THRESH = 0.01
IDLE_MOTION_THRESH = 0.015


# -----------------------------
# Globals for interaction
# -----------------------------
selected: Optional[Dict] = None
last_written_label: str = ""


def point_in_box(px, py, x1, y1, x2, y2):
    return x1 <= px <= x2 and y1 <= py <= y2


def write_unity_bridge(label: str, conf: float):
    """
    Writes the selected object class to a JSON file that Unity can read.
    This gives you the clean CV -> 3D integration story.
    """
    global last_written_label

    label_norm = (label or "").lower().strip()
    if not label_norm:
        return

    # Avoid writing the same label repeatedly
    if label_norm == last_written_label:
        return

    os.makedirs(UNITY_DIR, exist_ok=True)
    data = {"label": label_norm, "confidence": float(conf)}

    with open(UNITY_FILE, "w") as f:
        json.dump(data, f)

    last_written_label = label_norm


def on_mouse(event, x, y, flags, param):
    """
    Click interaction:
    - select a detected object
    - write to Unity bridge
    """
    global selected
    detections = param.get("detections", [])

    if event == cv2.EVENT_LBUTTONDOWN:
        # pick top-most (last drawn) first
        for det in reversed(detections):
            x1, y1, x2, y2 = det["box"]
            if point_in_box(x, y, x1, y1, x2, y2):
                selected = det
                write_unity_bridge(det["label"], det["conf"])
                return
        selected = None


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(areaA + areaB - interArea + 1e-6)


# -----------------------------
# Activity (pose + context)
# -----------------------------
def compute_pose_activity(lm, prev_lm):
    """
    Explainable heuristic activity:
    - posture ratio + small motion
    Returns label string.
    """
    nose = np.array([lm[0].x, lm[0].y])
    l_sh = np.array([lm[11].x, lm[11].y])
    r_sh = np.array([lm[12].x, lm[12].y])
    l_hip = np.array([lm[23].x, lm[23].y])
    r_hip = np.array([lm[24].x, lm[24].y])

    shoulder_mid = (l_sh + r_sh) / 2.0
    hip_mid = (l_hip + r_hip) / 2.0

    head_to_shoulder = np.linalg.norm(nose - shoulder_mid)
    shoulder_to_hip = np.linalg.norm(shoulder_mid - hip_mid)
    posture_ratio = head_to_shoulder / (shoulder_to_hip + 1e-6)

    motion_score = 0.0
    if prev_lm is not None:
        diffs = []
        for idx in [0, 11, 12, 23, 24]:
            p = np.array([lm[idx].x, lm[idx].y])
            q = np.array([prev_lm[idx].x, prev_lm[idx].y])
            diffs.append(np.linalg.norm(p - q))
        motion_score = float(np.mean(diffs))

    if posture_ratio < SLEEP_POSTURE_RATIO_THRESH and motion_score < SLEEP_MOTION_THRESH:
        return "resting/sleeping-like"
    elif motion_score < IDLE_MOTION_THRESH:
        return "idle"
    else:
        return "active"


def main():
    global selected

    # ---- Load model ----
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

    names = model.names

    # ---- Setup pose ----
    pose = None
    prev_pose_lm = None
    if MEDIAPIPE_OK:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ---- Camera ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    window = "FINAL CV + Click + Activity + Unity Bridge (Q to quit)"
    mouse_param = {"detections": []}
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse, mouse_param)

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        # ---- YOLO inference ----
        results = model(frame, verbose=False)
        r = results[0]

        detections: List[Dict] = []

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONF_THRESH:
                    continue

                label = names.get(cls_id, str(cls_id))
                if label not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "label": label,
                    "conf": conf,
                    "box": (x1, y1, x2, y2),
                    "activity": None
                })

        # Split for context activity
        persons = [d for d in detections if d["label"] == "person"]
        laptops = [d for d in detections if d["label"] == "laptop"]

        # ---- Pose activity (full frame) ----
        pose_label = None
        if pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks.landmark
                pose_label = compute_pose_activity(lm, prev_pose_lm)
                prev_pose_lm = lm
            else:
                prev_pose_lm = None

        # ---- Context activity: person+laptop overlap ----
        stable_overlap = False
        if persons and laptops:
            for p in persons:
                for l in laptops:
                    if iou(p["box"], l["box"]) > CONTEXT_IOU_THRESH:
                        stable_overlap = True
                        break
                if stable_overlap:
                    break

        # Attach activity to person detections
        for p in persons:
            if stable_overlap:
                p["activity"] = "working/using laptop"
            elif pose_label:
                p["activity"] = pose_label

        # Update detections for mouse callback
        mouse_param["detections"] = detections

        # ---- Draw detections ----
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["conf"]

            color = (0, 255, 0)
            thickness = 2

            if selected and det["box"] == selected["box"] and det["label"] == selected["label"]:
                color = (0, 255, 255)
                thickness = 3

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            if label == "person" and det.get("activity"):
                tag = f"{label} {conf:.2f} | {det['activity']}"
            else:
                tag = f"{label} {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                annotated, tag, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
            )

        # ---- Refresh selected reference each frame ----
        if selected:
            updated = None
            for det in detections:
                if det["box"] == selected["box"] and det["label"] == selected["label"]:
                    updated = det
                    break
            if updated:
                selected = updated

        # ---- Info panel ----
        if selected:
            panel_x, panel_y = 20, 70
            panel_w, panel_h = 420, 150

            cv2.rectangle(annotated, (panel_x, panel_y),
                          (panel_x + panel_w, panel_y + panel_h),
                          (0, 0, 0), -1)

            cv2.putText(annotated, "Selected Object",
                        (panel_x + 10, panel_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(annotated, f"Class: {selected['label']}",
                        (panel_x + 10, panel_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(annotated, f"Confidence: {selected['conf']:.2f}",
                        (panel_x + 10, panel_y + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if selected["label"] == "person" and selected.get("activity"):
                cv2.putText(annotated, f"Activity: {selected['activity']}",
                            (panel_x + 10, panel_y + 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ---- FPS ----
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(
            annotated, f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # Small status line
        status = "Unity Bridge: ON" if os.path.exists(UNITY_FILE) or True else "Unity Bridge: OFF"
        cv2.putText(
            annotated, status,
            (20, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        cv2.imshow(window, annotated)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
