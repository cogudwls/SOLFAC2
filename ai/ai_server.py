import cv2
import math
import joblib
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import Counter, deque
from threading import Lock
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).resolve().parent / "SOLFAC_AI_Runtime"
POSE_MODEL = BASE / "pose_landmarker_lite.task"
AI_MODEL = BASE / "models" / "elbow_model.pkl"
START_ANGLE = 160
BEND_ANGLE = 150
RETURN_ANGLE = 155

WINDOW = 5
STABLE_FRAMES = 2

# 짧은 관절 인식 실패 허용
SHORT_GAP = 5

# AI 예측을 채택할 최소 확신도
MIN_CONFIDENCE = 0.55

# 동작 중 유효한 자세 예측 최소 수
MIN_VOTES = 5

# 오류 라벨 비율 기준 (임시)
ERROR_RATIO = 0.20

# 운동 기준서 ELB-A / ELB-B 보조 규칙 (시연용 초기값, 실측 조정 필요)
ELBOW_TARGET_ANGLE = 100.0
ELBOW_UPPER_ARM_LIMIT = 35.0
ELBOW_RULE_MIN_FRAMES = 5


def elbow_upper_arm_angle(landmarks, side, width, height):
    # 어깨→팔꿈치와 어깨→골반 사이 각도: 몸통에 붙이면 0도에 가까움.
    ids = (11, 13, 23) if side == "left" else (12, 14, 24)
    pts = [landmarks[i] for i in ids]
    if any(p.visibility < 0.5 or not math.isfinite(p.x) or not math.isfinite(p.y) for p in pts):
        return np.nan
    shoulder, elbow, hip = [np.array([p.x * width, p.y * height]) for p in pts]
    arm, torso = elbow - shoulder, hip - shoulder
    denom = np.linalg.norm(arm) * np.linalg.norm(torso)
    if denom < 1e-8:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(arm, torso) / denom, -1, 1))))


FEATURE_COLUMNS = []

for i in range(33):
    FEATURE_COLUMNS.extend([
        f"p{i}_x",
        f"p{i}_y",
        f"p{i}_visibility"
    ])

def make_features(landmarks):
    # 골반 중심
    hip_x = (
        landmarks[23].x + landmarks[24].x
    ) / 2

    hip_y = (
        landmarks[23].y + landmarks[24].y
    ) / 2

    # 양어깨 간 거리
    shoulder_width = math.hypot(
        landmarks[11].x - landmarks[12].x,
        landmarks[11].y - landmarks[12].y
    )

    if shoulder_width < 1e-6:
        return None

    features = []

    for p in landmarks:
        features.extend([
            (p.x - hip_x) / shoulder_width,
            (p.y - hip_y) / shoulder_width,
            p.visibility
        ])

    return features

# ==========================================
# 5. 관절 각도
# ==========================================

def get_angle(landmarks, ids):
    points = []

    for i in ids:
        p = landmarks[i]

        if p.visibility < 0.5:
            return np.nan

        if not (
            math.isfinite(p.x)
            and math.isfinite(p.y)
        ):
            return np.nan

        points.append(
            np.array([p.x, p.y])
        )

    a, b, c = points

    v1 = a - b
    v2 = c - b

    denom = (
        np.linalg.norm(v1)
        * np.linalg.norm(v2)
    )

    if denom < 1e-8:
        return np.nan

    cosine = np.clip(
        np.dot(v1, v2) / denom,
        -1.0, 1.0
    )

    return float(
        math.degrees(math.acos(cosine))
    )

# 손목의 어깨 기준 상대 높이
def get_wrist_height(landmarks, shoulder_id, wrist_id):
    shoulder = landmarks[shoulder_id]
    wrist = landmarks[wrist_id]

    if shoulder.visibility < 0.5 or wrist.visibility < 0.5:
        return np.nan

    ls = landmarks[11]
    rs = landmarks[12]

    width = math.hypot(ls.x - rs.x, ls.y - rs.y)

    if width < 1e-6:
        return np.nan

    return float((shoulder.y - wrist.y) / width)

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(POSE_MODEL)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)


def analyze_frame(frame, pose, model, frame_number, timestamp_ms):
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = timestamp_ms

    detected = pose.detect_for_video(
        image,
        timestamp
    )

    result = {
        "frame": frame_number,
        "time_sec": round(
            timestamp_ms / 1000, 3
        ),
        "left_angle": np.nan,
        "left_wrist_height": np.nan,
        "right_wrist_height": np.nan,
        "right_angle": np.nan,
        "left_upper_arm_angle": np.nan,
        "right_upper_arm_angle": np.nan,
        "pose_label": "NO_POSE",
        "confidence": 0.0, "landmarks": []
    }

    if detected.pose_landmarks:
        landmarks = detected.pose_landmarks[0]
        result["landmarks"] = serialize_points(landmarks)
        height, width = frame.shape[:2]
        for side in ("left", "right"):
            result[f"{side}_upper_arm_angle"] = elbow_upper_arm_angle(landmarks, side, width, height)

        result["left_angle"] = get_angle(
            landmarks,
            [11, 13, 15]
        )

        result["right_angle"] = get_angle(
            landmarks,
            [12, 14, 16]
        )
        result["left_wrist_height"] = get_wrist_height(
            landmarks, 11, 15
        )

        result["right_wrist_height"] = get_wrist_height(
            landmarks, 12, 16
        )

        features = make_features(landmarks)

        if features is not None:
            X = np.asarray([features], dtype=np.float64)

            prediction = model.predict(X)[0]

            result["pose_label"] = str(
                prediction
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)[0]

                result["confidence"] = float(
                    np.max(probabilities)
                )
            else:
                result["confidence"] = 1.0

    return result



def summarize(frame_results):
    def detect_reps(side):
        key = f"{side}_angle"

        history = deque(maxlen=WINDOW)

        state = "WAIT"

        ready_count = 0
        return_count = 0

        missing = 0
        max_missing_in_rep = 0

        start = None
        cycles = []

        for row in frame_results:
            frame = row["frame"]
            raw = row[key]

            if not np.isfinite(raw):
                missing += 1

                if state == "MOVING":
                    max_missing_in_rep = max(
                        max_missing_in_rep,
                        missing
                    )

                history.clear()
                return_count = 0
                continue

            missing = 0

            history.append(raw)

            angle = float(
                np.median(history)
            )

            if state == "WAIT":
                if angle >= START_ANGLE:
                    ready_count += 1
                else:
                    ready_count = 0

                if ready_count >= STABLE_FRAMES:
                    state = "READY"

            elif state == "READY":
                if angle < BEND_ANGLE:
                    state = "MOVING"

                    start = frame
                    max_missing_in_rep = 0
                    return_count = 0

            elif state == "MOVING":
                if angle >= RETURN_ANGLE:
                    return_count += 1
                else:
                    return_count = 0

                if return_count >= STABLE_FRAMES:
                    if max_missing_in_rep <= SHORT_GAP:
                        status = "CONFIRMED"
                    else:
                        status = "REVIEW"

                    cycles.append({
                        "start": start,
                        "end": frame,
                        "status": status,
                        "max_missing": max_missing_in_rep
                    })

                    state = "WAIT"
                    ready_count = 0
                    return_count = 0

        return cycles

    # 손목 상승 → 복귀 운동 시도 감지
    def detect_wrist_reps(side):
        key = f"{side}_wrist_height"

        heights = np.array([
            row[key] for row in frame_results
        ], dtype=float)

        smoothed = np.full(len(heights), np.nan)

        for i in range(len(heights)):
            window = heights[max(0, i - 4):i + 1]
            valid = window[np.isfinite(window)]

            if np.isfinite(heights[i]) and len(valid) >= 3:
                smoothed[i] = float(np.median(valid))

        initial = smoothed[:10]
        initial = initial[np.isfinite(initial)]

        if len(initial) < 3:
            return []

        baseline = float(np.median(initial))

        state = "DOWN"
        rise_count = 0
        return_count = 0
        start = None
        cycles = []

        for i, value in enumerate(smoothed):
            if not np.isfinite(value):
                rise_count = 0
                return_count = 0
                continue

            rise = value - baseline

            if state == "DOWN":
                rise_count = (
                    rise_count + 1 if rise >= 0.30 else 0
                )

                if rise_count >= 3:
                    start = frame_results[i - 2]["frame"]
                    state = "UP"

            elif state == "UP":
                return_count = (
                    return_count + 1 if rise <= 0.12 else 0
                )

                if return_count >= 3:
                    cycles.append({
                        "start": start,
                        "end": frame_results[i]["frame"],
                        "status": "CONFIRMED",
                        "max_missing": 0
                    })
                    state = "DOWN"
                    rise_count = 0

        return cycles



    # 팔꿈치 각도 기반 감지
    left_angle_cycles = detect_reps("left")
    right_angle_cycles = detect_reps("right")

    # 손목 움직임 기반 감지
    left_wrist_cycles = detect_wrist_reps("left")
    right_wrist_cycles = detect_wrist_reps("right")

    # 같은 운동을 중복 집계하지 않도록
    # 우선 각도 방식, 없을 때 손목 방식을 사용
    left_cycles = (
        left_angle_cycles if left_angle_cycles
        else left_wrist_cycles
    )

    right_cycles = (
        right_angle_cycles if right_angle_cycles
        else right_wrist_cycles
    )

    print("\n[감지 방식별 결과]")
    print("왼팔 각도:", len(left_angle_cycles))
    print("오른팔 각도:", len(right_angle_cycles))
    print("왼손 손목:", len(left_wrist_cycles))
    print("오른손 손목:", len(right_wrist_cycles))


    print("\n왼팔 감지:", len(left_cycles), "회")
    print("오른팔 감지:", len(right_cycles), "회")

    # ==========================================
    # 8. 양팔 동작 구간 연결
    # ==========================================

    combined = []
    used_right = set()

    for left in left_cycles:
        matched = None
        best_overlap = 0

        for i, right in enumerate(right_cycles):
            if i in used_right:
                continue

            overlap = max(
                0,
                min(left["end"], right["end"])
                - max(left["start"], right["start"])
                + 1
            )

            if overlap > best_overlap:
                matched = i
                best_overlap = overlap

        if matched is None:
            combined.append({
                "start": left["start"],
                "end": left["end"],
                "motion_status": "REVIEW",
                "reason": "RIGHT_ARM_NOT_MATCHED"
            })
            continue

        right = right_cycles[matched]
        used_right.add(matched)

        start = min(
            left["start"], right["start"]
        )

        end = max(
            left["end"], right["end"]
        )

        # 양팔 동작 종료 시점이 크게 다르면 검토
        synchronized = (
            abs(left["end"] - right["end"]) <= 15
        )

        reliable = (
            left["status"] == "CONFIRMED"
            and right["status"] == "CONFIRMED"
            and synchronized
        )

        combined.append({
            "start": start,
            "end": end,
            "motion_status": (
                "CONFIRMED" if reliable else "REVIEW"
            ),
            "reason": (
                "OK" if reliable
                else "OCCLUSION_OR_ASYNC"
            )
        })

    for i, right in enumerate(right_cycles):
        if i not in used_right:
            combined.append({
                "start": right["start"],
                "end": right["end"],
                "motion_status": "REVIEW",
                "reason": "LEFT_ARM_NOT_MATCHED"
            })

    combined.sort(key=lambda x: x["start"])

    # ==========================================
    # 9. 운동 1회별 AI 자세 결과 결합
    # ==========================================

    rep_results = []

    for number, cycle in enumerate(combined, start=1):
        start = cycle["start"]
        end = cycle["end"]

        frames = [
            row for row in frame_results
            if start <= row["frame"] <= end
        ]

        valid = [
            row for row in frames
            if (
                row["pose_label"] in (
                    "ELB_COR",
                    "ELB_A",
                    "ELB_B"
                )
                and row["confidence"] >= MIN_CONFIDENCE
            )
        ]

        counts = Counter(
            row["pose_label"] for row in valid
        )

        if cycle["motion_status"] != "CONFIRMED":
            pose_label = "REVIEW"

        elif len(valid) < MIN_VOTES:
            pose_label = "REVIEW"

        else:
            a_ratio = (
                counts["ELB_A"] / len(valid)
            )

            b_ratio = (
                counts["ELB_B"] / len(valid)
            )

            if a_ratio >= ERROR_RATIO:
                pose_label = "ELB_A"

            elif b_ratio >= ERROR_RATIO:
                pose_label = "ELB_B"

            else:
                pose_label = "ELB_COR"

        # 보조 규칙: 동작 구간의 관절 각도로 AI 분류 결과를 보완.
        rule_errors = []
        if cycle["motion_status"] == "CONFIRMED":
            for side in ("left", "right"):
                angles = [row[f"{side}_angle"] for row in frames
                          if np.isfinite(row[f"{side}_angle"])]
                # 한 프레임의 튐 대신 하위 20% 각도를 사용.
                if len(angles) >= ELBOW_RULE_MIN_FRAMES and np.percentile(angles, 20) > ELBOW_TARGET_ANGLE:
                    if "ELB_A" not in rule_errors:
                        rule_errors.append("ELB_A")
                arm_angles = [row.get(f"{side}_upper_arm_angle", np.nan) for row in frames]
                arm_angles = [v for v in arm_angles if np.isfinite(v)]
                if len(arm_angles) >= ELBOW_RULE_MIN_FRAMES:
                    lifted_ratio = sum(v > ELBOW_UPPER_ARM_LIMIT for v in arm_angles) / len(arm_angles)
                    if lifted_ratio >= ERROR_RATIO and "ELB_B" not in rule_errors:
                        rule_errors.append("ELB_B")
            # AI가 이미 감지한 오류는 유지. 정상/미확정일 때만 보조 규칙 적용.
            if pose_label in ("ELB_COR", "REVIEW") and rule_errors:
                pose_label = "ELB_B" if "ELB_B" in rule_errors else "ELB_A"

        rep_results.append({
            "rep": number,
            "start_frame": start,
            "end_frame": end,
            "pose_label": pose_label,
            "motion_status": cycle["motion_status"],
            "reason": cycle["reason"],
            "ai_votes": dict(counts),
            "rule_errors": rule_errors
        })

    # ==========================================
    # 10. 운동 전체 결과
    # ==========================================

    # CONFIRMED와 REVIEW 모두 일단
    # 분석 가능한 동작 후보로 집계
    total_reps = len(rep_results)

    correct_reps = sum(
        r["pose_label"] == "ELB_COR"
        for r in rep_results
    )

    review_reps = sum(
        r["pose_label"] == "REVIEW"
        for r in rep_results
    )

    result_json = {
        "exercise": "ELBOW",
        "video": "browser",
        "total_reps": total_reps,
        "correct_reps": correct_reps,
        "review_reps": review_reps,
        "rep_results": rep_results
    }


    return result_json


# Knee and neck adapters: preserve original classifiers and counting rules.
import statistics
from contextlib import redirect_stdout
import io
from fastapi import Query

MODEL_FILES = {
    "elbow_flex_ext": AI_MODEL,
    "seated_knee_ext": BASE / "models" / "knee_model.pkl",
    "neck_side_pull": BASE / "models" / "neck_model_23.pkl",
}

SELECTED_IDS = [
    7, 8, 11, 12, 23, 24
]

EXTRA_NAMES = [
    "ear_height_diff",
    "shoulder_height_diff",
    "head_relative_tilt",
    "torso_horizontal_offset",
    "torso_length"
]

expected_names = []

for i in SELECTED_IDS:
    expected_names.extend([
        f"p{i}_x",
        f"p{i}_y",
        f"p{i}_visibility"
    ])

expected_names.extend(EXTRA_NAMES)
TILT_THRESHOLD = 12.0
CENTER_THRESHOLD = 6.0

HOLD_REQUIRED = 5.0
MIN_MOVEMENT_SECONDS = 0.3

BASELINE_FRAMES = 20
def extract_landmarks(landmarks, feature_names):

    # 골반 중심
    cx = (
        landmarks[23].x + landmarks[24].x
    ) / 2

    cy = (
        landmarks[23].y + landmarks[24].y
    ) / 2

    # 어깨 너비로 정규화
    shoulder_width = math.hypot(
        landmarks[11].x - landmarks[12].x,
        landmarks[11].y - landmarks[12].y
    )

    scale = max(
        shoulder_width, 0.03
    )

    def x(i):
        return (
            landmarks[i].x - cx
        ) / scale

    def y(i):
        return (
            landmarks[i].y - cy
        ) / scale

    values = {}

    # 선택한 6개 관절의 18개 특징값
    for i in SELECTED_IDS:

        values[f"p{i}_x"] = x(i)
        values[f"p{i}_y"] = y(i)

        values[f"p{i}_visibility"] = (
            landmarks[i].visibility
        )

    # 추가 5개 특징값
    values["ear_height_diff"] = (
        y(7) - y(8)
    )

    values["shoulder_height_diff"] = (
        y(11) - y(12)
    )

    values["head_relative_tilt"] = (
        values["ear_height_diff"]
        - values["shoulder_height_diff"]
    )

    shoulder_center_x = (
        x(11) + x(12)
    ) / 2

    shoulder_center_y = (
        y(11) + y(12)
    ) / 2

    values["torso_horizontal_offset"] = (
        shoulder_center_x
    )

    values["torso_length"] = math.hypot(
        shoulder_center_x,
        shoulder_center_y
    )

    # 학습 모델에 저장된 순서대로 반환
    return [
        values[name]
        for name in feature_names
    ]


def line_angle(a, b):

    dx = b.x - a.x
    dy = b.y - a.y

    return math.degrees(
        math.atan2(dy, dx)
    )


def wrap_180(value):

    return (value + 90) % 180 - 90



# Provisional geometry thresholds; calibrate against actual test footage.
def serialize_points(lm):
    return [{"x": float(p.x), "y": float(p.y), "visibility": float(p.visibility)}
            if all(math.isfinite(v) for v in (p.x,p.y,p.visibility)) else None for p in lm]


def geometry_metrics(lm, width, height):
    def point(i):
        p = lm[i]
        if p.visibility < .5 or not all(math.isfinite(v) for v in (p.x,p.y)):
            raise ValueError()
        return np.array([p.x*width, p.y*height])
    out = {}
    try:
        ls, rs, lh, rh = [point(i) for i in (11,12,23,24)]
        torso = (ls+rs-lh-rh)/2
        scale = np.linalg.norm(torso)
        if scale < 1: return out
        out["torso"] = math.degrees(math.atan2(torso[0], -torso[1]))
        out["shoulder"] = float((ls[1]-rs[1])/scale)
        for side, hip, knee in (("left",lh,25),("right",rh,26)):
            try: out[side+"_thigh"] = float((point(knee)[1]-hip[1])/scale)
            except ValueError: pass
        try: out["head"] = wrap_180(line_angle(lm[7],lm[8])-line_angle(lm[11],lm[12])) if min(lm[7].visibility,lm[8].visibility)>=.5 else None
        except ValueError: pass
    except ValueError: pass
    return out


def apply_other_rules(item, row):
    metrics = row.pop("geometry", {})
    samples = item.setdefault("geometry_baseline", [])
    if len(samples)<15:
        if "torso" in metrics: samples.append(metrics)
        return []
    base = {k: statistics.median([x[k] for x in samples if x.get(k) is not None])
            for k in metrics if any(x.get(k) is not None for x in samples)}
    def delta(k):
        return metrics[k]-base[k] if metrics.get(k) is not None and k in base else 0
    errors = []
    if item["exercise"] == "seated_knee_ext":
        for side in ("left","right"):
            if row[side+"_angle"] >= 130:
                if abs(delta("torso")) > 15: errors.append("KNEE_A")
                if delta(side+"_thigh") < -.15: errors.append("KNEE_B_"+side)
    elif abs(delta("head")) >= 12:
        if abs(delta("torso")) > 12: errors.append("NECK_A")
        # Positive head tilt corresponds to right image-side ear moving down.
        if delta("head") * delta("shoulder") > 0 and abs(delta("shoulder")) > .10:
            errors.append("NECK_B")
    return sorted(set(errors))


def analyze_other(frame, pose, bundle, number, timestamp, exercise):
    detected = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), timestamp)
    model = bundle["model"]
    row = {"frame": number, "time_sec": timestamp / 1000, "label": "NO_POSE",
           "confidence": 0.0, "left_angle": np.nan, "right_angle": np.nan,
           "time": timestamp / 1000, "raw_tilt": None, "ai_prediction": "NO_POSE", "landmarks": []}
    if not detected.pose_landmarks:
        return row
    lm = detected.pose_landmarks[0]
    row["landmarks"] = serialize_points(lm)
    row["geometry"] = geometry_metrics(lm, frame.shape[1], frame.shape[0])
    if exercise == "seated_knee_ext":
        row["left_angle"] = get_angle(lm, [23,25,27])
        row["right_angle"] = get_angle(lm, [24,26,28])
        values = make_features(lm)
    else:
        values = extract_landmarks(lm, bundle["feature_names"])
        if all(lm[i].visibility >= 0.4 for i in [7,8,11,12]):
            row["raw_tilt"] = wrap_180(line_angle(lm[7],lm[8]) - line_angle(lm[11],lm[12]))
    if values is not None and np.isfinite(values).all():
        X = np.asarray([values], dtype=np.float64)
        label = str(model.predict(X)[0])
        row["label"] = row["ai_prediction"] = label
        row["confidence"] = float(np.max(model.predict_proba(X)[0])) if hasattr(model,"predict_proba") else 1.0
    return row

SMOOTH_WINDOW = 5

def summarize_knee(frames):
    def smooth_angles(side):
        history = deque(maxlen=SMOOTH_WINDOW)
        result = []

        for row in frames:
            angle = row[f"{side}_angle"]

            if not np.isfinite(angle):
                history.clear()
                result.append(np.nan)
                continue

            history.append(angle)

            result.append(
                float(np.median(history))
            )

        return result

    left = smooth_angles("left")
    right = smooth_angles("right")



    # =========================================
    # 8. 무릎 상태 분류
    # =========================================

    KNEE_BENT = 120
    KNEE_EXTENDED = 140

    HOLD_REQUIRED = 5.0
    STABLE = 3

    def knee_state(angle):
        if not np.isfinite(angle):
            return "UNKNOWN"

        if angle <= KNEE_BENT:
            return "BENT"

        if angle >= KNEE_EXTENDED:
            return "EXTENDED"

        return "TRANSITION"


    # =========================================
    # 9. 다리별 운동 횟수 및 5초 유지
    # =========================================

    def detect_knee_reps(side, angles):

        state = "WAIT_EXTENDED"

        extended_count = 0
        bent_count = 0

        saw_bent_before = False
        start_frame = None

        current_hold_start = None
        longest_hold = 0.0

        missing_in_cycle = False
        uncertain_start = False

        cycles = []

        print(f"\n========== {side.upper()} 무릎 ==========")

        for i, angle in enumerate(angles):

            frame = frames[i]["frame"]
            posture = knee_state(angle)

            # ---------------------------------
            # 아직 다리를 펴지 않은 상태
            # ---------------------------------

            if state == "WAIT_EXTENDED":

                if posture == "BENT":
                    saw_bent_before = True

                if posture == "EXTENDED":
                    extended_count += 1
                else:
                    extended_count = 0

                if extended_count >= STABLE:

                    state = "EXTENDED"

                    start_frame = frame - STABLE + 1

                    # 준비 자세가 관측되지 않았으면
                    # 운동 시작이 불확실한 것으로 기록
                    uncertain_start = not saw_bent_before

                    current_hold_start = start_frame
                    longest_hold = frames[i]["time_sec"] - frames[start_frame]["time_sec"]

                    bent_count = 0
                    missing_in_cycle = False

                    print(
                        f"프레임 {start_frame}: "
                        f"다리 펴기 감지"
                    )

            # ---------------------------------
            # 다리를 편 이후의 동작
            # ---------------------------------

            elif state == "EXTENDED":

                if posture == "EXTENDED":

                    bent_count = 0

                    if current_hold_start is None:
                        current_hold_start = frame

                    duration = frames[i]["time_sec"] - frames[current_hold_start]["time_sec"]

                    longest_hold = max(
                        longest_hold, duration
                    )

                elif posture == "UNKNOWN":

                    # 측정할 수 없는 시간은
                    # 연속 유지시간에 포함하지 않음
                    current_hold_start = None
                    missing_in_cycle = True
                    bent_count = 0

                else:

                    # TRANSITION 또는 BENT가 나오면
                    # 연속 유지 구간 종료
                    current_hold_start = None

                    if posture == "BENT":
                        bent_count += 1
                    else:
                        bent_count = 0

                    if bent_count >= STABLE:

                        held = (
                            longest_hold >= HOLD_REQUIRED
                        )
                        # 횟수 감지와 5초 유지 판정은 별개의 결과
                    
                        count_detected = True
                        hold_verified = held


                        if (
                            uncertain_start
                            or missing_in_cycle
                        ):
                            status = "REVIEW"
                        else:
                            status = "CONFIRMED"

                        cycle = {
        "side": side.upper(),
        "start_frame": start_frame,
        "end_frame": frame,
        "hold_seconds": round(longest_hold, 2),
        "held_5_sec": held,
        "count_detected": True,
        "hold_verified": held,
        "motion_status": status
    }


                        cycles.append(cycle)

                        print(
                            f"프레임 {frame}: "
                            f"복귀 감지 | "
                            f"최장 유지 {longest_hold:.2f}초 | "
                            f"5초 성공: {held} | "
                            f"{status}"
                        )

                        # 다음 동작 준비
                        state = "WAIT_EXTENDED"

                        extended_count = 0
                        bent_count = 0

                        saw_bent_before = True
                        current_hold_start = None

        if state == "EXTENDED":
            print("주의: 복귀가 확인되지 않은 동작 있음")

        print(
            f"{side.upper()} 감지 동작:",
            len(cycles),
            "회"
        )

        return cycles


    # =========================================
    # 양쪽 무릎 실행
    # =========================================

    left_reps = detect_knee_reps(
        "left", left
    )

    right_reps = detect_knee_reps(
        "right", right
    )

    # 좌우 개별 동작 후보
    reps = left_reps + right_reps

    reps.sort(
        key=lambda x: x["start_frame"]
    )

    print("\n========== 무릎 횟수 요약 ==========")

    print("왼쪽 무릎:", len(left_reps), "회")
    print("오른쪽 무릎:", len(right_reps), "회")
    print("전체 다리 동작 후보:", len(reps), "회")

    print("※ 양다리를 묶은 세트 횟수는 아직 아님")

    # =========================================
    # 10. AI 자세 판정 결합
    # =========================================

    rep_results = []

    for i, rep in enumerate(reps, start=1):
        segment = [
            row for row in frames
            if (
                rep["start_frame"]
                <= row["frame"]
                <= rep["end_frame"]
            )
        ]

        valid = [
            row for row in segment
            if (
                row["label"] in (
                    "KNEE_COR", "KNEE_A", "KNEE_B"
                )
                and row["confidence"] >= MIN_CONFIDENCE
            )
        ]

        counts = Counter(
            row["label"] for row in valid
        )

        if rep["motion_status"] == "REVIEW":
            label = "REVIEW"

        elif len(valid) < MIN_VOTES:
            label = "REVIEW"

        else:
            a_ratio = counts["KNEE_A"] / len(valid)
            b_ratio = counts["KNEE_B"] / len(valid)

            if a_ratio >= ERROR_RATIO:
                label = "KNEE_A"

            elif b_ratio >= ERROR_RATIO:
                label = "KNEE_B"

            else:
                label = "KNEE_COR"

        for rule, key in (("KNEE_A", "KNEE_A"), ("KNEE_B_"+rep["side"].lower(), "KNEE_B")):
            if sum(rule in r.get("rule_errors", []) for r in segment) >= max(3, len(segment)*ERROR_RATIO):
                if label == "KNEE_COR": label = key
        posture_label = label
        # 5초를 채우지 못하면 별도 표시
        if label == "KNEE_COR" and not rep["held_5_sec"]:
            label = "HOLD_FAILED"

        rep_results.append({
            "rep": i,
            **rep,
            "pose_label": label,
            "posture_label": posture_label,
            "ai_votes": dict(counts)
        })

    # =========================================
    # 11. 최종 결과 생성
    # =========================================

    total_reps = len(rep_results)

    correct_reps = sum(
        r["pose_label"] == "KNEE_COR"
        and r["held_5_sec"]
        for r in rep_results
    )

    review_reps = sum(
        r["pose_label"] == "REVIEW"
        for r in rep_results
    )

    result = {
        "exercise": "KNEE",
        "video": "browser",
        "total_reps": total_reps,
        "correct_reps": correct_reps,
        "review_reps": review_reps,
        "rep_results": rep_results
    }
    result["count_unit"] = "single_leg_movement"
    return result


def summarize_neck(records):
    records = [r.copy() for r in records]
    initial_angles = [
        r["raw_tilt"]
        for r in records[:BASELINE_FRAMES]
        if r["raw_tilt"] is not None
    ]

    if len(initial_angles) < 5:

        raise RuntimeError(
            "초기 중앙 자세를 충분히 인식하지 못했습니다."
        )

    baseline = statistics.median(
        initial_angles
    )

    print(
        "\n초기 기준 각도:",
        round(baseline, 2)
    )

    for record in records:

        raw = record["raw_tilt"]

        if raw is None:

            record["tilt"] = None
            record["state"] = "UNKNOWN"

            continue

        tilt = wrap_180(
            raw - baseline
        )

        record["tilt"] = round(
            tilt, 2
        )

        if tilt >= TILT_THRESHOLD:

            state = "POSITIVE"

        elif tilt <= -TILT_THRESHOLD:

            state = "NEGATIVE"

        elif abs(tilt) <= CENTER_THRESHOLD:

            state = "CENTER"

        else:

            state = "TRANSITION"

        record["state"] = state


    # ==========================================
    # 9. 목 기울임 + 5초 유지 감지
    # ==========================================

    events = []

    active_side = None

    start_frame = None
    hold_start = None

    longest_hold = 0.0

    saw_center = False
    has_missing = False

    for record in records:

        state = record["state"]
        frame = record["frame"]

        # ----------------------------------
        # 아직 동작하지 않은 상태
        # ----------------------------------

        if active_side is None:

            if state == "CENTER":
                saw_center = True

            if state in (
                "POSITIVE",
                "NEGATIVE"
            ):

                active_side = state

                start_frame = frame
                hold_start = frame

                longest_hold = 0.0

                has_missing = not saw_center

        # ----------------------------------
        # 목 기울임 진행 중
        # ----------------------------------

        else:

            if state == active_side:

                if hold_start is None:
                    hold_start = frame

                duration = record["time"] - records[hold_start]["time"]

                longest_hold = max(
                    longest_hold,
                    duration
                )

            # ----------------------------------
            # 중앙 복귀 → 동작 완료
            # ----------------------------------

            elif state == "CENTER":

                movement_seconds = record["time"] - records[start_frame]["time"]

                event = {
                    "side": active_side,
                    "start_frame": start_frame,
                    "end_frame": frame,
                    "movement_seconds": round(
                        movement_seconds, 2
                    ),
                    "hold_seconds": round(
                        longest_hold, 2
                    ),
                    "held_5_sec": (
                        longest_hold >= HOLD_REQUIRED
                    ),
                    "review": has_missing
                }

                if movement_seconds >= MIN_MOVEMENT_SECONDS:

                    events.append(event)

                else:

                    print(
                        f"순간 움직임 제외: "
                        f"{active_side} "
                        f"{movement_seconds:.2f}초"
                    )

                # 다음 동작 준비
                active_side = None
                start_frame = None
                hold_start = None

                longest_hold = 0.0

                saw_center = True
                has_missing = False

            # ----------------------------------
            # 전환 또는 인식 실패
            # ----------------------------------

            else:

                hold_start = None

                if state == "UNKNOWN":
                    has_missing = True


    # ==========================================
    # 10. 동작별 AI 자세 판정
    # ==========================================

    for event in events:

        start = event["start_frame"]
        end = event["end_frame"]

        votes = Counter(
            records[i]["ai_prediction"]
            for i in range(start, end + 1)
            if records[i]["ai_prediction"] != "NO_POSE"
        )

        if votes:

            label, count = votes.most_common(1)[0]

            total_votes = sum(
                votes.values()
            )

            event["ai_posture"] = label

            event["ai_vote_ratio"] = round(
                count / total_votes, 3
            )

        else:

            event["ai_posture"] = "REVIEW"
            event["ai_vote_ratio"] = 0.0

            event["review"] = True

        segment = records[start:end+1]
        for rule in ("NECK_A", "NECK_B"):
            if sum(rule in r.get("rule_errors", []) for r in segment) >= max(3, len(segment)*ERROR_RATIO):
                if event["ai_posture"] == "NECK_COR": event["ai_posture"] = rule
        event["posture_review"] = (
            event["ai_posture"] != "NECK_COR"
            or event["review"]
        )


    # ==========================================
    # 11. 좌우 운동 한 세트 계산
    # ==========================================

    sets = []

    pending = None

    for event in events:

        if pending is None:

            pending = event

        elif event["side"] != pending["side"]:

            sets.append({
                "first": pending,
                "second": event,
                "both_held_5_sec": (
                    pending["held_5_sec"]
                    and event["held_5_sec"]
                ),
                "posture_review": (
                    pending["posture_review"]
                    or event["posture_review"]
                )
            })

            pending = None

        else:

            pending = event

    return {"exercise":"NECK", "video":"browser", "total_reps":len(sets),
            "correct_reps":sum(s["both_held_5_sec"] and not s["posture_review"] for s in sets),
            "review_reps":sum(s["posture_review"] for s in sets),
            "total_side_movements":len(events), "events":events, "sets":sets,
            "count_unit":"left_right_set"}


def elbow_live_feedback(item, result):
    if result["pose_label"] == "NO_POSE":
        return None
    recent = item["frames"][-ELBOW_RULE_MIN_FRAMES:]
    for side in ("left", "right"):
        values = [r.get(f"{side}_upper_arm_angle", np.nan) for r in recent]
        if len(values) == ELBOW_RULE_MIN_FRAMES and sum(np.isfinite(v) and v > ELBOW_UPPER_ARM_LIMIT for v in values) >= 3:
            return "ELB_B"
    last_rep = item.get("last_rep")
    if last_rep and result["time_sec"] - last_rep["end_time"] <= 2.5:
        if last_rep["pose_label"] in ("ELB_A", "ELB_B", "REVIEW"):
            return last_rep["pose_label"]
    return None


def summarize_session(item):
    exercise = item["exercise"]
    if exercise == "elbow_flex_ext":
        return summarize(item["frames"])
    if exercise == "seated_knee_ext":
        return summarize_knee(item["frames"])
    return summarize_neck(item["frames"])
app = FastAPI(title="SOLFAC AI")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"], allow_methods=["*"], allow_headers=["*"])
lock = Lock()
sessions = {}
models = {}

@app.get("/")
def health():
    return {"service": "SOLFAC AI", "status": "ready", "supported_exercises": list(MODEL_FILES)}

@app.post("/ai/{session_id}/start")
def start(session_id: str, exercise_id: str = Query("elbow_flex_ext")):
    if exercise_id not in MODEL_FILES:
        raise HTTPException(400, "Unsupported exercise")
    with lock:
        if session_id in sessions:
            raise HTTPException(409, "AI session already exists")
        if exercise_id not in models:
            bundle = joblib.load(MODEL_FILES[exercise_id])
            expected = expected_names if exercise_id == "neck_side_pull" else FEATURE_COLUMNS
            if exercise_id != "elbow_flex_ext" and list(bundle["feature_names"]) != expected:
                raise HTTPException(500, "Model feature names mismatch")
            models[exercise_id] = bundle
        pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        sessions[session_id] = {"pose": pose, "frames": [], "last": -1, "exercise": exercise_id, "live": None, "last_live": -1}
    return {"status": "started"}

@app.post("/ai/{session_id}/frame")
def receive_frame(session_id: str, image: UploadFile = File(...), timestamp_ms: int = Form(...)):
    raw = image.file.read(2_000_001)
    if len(raw) > 2_000_000:
        raise HTTPException(413, "Frame too large")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image")
    with lock:
        item = sessions.get(session_id)
        if item is None:
            raise HTTPException(404, "AI session not found")
        if timestamp_ms < 0 or timestamp_ms <= item["last"]:
            raise HTTPException(400, "Timestamp must increase")
        exercise = item["exercise"]
        bundle = models[exercise]
        if exercise == "elbow_flex_ext":
            result = analyze_frame(frame, item["pose"], bundle["model"], len(item["frames"]), timestamp_ms)
        else:
            result = analyze_other(frame, item["pose"], bundle, len(item["frames"]), timestamp_ms, exercise)
        if exercise != "elbow_flex_ext":
            result["rule_errors"] = apply_other_rules(item, result)
        points = result.pop("landmarks", [])
        item["frames"].append(result)
        item["last"] = timestamp_ms
        if timestamp_ms - item["last_live"] >= 1000:
            try:
                with redirect_stdout(io.StringIO()):
                    live = summarize_session(item)
                item["live"] = {key: live[key] for key in ("total_reps", "correct_reps")}
                if live.get("rep_results"):
                    last_rep = live["rep_results"][-1]
                    item["last_rep"] = {**last_rep, "end_time": item["frames"][last_rep["end_frame"]]["time_sec"]}
                elif live.get("events"):
                    e = live["events"][-1]
                    item["last_rep"] = {"pose_label": e["ai_posture"] if e["ai_posture"] != "NECK_COR" else ("HOLD_FAILED" if not e["held_5_sec"] else "NECK_COR"), "end_time": item["frames"][e["end_frame"]]["time_sec"]}
            except RuntimeError:
                item["live"] = None
            item["last_live"] = timestamp_ms
        feedback_label = elbow_live_feedback(item, result) if exercise == "elbow_flex_ext" else None
        if exercise != "elbow_flex_ext":
            recent = item["frames"][-5:]
            rules = Counter(e.split("_left")[0].split("_right")[0] for r in recent for e in r.get("rule_errors", []))
            if rules and rules.most_common(1)[0][1] >= 3:
                feedback_label = rules.most_common(1)[0][0]
        last = item.get("last_rep")
        if not feedback_label and last and 0 <= result["time_sec"]-last["end_time"] <= 3:
            feedback_label = last["pose_label"]
        return {"landmarks": points, "frames": len(item["frames"]), "pose_label": result.get("pose_label", result.get("label")), "confidence": result["confidence"], "live": item["live"], "feedback_label": feedback_label}

@app.post("/ai/{session_id}/finish")
def finish(session_id: str):
    with lock:
        item = sessions.get(session_id)
        if item is None:
            raise HTTPException(404, "AI session not found")
        if not item["frames"]:
            raise HTTPException(409, "No analyzed frames")
        try:
            result = summarize_session(item)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        item["pose"].close()
        del sessions[session_id]
        return result

@app.delete("/ai/{session_id}")
def cancel(session_id: str):
    with lock:
        item = sessions.pop(session_id, None)
        if item is not None:
            item["pose"].close()
    return {"status": "cancelled"}
