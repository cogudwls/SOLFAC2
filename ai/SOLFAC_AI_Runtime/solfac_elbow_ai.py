
import cv2
import csv
import json
import math
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from pathlib import Path
from collections import Counter, deque

# ==========================================
# 1. 기본 경로
# ==========================================

BASE = Path(__file__).resolve().parent

DATASET = Path(
    r"E:\SOLFAC_Dataset\SOLFAC_Dataset"
)

VIDEO_NAME = None

POSE_MODEL = BASE / "pose_landmarker_lite.task"
AI_MODEL = BASE / "models" / "elbow_model.pkl"

OUTPUT = BASE / "results" / "elbow"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. 실험용 파라미터
# ==========================================

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

# ==========================================
# 3. 파일과 모델 불러오기
# ==========================================


# 학습에 사용하지 않은 팔꿈치 테스트 영상 선택
TEST_FOLDER = DATASET / "raw" / "posture_test" / "elbow"

if not TEST_FOLDER.exists():
    raise FileNotFoundError(
        f"테스트 폴더가 없습니다: {TEST_FOLDER}"
    )

matches = sorted(TEST_FOLDER.rglob("*.mp4"))

if not matches:
    raise FileNotFoundError(
        "팔꿈치 POSTURE_TEST 영상이 없습니다."
    )

# 첫 번째 테스트 영상 선택
video_path = matches[0]

# 기존 코드에서 사용하는 영상 이름도 갱신
VIDEO_NAME = video_path.name

print("선택된 테스트 영상:", VIDEO_NAME)
print("영상 위치:", video_path)


if not POSE_MODEL.exists():
    raise FileNotFoundError(POSE_MODEL)

if not AI_MODEL.exists():
    raise FileNotFoundError(AI_MODEL)

model_data = joblib.load(AI_MODEL)
model = model_data["model"]

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("영상 열기 실패")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

# ==========================================
# 4. 관절 특징값 추출
#    학습용 99개 특징값과 동일한 형식
# ==========================================

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

# ==========================================
# 6. 원본 영상 전체 프레임 분석
# ==========================================

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(POSE_MODEL)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

frame_results = []

print("\n========== SOLFAC 팔꿈치 AI ==========")
print("영상:", VIDEO_NAME)
print("FPS:", round(fps, 2))

with mp.tasks.vision.PoseLandmarker.create_from_options(
    options
) as pose:

    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            frame_number * 1000 / fps
        )

        detected = pose.detect_for_video(
            image,
            timestamp
        )

        result = {
            "frame": frame_number,
            "time_sec": round(
                frame_number / fps, 3
            ),
            "left_angle": np.nan,
            "left_wrist_height": np.nan,
            "right_wrist_height": np.nan,
            "right_angle": np.nan,
            "pose_label": "NO_POSE",
            "confidence": 0.0
        }

        if detected.pose_landmarks:
            landmarks = detected.pose_landmarks[0]

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

        frame_results.append(result)

        frame_number += 1

cap.release()

print("총 분석 프레임:", len(frame_results))

# ==========================================
# 7. 관절 인식 실패를 고려한 횟수 측정
# ==========================================

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

    # 양팔이 충분히 굽혀지지 않은 동작인지 검사
    left_angles = [
        row["left_angle"] for row in frames
        if np.isfinite(row["left_angle"])
    ]

    right_angles = [
        row["right_angle"] for row in frames
        if np.isfinite(row["right_angle"])
    ]

    if left_angles and right_angles:
        left_min = min(left_angles)
        right_min = min(right_angles)

        # TRAIN 데이터에서 확인한 굽힘 부족 특징
        if (
            cycle["motion_status"] == "CONFIRMED"
            and left_min >= 150
            and right_min >= 150
        ):
            # 확정 ELB_A가 아니라 검증할 후보
            pose_label = "ELB_A_CANDIDATE"

            print(
                f"[굽힘 부족 의심] "
                f"왼팔 {left_min:.1f}도 / "
                f"오른팔 {right_min:.1f}도"
            )


    rep_results.append({
        "rep": number,
        "start_frame": start,
        "end_frame": end,
        "pose_label": pose_label,
        "motion_status": cycle["motion_status"],
        "reason": cycle["reason"],
        "ai_votes": dict(counts)
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
    "video": VIDEO_NAME,
    "total_reps": total_reps,
    "correct_reps": correct_reps,
    "review_reps": review_reps,
    "rep_results": rep_results
}

# ==========================================
# 11. 결과 저장
# ==========================================

stem = Path(VIDEO_NAME).stem

json_path = OUTPUT / f"{stem}_result.json"
csv_path = OUTPUT / f"{stem}_frames.csv"

with open(
    json_path, "w",
    encoding="utf-8"
) as f:
    json.dump(
        result_json,
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    csv_path, "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,

fieldnames=[
    "frame",
    "time_sec",
    "left_angle",
    "right_angle",
    "left_wrist_height",
    "right_wrist_height",
    "pose_label",
    "confidence"
]

    )

    writer.writeheader()
    writer.writerows(frame_results)

# ==========================================
# 12. 터미널 출력
# ==========================================

print("\n========== 운동별 AI 분석 ==========")

for rep in rep_results:
    print(
        f"{rep['rep']}회 | "
        f"프레임 {rep['start_frame']}~"
        f"{rep['end_frame']} | "
        f"{rep['pose_label']} | "
        f"{rep['motion_status']}"
    )

print("\n========== 최종 결과 ==========")
print("total_reps:", total_reps)
print("correct_reps:", correct_reps)
print("review_reps:", review_reps)

print("\nJSON 저장:", json_path)
print("CSV 저장:", csv_path)

print("\n주의:")
print("이 결과는 통합 프로그램의 임시 예측입니다.")
print("total_reps에는 REVIEW 후보도 포함됩니다.")
print("정확도 및 오자세 감지 성능은 별도 검증이 필요합니다.")
print("\nSOLFAC 팔꿈치 AI 통합 분석 완료!")
