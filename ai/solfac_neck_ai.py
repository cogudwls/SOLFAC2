
import cv2
import json
import math
import statistics
import sys

from collections import Counter
from pathlib import Path

import joblib
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. 경로 및 분석 영상 설정
# ==========================================

BASE = Path(__file__).resolve().parent

DATASET = Path(
    r"E:\SOLFAC_Dataset\SOLFAC_Dataset"
)

POSE_MODEL = BASE / "pose_landmarker_lite.task"

AI_MODEL = BASE / "models" / "neck_model_23.pkl"

OUTPUT = BASE / "results" / "neck"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEFAULT_VIDEO = "U01_S002_NECK_TRAIN_A_001.mp4"

VIDEO_NAME = (
    sys.argv[1]
    if len(sys.argv) > 1
    else DEFAULT_VIDEO
)

if not DATASET.is_dir():
    raise FileNotFoundError(
        f"데이터셋 폴더 없음: {DATASET}"
    )

matches = [
    p for p in DATASET.rglob("*.mp4")
    if p.name.casefold() == VIDEO_NAME.casefold()
]

if len(matches) != 1:
    raise FileNotFoundError(
        f"영상 검색 결과 {len(matches)}개: {VIDEO_NAME}"
    )

video_path = matches[0]

for path in (POSE_MODEL, AI_MODEL):
    if not path.is_file():
        raise FileNotFoundError(
            f"모델 파일 없음: {path}"
        )


# ==========================================
# 2. AI 모델 로드 및 특징값 검사
# ==========================================

model_data = joblib.load(AI_MODEL)

model = model_data["model"]

feature_names = list(
    model_data["feature_names"]
)

# 목 중심 관절 6개
# 귀: 7, 8
# 어깨: 11, 12
# 골반: 23, 24

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

if feature_names != expected_names:
    raise ValueError(
        "23개 모델의 특징값 이름 또는 순서가 다릅니다."
    )

print("\n========== SOLFAC 목 AI ==========")

print("분석 영상:", video_path.name)
print("AI 모델:", AI_MODEL.name)
print("입력 특징값:", len(feature_names), "개")


# ==========================================
# 3. 임시 운동 감지 기준
# ==========================================

TILT_THRESHOLD = 12.0
CENTER_THRESHOLD = 6.0

HOLD_REQUIRED = 5.0
MIN_MOVEMENT_SECONDS = 0.3

BASELINE_FRAMES = 20


# ==========================================
# 4. 목 중심 23개 특징값 추출
# ==========================================

def extract_landmarks(landmarks):

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


# ==========================================
# 5. MediaPipe 설정
# ==========================================

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(POSE_MODEL)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

cap = cv2.VideoCapture(
    str(video_path)
)

if not cap.isOpened():
    raise RuntimeError(
        "영상을 열 수 없습니다."
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if not math.isfinite(fps) or fps <= 0:

    cap.release()

    raise RuntimeError(
        "FPS 확인 실패"
    )

print("FPS:", fps)

records = []

frame_idx = 0
last_timestamp = -1


# ==========================================
# 6. 영상 프레임별 AI 분석
# ==========================================

try:

    with vision.PoseLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = round(
                frame_idx * 1000 / fps
            )

            timestamp = max(
                timestamp,
                last_timestamp + 1
            )

            last_timestamp = timestamp

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            record = {
                "frame": frame_idx,
                "time": round(
                    frame_idx / fps, 3
                ),
                "raw_tilt": None,
                "ai_prediction": "NO_POSE"
            }

            if result.pose_landmarks:

                lm = result.pose_landmarks[0]

                # --------------------------
                # A. AI 자세 분류
                # --------------------------

                features = extract_landmarks(
                    lm
                )

                X = np.asarray(
                    [features],
                    dtype=np.float64
                )

                prediction = model.predict(X)[0]

                record["ai_prediction"] = str(
                    prediction
                )

                # --------------------------
                # B. 목 기울임 계산
                # --------------------------

                ids = [7, 8, 11, 12]

                valid = all(
                    lm[i].visibility >= 0.4
                    for i in ids
                )

                if valid:

                    ear_angle = line_angle(
                        lm[7],
                        lm[8]
                    )

                    shoulder_angle = line_angle(
                        lm[11],
                        lm[12]
                    )

                    raw_tilt = wrap_180(
                        ear_angle - shoulder_angle
                    )

                    record["raw_tilt"] = raw_tilt

            records.append(record)

            frame_idx += 1

finally:

    cap.release()

print(
    "분석 프레임:",
    len(records)
)

if not records:
    raise RuntimeError(
        "분석 가능한 프레임이 없습니다."
    )


# ==========================================
# 7. 프레임별 AI 예측 결과
# ==========================================

ai_counts = Counter(
    r["ai_prediction"]
    for r in records
)

print(
    "\n========== 목 AI 프레임 예측 =========="
)

for label, count in ai_counts.most_common():

    print(
        f"{label}: {count}프레임"
    )


# ==========================================
# 8. 초기 중앙 자세 기준
# ==========================================

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

            longest_hold = 1 / fps

            has_missing = not saw_center

    # ----------------------------------
    # 목 기울임 진행 중
    # ----------------------------------

    else:

        if state == active_side:

            if hold_start is None:
                hold_start = frame

            duration = (
                frame - hold_start + 1
            ) / fps

            longest_hold = max(
                longest_hold,
                duration
            )

        # ----------------------------------
        # 중앙 복귀 → 동작 완료
        # ----------------------------------

        elif state == "CENTER":

            movement_seconds = (
                frame - start_frame + 1
            ) / fps

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


# ==========================================
# 12. 최종 결과 출력
# ==========================================

print(
    "\n========== 목 운동 + AI 분석 =========="
)

for idx, event in enumerate(events, 1):

    print(
        f"{idx}회 | "
        f"{event['side']} | "
        f"유지 {event['hold_seconds']}초 | "
        f"5초: {event['held_5_sec']} | "
        f"AI: {event['ai_posture']} | "
        f"AI 투표율: {event['ai_vote_ratio']:.1%} | "
        f"검토: {event['posture_review']}"
    )

print(
    "\n========== 최종 결과 =========="
)

print(
    "한쪽 기울임 완료:",
    len(events),
    "회"
)

print(
    "좌우 세트 후보:",
    len(sets),
    "세트"
)

successful_sets = sum(
    s["both_held_5_sec"]
    for s in sets
)

print(
    "양쪽 5초 유지 세트:",
    successful_sets
)


# ==========================================
# 13. JSON 저장
# ==========================================

output_data = {
    "video": video_path.name,
    "fps": fps,
    "total_frames": len(records),
    "model": AI_MODEL.name,
    "feature_count": len(feature_names),
    "baseline_angle": round(baseline, 2),
    "ai_frame_predictions": dict(ai_counts),
    "total_side_movements": len(events),
    "total_sets": len(sets),
    "events": events,
    "sets": sets,
    "notes": [
        "CVAT 라벨로 학습한 목 자세 분류 모델 사용",
        "귀, 어깨, 골반의 23개 특징값 사용",
        "목 각도 기준은 검증 전 임시 실험값",
        "초기 20프레임을 중앙 자세로 가정",
        "AI 투표율은 정답 확률이 아님",
        "실제 왼쪽과 오른쪽 방향 매핑은 미검증",
        "운동 세트와 자세 판정은 최종 검증 전"
    ]
}

output_path = OUTPUT / (
    video_path.stem + "_result.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_data,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    "\nJSON 저장:",
    output_path
)

print(
    "SOLFAC 목 통합 AI 분석 완료!"
)
