
import cv2
import json
import math
import joblib
import numpy as np
import mediapipe as mp

from pathlib import Path
from collections import Counter, deque

# =========================================
# 1. 경로 설정
# =========================================

BASE = Path(__file__).resolve().parent

DATASET = Path(
    r"E:\SOLFAC_Dataset\SOLFAC_Dataset"
)

POSE_MODEL = BASE / "pose_landmarker_lite.task"
AI_MODEL = BASE / "models" / "knee_model.pkl"

OUTPUT = BASE / "results" / "knee"
OUTPUT.mkdir(parents=True, exist_ok=True)


# 분석할 영상 이름
import sys

VIDEO_NAME = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "U02_S001_KNEE_COUNT_TEST_COR_001.mp4"
)

# 영상이 들어 있는 폴더
VIDEO_FOLDER = DATASET / "raw" / "count_test" / "knee"


# 우선 TRAIN 정상 영상으로 기능 확인

VIDEO_FOLDER = DATASET / "raw" / "count_test" / "knee"


VIDEO_FOLDER = DATASET / "raw" / "train" / "knee"

# =========================================
# 2. 실험용 설정값
# =========================================

# 무릎을 굽힌 준비 자세
BENT_ANGLE = 120

# 무릎을 편 자세
EXTENDED_ANGLE = 155

# 준비 및 복귀 확인 프레임
STABLE_FRAMES = 3

# 5초 유지
HOLD_SECONDS = 5.0

# 순간적인 각도 튐 완화
SMOOTH_WINDOW = 5

# 짧은 추적 오류만 허용
MAX_MISSING_FRAMES = 3

# 모델 예측 신뢰도 기준
MIN_CONFIDENCE = 0.55

# 자세 판정용 최소 프레임
MIN_VOTES = 5

# 임시 오자세 비율 기준
ERROR_RATIO = 0.20

# =========================================
# 3. 모델 및 영상 불러오기
# =========================================

matches = [
    p for p in DATASET.rglob("*.mp4")
    if p.name.upper() == VIDEO_NAME.upper()
]

if len(matches) != 1:
    raise FileNotFoundError(
        f"영상 확인 필요: {VIDEO_NAME}"
    )

video_path = matches[0]

model_data = joblib.load(AI_MODEL)

# 기존 학습 파일은 dict 구조
model = model_data["model"]

feature_names = model_data["feature_names"]

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("영상을 열 수 없습니다.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("FPS를 확인할 수 없습니다.")

print("\n========== SOLFAC 무릎 AI ==========")
print("영상:", VIDEO_NAME)
print("FPS:", fps)

# =========================================
# 4. 관절 각도 계산
# =========================================

def get_angle(landmarks, a, b, c):
    points = []

    for i in [a, b, c]:
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

    p1, p2, p3 = points

    v1 = p1 - p2
    v2 = p3 - p2

    denominator = (
        np.linalg.norm(v1)
        * np.linalg.norm(v2)
    )

    if denominator < 1e-8:
        return np.nan

    cosine = np.clip(
        np.dot(v1, v2) / denominator,
        -1.0, 1.0
    )

    return float(
        math.degrees(math.acos(cosine))
    )

# =========================================
# 5. AI 특징값 추출
# =========================================

def extract_features(landmarks):
    hip_x = (
        landmarks[23].x + landmarks[24].x
    ) / 2

    hip_y = (
        landmarks[23].y + landmarks[24].y
    ) / 2

    shoulder_width = math.hypot(
        landmarks[11].x - landmarks[12].x,
        landmarks[11].y - landmarks[12].y
    )

    if shoulder_width < 1e-6:
        return None

    values = []

    for p in landmarks:
        values.extend([
            (p.x - hip_x) / shoulder_width,
            (p.y - hip_y) / shoulder_width,
            p.visibility
        ])

    if len(values) != len(feature_names):
        raise RuntimeError(
            "학습 모델의 특징값 개수가 일치하지 않습니다."
        )

    return np.asarray(
        [values], dtype=np.float64
    )

# =========================================
# 6. 원본 영상 전체 프레임 분석
# =========================================

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(POSE_MODEL)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

frames = []

with mp.tasks.vision.PoseLandmarker.create_from_options(
    options
) as pose:

    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            frame_number * 1000 / fps
        )

        detected = pose.detect_for_video(
            image, timestamp
        )

        row = {
            "frame": frame_number,
            "time_sec": frame_number / fps,
            "left_angle": np.nan,
            "right_angle": np.nan,
            "label": "NO_POSE",
            "confidence": 0.0
        }

        if detected.pose_landmarks:
            lm = detected.pose_landmarks[0]

            # 왼쪽 고관절-무릎-발목
            row["left_angle"] = get_angle(
                lm, 23, 25, 27
            )

            # 오른쪽 고관절-무릎-발목
            row["right_angle"] = get_angle(
                lm, 24, 26, 28
            )

            features = extract_features(lm)

            if features is not None:
                prediction = model.predict(features)[0]

                row["label"] = str(prediction)

                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(
                        features
                    )[0]

                    row["confidence"] = float(
                        np.max(probability)
                    )
                else:
                    row["confidence"] = 1.0

        # 무릎 관절 인식 상태 이미지 저장
        if frame_number in [30, 75, 110, 145, 180, 215, 250]:

            debug_frame = frame.copy()

            if detected.pose_landmarks:
                lm = detected.pose_landmarks[0]

                h, w = debug_frame.shape[:2]

                for ids, color in [
                    ([23, 25, 27], (255, 0, 0)),
                    ([24, 26, 28], (0, 0, 255))
                ]:
                    points = []

                    for idx in ids:
                        p = lm[idx]

                        x = int(p.x * w)
                        y = int(p.y * h)

                        points.append((x, y))

                        cv2.circle(
                            debug_frame,
                            (x, y),
                            7,
                            color,
                            -1
                        )

                        cv2.putText(
                            debug_frame,
                            str(idx),
                            (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                    cv2.line(
                        debug_frame,
                        points[0],
                        points[1],
                        color,
                        3
                    )

                    cv2.line(
                        debug_frame,
                        points[1],
                        points[2],
                        color,
                        3
                    )

            # 프레임과 양쪽 무릎 각도 표시
            cv2.rectangle(
                debug_frame,
                (0, 0),
                (450, 105),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                debug_frame,
                f"FRAME: {frame_number}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                debug_frame,
                f"LEFT: {row['left_angle']:.1f}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            cv2.putText(
                debug_frame,
                f"RIGHT: {row['right_angle']:.1f}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            debug_dir = OUTPUT / "diagnostic"
            debug_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            cv2.imwrite(
                str(
                    debug_dir
                    / f"knee_frame_{frame_number}.png"
                ),
                debug_frame
            )

        frames.append(row)
        frame_number += 1

cap.release()

print("분석 프레임:", len(frames))

# =========================================
# 7. 양쪽 무릎 각도 안정화
# =========================================

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
                longest_hold = STABLE / fps

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

                duration = (
                    frame - current_hold_start + 1
                ) / fps

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

    # 5초를 채우지 못하면 별도 표시
    if label == "KNEE_COR" and not rep["held_5_sec"]:
        label = "HOLD_FAILED"

    rep_results.append({
        "rep": i,
        **rep,
        "pose_label": label,
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
    "video": VIDEO_NAME,
    "fps": fps,
    "total_reps": total_reps,
    "correct_reps": correct_reps,
    "review_reps": review_reps,
    "rep_results": rep_results
}

save_path = (
    OUTPUT / f"{Path(VIDEO_NAME).stem}_result.json"
)

with open(
    save_path, "w",
    encoding="utf-8"
) as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )

# =========================================
# 12. 터미널 결과
# =========================================

print("\n========== 무릎 AI 프레임 예측 ==========")

for label, count in Counter(
    row["label"] for row in frames
).most_common():
    print(label, ":", count)

print("\n========== 운동 횟수 및 5초 유지 ==========")

for rep in rep_results:
    print(
        f"{rep['rep']}회 | "
        f"{rep['hold_seconds']}초 유지 | "
        f"5초 성공: {rep['held_5_sec']} | "
        f"{rep['pose_label']} | "
        f"{rep['motion_status']}"
    )

print("\n========== 최종 결과 ==========")
print("total_reps:", total_reps)
print("correct_reps:", correct_reps)
print("review_reps:", review_reps)

print("\nJSON 저장:", save_path)

print("\n주의:")
print("각도 기준은 임시 실험값입니다.")
print("운동 시작/완료 감지와 자세 분류는 검증 전입니다.")
print("5초 유지시간은 관측 가능한 상태를 기준으로 계산합니다.")

print("\nSOLFAC 무릎 AI 분석 완료!")
