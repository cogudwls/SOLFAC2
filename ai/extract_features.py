
import cv2
import csv
import math
import mediapipe as mp

from pathlib import Path
from collections import Counter, defaultdict

# ==========================================
# 1. 파일 경로
# ==========================================

BASE = Path(__file__).resolve().parent

LABEL_CSV = (
    BASE / "prepared_labels" / "solfac_frame_labels.csv"
)

MODEL = BASE / "pose_landmarker_lite.task"

OUTPUT = BASE / "prepared_labels" / "train_features.csv"

if not LABEL_CSV.exists():
    raise FileNotFoundError("라벨 CSV 파일이 없습니다.")

if not MODEL.exists():
    raise FileNotFoundError("MediaPipe 모델 파일이 없습니다.")

# ==========================================
# 2. CSV에서 영상별 라벨 읽기
# ==========================================

videos = defaultdict(dict)

with open(LABEL_CSV, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        video_path = row["video_path"]
        label = row["label"]

        # 라벨 없는 프레임은 학습 대상에서 제외
        if not video_path or not label:
            continue

        frame = int(row["frame"])

        videos[video_path][frame] = {
            "label": label,
            "exercise": row["exercise"]
        }

print("학습 대상 영상:", len(videos), "개")

# ==========================================
# 3. MediaPipe 설정
# ==========================================

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

# 33개 관절의 좌표(x, y)와 가시성
feature_names = []

for i in range(33):
    feature_names.extend([
        f"p{i}_x",
        f"p{i}_y",
        f"p{i}_visibility"
    ])

# ==========================================
# 4. 관절 좌표를 특징값으로 변환
# ==========================================

def extract_landmarks(landmarks):

    # 왼쪽/오른쪽 골반의 중심
    cx = (landmarks[23].x + landmarks[24].x) / 2
    cy = (landmarks[23].y + landmarks[24].y) / 2

    # 어깨 너비를 이용해 좌표 크기 정규화
    shoulder_width = math.hypot(
        landmarks[11].x - landmarks[12].x,
        landmarks[11].y - landmarks[12].y
    )

    scale = max(shoulder_width, 0.03)

    features = []

    for p in landmarks:
        features.extend([
            (p.x - cx) / scale,
            (p.y - cy) / scale,
            p.visibility
        ])

    return features

# ==========================================
# 5. TRAIN 영상 33개 분석
# ==========================================

OUTPUT.parent.mkdir(exist_ok=True)

saved = 0
not_detected = 0
failed = []
label_counts = Counter()

with open(
    OUTPUT, "w", newline="", encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "video_path",
        "exercise",
        "frame",
        "label"
    ] + feature_names)

    for index, (video_path, labels) in enumerate(
        videos.items(), start=1
    ):

        path = Path(video_path)

        print(
            f"\n[{index}/{len(videos)}] "
            f"{path.name}"
        )

        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            print("영상 열기 실패")
            failed.append(path.name)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(fps) or fps <= 0:
            fps = 30

        frame_number = 0
        video_saved = 0

        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as pose:

            while True:
                success, frame = cap.read()

                if not success:
                    break

                # 라벨 없는 프레임은 건너뛰기
                if frame_number not in labels:
                    frame_number += 1
                    continue

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

                result = pose.detect_for_video(
                    image, timestamp
                )

                if not result.pose_landmarks:
                    not_detected += 1
                    frame_number += 1
                    continue

                landmarks = result.pose_landmarks[0]

                features = extract_landmarks(landmarks)

                info = labels[frame_number]

                writer.writerow([
                    str(path),
                    info["exercise"],
                    frame_number,
                    info["label"]
                ] + features)

                label_counts[info["label"]] += 1

                saved += 1
                video_saved += 1
                frame_number += 1

        cap.release()

        print("추출 완료:", video_saved, "프레임")

        # 라벨이 지정된 마지막 프레임까지 읽었는지 확인
        if labels and frame_number <= max(labels):
            print("주의: 영상과 XML 프레임 범위 불일치!")
            failed.append(path.name)

# ==========================================
# 6. 최종 결과 출력
# ==========================================

print("\n========== 추출 결과 ==========")

print("관절 데이터 저장:", saved, "프레임")
print("사람 인식 실패:", not_detected, "프레임")

print("\n라벨별 학습 데이터:")
for label, count in sorted(label_counts.items()):
    print(f"{label}: {count}개")

if failed:
    print("\n확인 필요한 영상:")
    for name in sorted(set(failed)):
        print("-", name)

print("\n저장 위치:")
print(OUTPUT)

print("\n학습용 특징값 추출 완료!")
