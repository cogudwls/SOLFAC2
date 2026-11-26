
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ==========================================
# 1. 기본 경로
# ==========================================

BASE = Path(__file__).resolve().parent

OUTPUT = BASE / "results" / "integrated"
OUTPUT.mkdir(parents=True, exist_ok=True)

EXERCISES = {
    "1": {
        "name": "elbow",
        "display": "팔꿈치 굽히기",
        "script": "solfac_elbow_ai.py",
    },
    "2": {
        "name": "knee",
        "display": "앉아서 무릎 펴기",
        "script": "solfac_knee_ai.py",
    },
    "3": {
        "name": "neck",
        "display": "목 좌우 기울이기",
        "script": "solfac_neck_ai.py",
    },
}

# ==========================================
# 2. 운동 선택
# ==========================================

def select_exercise():

    print("\n================================")
    print("        SOLFAC 통합 운동 AI")
    print("================================")

    for key, value in EXERCISES.items():
        print(f"{key}. {value['display']}")

    print("0. 종료")

    while True:
        selected = input("\n운동 번호 선택: ").strip()

        if selected == "0":
            return None

        if selected in EXERCISES:
            return EXERCISES[selected]

        print("0~3 중에서 선택해 주세요.")


# ==========================================
# 3. 개별 AI 프로그램 실행
# ==========================================

def run_exercise(exercise):

    script_path = BASE / exercise["script"]

    if not script_path.is_file():
        print(f"\n[오류] 파일 없음: {script_path}")
        return

    print("\n--------------------------------")
    print("운동:", exercise["display"])
    print("실행 파일:", exercise["script"])
    print("--------------------------------")

    video_name = input(
        "영상 파일명 입력 "
        "(Enter: 개별 AI 기본 영상): "
    ).strip()

    command = [
        sys.executable,
        str(script_path),
    ]

    if video_name:
        command.append(video_name)

    started_at = datetime.now().astimezone()

    print("\nAI 분석 시작!\n")

    # 개별 AI 실행 및 출력 실시간 표시
    completed = subprocess.run(
        command,
        cwd=BASE,
        check=False,
    )

    finished_at = datetime.now().astimezone()

    status = (
        "success"
        if completed.returncode == 0
        else "failed"
    )

    # 실행 기록 저장
    run_record = {
        "exercise": exercise["name"],
        "exercise_display": exercise["display"],
        "script": exercise["script"],
        "video_argument": (
            video_name if video_name else None
        ),
        "status": status,
        "return_code": completed.returncode,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "note": (
            "개별 AI의 분석 JSON은 각 운동별 "
            "results 폴더에 별도로 저장됩니다."
        ),
    }

    timestamp = started_at.strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    result_path = OUTPUT / (
        f"{exercise['name']}_{timestamp}.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            run_record,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n================================")
    print("통합 실행 상태:", status)
    print("실행 기록:", result_path)
    print("================================")


# ==========================================
# 4. 메인 프로그램
# ==========================================

def main():

    while True:

        exercise = select_exercise()

        if exercise is None:
            print("\nSOLFAC 종료!")
            break

        run_exercise(exercise)


if __name__ == "__main__":
    main()
