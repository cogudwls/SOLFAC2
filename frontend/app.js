// 백엔드 API 공통 요청 함수
const API_BASE = "http://127.0.0.1:8000/api";
const AI_BASE = "http://127.0.0.1:8001";

const AI_EXERCISES = new Set(["elbow_flex_ext", "seated_knee_ext", "neck_side_pull"]);
let aiRunning = false;
let aiFrameTask = null;
let aiResult = null;

async function aiRequest(path, options = {}) {
  const response = await fetch(`${AI_BASE}${path}`, options);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "AI 서버 요청 실패");
  }

  return data;
}

// AI 분석 세션 시작
async function startAI() {
  aiResult = null;

  await aiRequest(`/ai/${state.sessionId}/start?exercise_id=${encodeURIComponent(state.selectedExercise.id)}`, {
    method: "POST"
  });

  aiRunning = true;
  aiFrameTask = sendAIFrames();
}

// 웹캠 영상 프레임 전송
let warningUntil = 0;
let warningMessage = "";
const poseMessages = {
  ELB_A: "팔꿈치를 조금 더 굽혀 주세요.", ELB_B: "팔꿈치를 몸 가까이에 붙여 주세요.",
  KNEE_A: "상체를 똑바로 유지해 주세요.", KNEE_B: "허벅지를 의자에 붙여 주세요.",
  NECK_A: "몸은 그대로 두고 목만 기울여 주세요.", NECK_B: "어깨에 힘을 빼고 내려 주세요.",
  HOLD_FAILED: "다음 동작은 5초 이상 유지해 주세요.", REVIEW: "자세를 다시 확인해 주세요."
};
function showAIFeedback(a) {
  if (!a.landmarks?.length) {
    setFeedback(state.selectedExercise.outOfFrameMsg, true); return;
  }
  const label = a.feedback_label || (a.confidence >= .55 ? a.pose_label : "");
  if (poseMessages[label]) {
    warningMessage = poseMessages[label]; warningUntil = performance.now() + 3000;
  }
  if (performance.now() < warningUntil) setFeedback(warningMessage, true);
  else if (["ELB_COR", "KNEE_COR", "NECK_COR"].includes(label)) setFeedback("좋아요! 지금 자세를 유지해 주세요.", false);
  else setFeedback("자세를 확인하고 있습니다.", false);
}
function drawSkeleton(points, w, h) {
  const c = document.getElementById("skeletonCanvas");
  if (!c || !w || !h) return;
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  const edges = [[7,8],[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,31],[28,32]];
  const valid = i => points[i] && points[i].visibility >= .5;
  ctx.strokeStyle = "#2cf0a0"; ctx.lineWidth = Math.max(2,w/180); ctx.lineCap = "round";
  for (const [i,j] of edges) if (valid(i) && valid(j)) {
    ctx.beginPath(); ctx.moveTo(points[i].x*w,points[i].y*h); ctx.lineTo(points[j].x*w,points[j].y*h); ctx.stroke();
  }
  for (const i of new Set(edges.flat().concat([0]))) if (valid(i)) {
    ctx.beginPath(); ctx.arc(points[i].x*w,points[i].y*h,Math.max(3,w/110),0,Math.PI*2); ctx.fillStyle="#ffdc64";ctx.fill();
  }
}
function startDemo(id) {
  const video = document.getElementById("demoVideo");
  video.src = `assets/${id}.mp4`; video.play().catch(() => {});
}
function stopDemo() { document.getElementById("demoVideo")?.pause(); }

async function sendAIFrames() {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  let lastTimestamp = -1;

  while (aiRunning) {
    const video = els.webcamVideo;

    if (video.readyState >= 2 && video.videoWidth > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      ctx.drawImage(video, 0, 0);

      const blob = await new Promise(resolve =>
        canvas.toBlob(resolve, "image/jpeg", 0.85)
      );

      if (!aiRunning) break;

      if (blob) {
        const timestamp = Math.max(
          Math.floor(performance.now()),
          lastTimestamp + 1
        );

        lastTimestamp = timestamp;

        const formData = new FormData();
        formData.append("image", blob, "frame.jpg");
        formData.append("timestamp_ms", String(timestamp));

        try {
          const analysis = await aiRequest(`/ai/${state.sessionId}/frame`, {
            method: "POST",
            body: formData
          });
          if (analysis.live) {
            state.totalRepCount = analysis.live.total_reps;
            state.repCount = analysis.live.correct_reps;
            els.repCountDisplay.textContent = `${state.repCount}회`;
            document.getElementById("totalRepDisplay").textContent = `${state.totalRepCount}회`;
          }
          drawSkeleton(analysis.landmarks || [], canvas.width, canvas.height);
          showAIFeedback(analysis);
        } catch (err) {
          console.error("AI 프레임 분석 실패:", err);
        }
      }
    }

    await new Promise(resolve => setTimeout(resolve, 50));
  }
}

// AI 분석 종료 및 최종 결과 수신
async function finishAI() {
  aiRunning = false;

  if (aiFrameTask) {
    await aiFrameTask;
    aiFrameTask = null;
  }

  if (!aiResult) {
    aiResult = await aiRequest(
      `/ai/${state.sessionId}/finish`,
      { method: "POST" }
    );
  }

  return aiResult;
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers
    }
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "API 요청에 실패했습니다."
    );
  }

  return data;
}

/* =========================================
   1. 재활 운동 마스터 데이터 정의
   - upper: 상체 재활 운동 목록 (팔꿈치 굽혔다 펴기, 목 옆으로 당기기)
   - lower: 하체 재활 운동 목록 (앉아서 무릎 펴기)
   ========================================= */
const EXERCISES = {
  upper: [
    {
      id: "elbow_flex_ext",
      name: "팔꿈치 굽혔다 펴기",
      part: "상체/팔",
      view: "정면",
      icon: "💪",
      framingGuide: "상반신이 화면 중앙에 오도록 1.5m ~ 2.5m 거리를 유지해 주세요.",
      outOfFrameMsg: "상반신이 화면 안에 들어오도록 위치를 조정해 주세요.",
      mispostureMsg: "손목이 안쪽으로 꺾였어요. 손목과 팔을 일직선으로 펴주세요.",
      steps: [
        "허리를 곧게 펴세요.",
        "팔꿈치를 구부려 어깨 쪽으로 올렸다가 다시 폅니다."
      ]
    },
    {
      id: "neck_side_pull",
      name: "목 옆으로 당기기",
      part: "상체/목",
      view: "정면",
      icon: "🧘",
      framingGuide: "상반신이 화면 중앙에 오도록 1.5m ~ 2.5m 거리를 유지해 주세요.",
      outOfFrameMsg: "상반신이 화면 안에 들어오도록 위치를 조정해 주세요.",
      mispostureMsg: "목이 너무 과하게 꺾이지 않도록 하세요.",
      steps: [
        "시선을 정면에 두고 바르게 앉거나 서세요.",
        "어깨가 올라가지 않게 주의하며 목을 옆으로 천천히 기울입니다."
      ]
    }
  ],
  lower: [
    {
      id: "seated_knee_ext",
      name: "앉아서 무릎 펴기",
      part: "하체/무릎",
      view: "측면",
      icon: "🦵",
      framingGuide: "전신이 화면 중앙에 오도록 1.5m ~ 2.5m 거리를 유지해 주세요.",
      outOfFrameMsg: "전신이 화면 안에 들어오도록 위치를 조정해 주세요.",
      mispostureMsg: "허리를 굽히지 마세요.",
      steps: [
        "의자에 등을 기대지 않고 바르게 앉으세요.",
        "무릎을 천천히 펴서 올렸다가 내립니다."
      ]
    }
  ]
};

/* =========================================
   2. 로컬 스토리지 키 및 앱 상태(State) 관리
   ========================================= */
const STORAGE_KEY = "solfac2_rehab_history_v1"; // 운동 이력 저장용 localStorage 키

const state = {
  category: "upper",         // 현재 선택된 부위 탭 ("upper" | "lower")
  selectedExercise: null,  // 현재 선택된 운동 객체
  sessionId: null,         // 현재 진행 중인 운동 세션 ID
  repCount: 0,             // 정자세 수행 횟수
  totalRepCount: 0,        // 전체 운동 수행 횟수
  elapsedSec: 0,           // 경과 시간(초)
  postureFeedback: "",     // 최근 자세 피드백 문구
  isWarning: false,        // 경고 상태 여부
  timerId: null,           // 경과 시간 타이머(setInterval) ID
  stream: null,             // 웹캠 MediaStream 객체
  userId: null,
  users: []
};

/* =========================================
   3. DOM 요소 참조 모음 (els)
   ========================================= */
const els = {
  // 화면(Screen) 섹션
  scr01: document.getElementById("scr-01"),
  scr02: document.getElementById("scr-02"),
  scr03: document.getElementById("scr-03"),

  // SCR-01 (운동 시작 화면) 요소
  tabUpper: document.getElementById("tabUpper"),
  tabLower: document.getElementById("tabLower"),
  exerciseCardGrid: document.getElementById("exerciseCardGrid"),
  framingGuideBox: document.getElementById("framingGuideBox"),
  scr01Alert: document.getElementById("scr01Alert"),
  startWorkoutBtn: document.getElementById("startWorkoutBtn"),

  // SCR-02 (운동 진행 화면) 요소
  activeExerciseTitle: document.getElementById("activeExerciseTitle"),
  sessionIdLabel: document.getElementById("sessionIdLabel"),
  guideVisual: document.getElementById("guideVisual"),
  guideSteps: document.getElementById("guideSteps"),
  webcamVideo: document.getElementById("webcamVideo"),
  cameraErrorBox: document.getElementById("cameraErrorBox"),
  repCountDisplay: document.getElementById("repCountDisplay"),
  elapsedTimeDisplay: document.getElementById("elapsedTimeDisplay"),
  postureFeedbackDisplay: document.getElementById("postureFeedbackDisplay"),
  simNormalBtn: document.getElementById("simNormalBtn"),
  simMispostureBtn: document.getElementById("simMispostureBtn"),
  retryCamBtn: document.getElementById("retryCamBtn"),
  finishWorkoutBtn: document.getElementById("finishWorkoutBtn"),
  saveErrorBox: document.getElementById("saveErrorBox"),

  // SCR-03 (결과·이력 화면) 요소
  returnHomeBtn: document.getElementById("returnHomeBtn"),
  latestSummaryContent: document.getElementById("latestSummaryContent"),
  emptyHistoryMsg: document.getElementById("emptyHistoryMsg"),
  historyTableBody: document.getElementById("historyTableBody")
};

/* =========================================
   4. 화면 전환 및 SCR-01 렌더링 함수
   ========================================= */
// 전달받은 screenId에 해당하는 화면만 활성화(.active)
function showScreen(screenId) {
  [els.scr01, els.scr02, els.scr03].forEach((sec) => {
    sec.classList.toggle("active", sec.id === screenId);
  });
}

// 현재 선택된 부위(상체/하체)의 운동 카드를 화면에 렌더링
function renderExerciseCards() {
  const list = EXERCISES[state.category] || [];
  els.exerciseCardGrid.innerHTML = "";

  list.forEach((ex) => {
    const card = document.createElement("div");
    card.className = `exercise-card ${state.selectedExercise?.id === ex.id ? "selected" : ""}`;
    card.setAttribute("role", "button");
    card.tabIndex = 0; // 키보드 포커스 가능하도록 설정
    card.innerHTML = `
      <div class="thumb">${ex.icon}</div>
      <div class="name">${ex.name}</div>
      <div class="meta">[ ${ex.part} · ${ex.view} ]</div>
    `;

    // 카드 선택 시 상태 갱신 및 안내 문구 업데이트
    const selectHandler = () => {
      state.selectedExercise = ex;
      els.scr01Alert.textContent = "";
      els.framingGuideBox.textContent = ex.framingGuide;
      renderExerciseCards();
    };

    card.addEventListener("click", selectHandler);
    // Enter 또는 Space 키로도 카드 선택 지원 (접근성)
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectHandler();
      }
    });

    els.exerciseCardGrid.appendChild(card);
  });
}

// 상체/하체 재활 탭 전환 처리
function setCategory(cat) {
  state.category = cat;
  state.selectedExercise = null; // 부위 변경 시 기존 선택 운동 초기화
  els.scr01Alert.textContent = "";

  const isUpper = cat === "upper";
  els.tabUpper.classList.toggle("active", isUpper);
  els.tabLower.classList.toggle("active", !isUpper);
  els.tabUpper.setAttribute("aria-selected", String(isUpper));
  els.tabLower.setAttribute("aria-selected", String(!isUpper));

  // 부위에 따른 기본 촬영 거리/구도 안내 문구 설정
  els.framingGuideBox.textContent = isUpper
    ? "상반신이 화면 중앙에 오도록 1.5m ~ 2.5m 거리를 유지해 주세요."
    : "전신이 화면 중앙에 오도록 1.5m ~ 2.5m 거리를 유지해 주세요.";

  renderExerciseCards();
}

/* =========================================
   5. 웹캠 제어 및 자세 피드백 함수
   ========================================= */
// 전면 웹캠 스트림 시작
async function startCamera() {
  els.cameraErrorBox.classList.add("hidden");
  els.cameraErrorBox.textContent = "";
  stopCamera(); // 중복 실행 방지를 위해 기존 스트림 먼저 정리

  // 브라우저 getUserMedia 지원 여부 확인
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    els.cameraErrorBox.textContent = "카메라 연결을 확인해 주세요. (브라우저 카메라 미지원)";
    els.cameraErrorBox.classList.remove("hidden");
    return;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false
    });
    els.webcamVideo.srcObject = state.stream;
  } catch (err) {
    // 권한 거부 또는 장치 미연결 시 오류 메시지 표시
    els.cameraErrorBox.textContent = "카메라 연결을 확인해 주세요.";
    els.cameraErrorBox.classList.remove("hidden");
  }
}

// 웹캠 트랙 정지 및 비디오 소스 해제
function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  els.webcamVideo.srcObject = null;
  const overlay = document.getElementById("skeletonCanvas");
  overlay.getContext("2d").clearRect(0,0,overlay.width,overlay.height);
}

// 자세 피드백 문구 및 정상/경고 스타일 반영
function setFeedback(text, isWarning = false) {
  state.postureFeedback = text;
  state.isWarning = isWarning;
  els.postureFeedbackDisplay.textContent = text;
  document.getElementById("feedbackStatus").textContent = isWarning ? "자세 확인" : "운동 안내";
  els.postureFeedbackDisplay.classList.toggle("warning", isWarning);
}

/* =========================================
   6. 운동 세션 시작 (FUN-01 / SCR-02 진입)
   ========================================= */
async function handleStartWorkout() {
  if (!state.userId) {
    els.scr01Alert.textContent = "사용자를 먼저 선택해 주세요.";
    return;
  }
  // 운동 미선택 시 안내 메시지 출력 후 중단
  if (!state.selectedExercise) {
    els.scr01Alert.textContent = "수행할 재활 운동을 먼저 선택해 주세요.";
    return;
  }

  const ex = state.selectedExercise;

  els.startWorkoutBtn.disabled = true;

  try {
    const session = await apiRequest("/sessions", {
      method: "POST",
      body: JSON.stringify({
        user_id: state.userId,
        exercise_id: ex.id
      })
    });

    state.sessionId = session.session_id;

  } catch (err) {
    els.scr01Alert.textContent =
      "운동 세션 생성 실패: " + err.message;
    return;

  } finally {
    els.startWorkoutBtn.disabled = false;
  }
  
  aiResult = null;
  state.repCount = 0;
  state.totalRepCount = 0;
  state.elapsedSec = 0;
  els.saveErrorBox.classList.add("hidden");

  // SCR-02 UI에 선택 운동 가이드 및 초기 측정값 반영
  els.activeExerciseTitle.textContent = ex.name;
  els.sessionIdLabel.textContent = `세션: ${state.sessionId} · ${ex.part} (${ex.view})`;
  warningUntil = 0; warningMessage = "";
  startDemo(ex.id);
  els.guideSteps.innerHTML = ex.steps.map((s) => `<li>${s}</li>`).join("");
  els.repCountDisplay.textContent = "0회";
  document.getElementById("totalRepDisplay").textContent = "0회";
  els.simNormalBtn.disabled = AI_EXERCISES.has(ex.id);
  els.simMispostureBtn.disabled = AI_EXERCISES.has(ex.id);
  els.elapsedTimeDisplay.textContent = "0초";
  setFeedback(ex.outOfFrameMsg, false);

  // 운동 진행 화면으로 전환 후 카메라 시작
  showScreen("scr-02");
  await startCamera();

  // 선택 운동 AI 분석 시작
  if (AI_EXERCISES.has(ex.id) && state.stream) {
    try {
      await startAI();
    } catch (err) {
      els.saveErrorBox.textContent =
        "AI 연결 실패: " + err.message;
      els.saveErrorBox.classList.remove("hidden");
    }
  }

  // 1초 단위 경과 시간 타이머 시작
  clearInterval(state.timerId);
  state.timerId = setInterval(() => {
    state.elapsedSec += 1;
    els.elapsedTimeDisplay.textContent = `${state.elapsedSec}초`;
  }, 1000);
}

/* =========================================
   7. 운동 이력 저장 및 SCR-03 렌더링 함수
   ========================================= */
// localStorage에서 저장된 운동 이력 불러오기
function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

// 새 운동 기록을 이력 맨 앞에 추가하고 localStorage에 저장
function saveHistoryRecord(record) {
  const list = loadHistory();
  list.unshift(record);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  return list;
}

// Date 객체를 "YYYY년 MM월 DD일, HH시 MM분" 형식으로 변환
function formatKoreanDateTime(dateObj) {
  const y = dateObj.getFullYear();
  const m = String(dateObj.getMonth() + 1).padStart(2, "0");
  const d = String(dateObj.getDate()).padStart(2, "0");
  const hh = String(dateObj.getHours()).padStart(2, "0");
  const mm = String(dateObj.getMinutes()).padStart(2, "0");
  return `${y}년 ${m}월 ${d}일, ${hh}시 ${mm}분`;
}

// SCR-03 결과 요약 카드 및 전체 이력 테이블 렌더링
function renderResultScreen(latestRecord, allRecords) {
  const exIcon =
    Object.values(EXERCISES)
      .flat()
      .find((item) => item.name === latestRecord.exerciseName)?.icon || "🏃";

  // 방금 완료한 운동 요약 카드 출력
  els.latestSummaryContent.innerHTML = `
    <div class="summary-thumb">${exIcon}</div>
    <div>
      <div><strong>${latestRecord.exerciseName}</strong></div>
      <div class="summary-stats">
        <span>총 운동 시간: <strong>${latestRecord.durationSec}초</strong></span>
        <span>전체 운동 횟수: <strong>${latestRecord.totalReps}회</strong></span>
        <span>정자세로 수행한 횟수: <strong>${latestRecord.reps}회</strong></span>
      </div>
      <div style="margin-top: 8px;">
        <strong>자세 분석 결과:</strong> ${latestRecord.feedback}
      </div>
    </div>
  `;

  // 이력이 비어 있을 때 안내 박스 표시
  if (!allRecords.length) {
    els.emptyHistoryMsg.classList.remove("hidden");
    els.historyTableBody.innerHTML = "";
    return;
  }

  // 전체 운동 이력 테이블 행 생성
  els.emptyHistoryMsg.classList.add("hidden");
  els.historyTableBody.innerHTML = allRecords
    .map(
      (row) => `
      <tr>
        <td>${row.dateText}</td>
        <td>${row.exerciseName}</td>
        <td>${row.durationSec} sec</td>
        <td>${row.reps}회</td>
        <td>${row.feedback}</td>
      </tr>
    `
    )
    .join("");
}

/* =========================================
   8. 운동 종료 처리 (FUN-08 / SCR-03 이동)
   ========================================= */
   async function handleFinishWorkout() {
    els.finishWorkoutBtn.disabled = true;
    els.saveErrorBox.classList.add("hidden");
  
    try {
      // 선택 운동 AI 최종 분석 결과 수신
      if (AI_EXERCISES.has(state.selectedExercise.id)) {
        const ai = await finishAI();
    
        state.totalRepCount = ai.total_reps;
        state.repCount = ai.correct_reps;
    
        console.log("AI 최종 분석 결과:", ai);
      }
    
      // 실제 DB에 운동 종료 및 결과 저장
      const result = await apiRequest(
        `/sessions/${state.sessionId}/finish`, {
          method: "POST",
          body: JSON.stringify({
            rep_count: state.totalRepCount,
            correct_rep_count: state.repCount
          })
        }
      );
  
      clearInterval(state.timerId);
      state.timerId = null;
      stopCamera();
      stopDemo();
  
      const record = {
        sessionId: state.sessionId,
        dateText: formatKoreanDateTime(new Date()),
        exerciseName: state.selectedExercise.name,
        durationSec: result.duration_sec,
        reps: result.correct_rep_count,
        totalReps: result.rep_count,
        feedback: "운동 결과가 저장되었습니다."
      };
  
      renderResultScreen(record, [record]);
      showScreen("scr-03");
  
    } catch (err) {
      els.saveErrorBox.textContent =
        "운동 종료 처리 실패: " + err.message;
      els.saveErrorBox.classList.remove("hidden");
  
    } finally {
      els.finishWorkoutBtn.disabled = false;
    }
  }
  
/* =========================================
   9. 이벤트 리스너 등록 및 초기 화면 렌더링
   ========================================= */
// 상·하체 탭 및 시작·재연결 버튼 이벤트
els.tabUpper.addEventListener("click", () => setCategory("upper"));
els.tabLower.addEventListener("click", () => setCategory("lower"));
els.startWorkoutBtn.addEventListener("click", handleStartWorkout);
els.retryCamBtn.addEventListener("click", async () => {
  await startCamera();
  if (state.stream && !aiRunning && !aiResult && AI_EXERCISES.has(state.selectedExercise?.id)) {
    try { await startAI(); } catch (err) {
      els.saveErrorBox.textContent = "AI 연결 실패: " + err.message;
      els.saveErrorBox.classList.remove("hidden");
    }
  }
});

// 테스트용 시뮬레이션 버튼 (정상 동작 +1 / 오자세 피드백)
els.simNormalBtn.addEventListener("click", () => {
  if (AI_EXERCISES.has(state.selectedExercise?.id)) return;
  state.repCount += 1;
  els.repCountDisplay.textContent = `${state.repCount}회`;
  setFeedback("좋습니다! 자세를 유지하세요.", false);
});

els.simMispostureBtn.addEventListener("click", () => {
  if (AI_EXERCISES.has(state.selectedExercise?.id)) return;
  const msg = state.selectedExercise?.mispostureMsg || "자세를 바르게 유지해 주세요.";
  setFeedback(msg, true);
});

// 운동 종료 및 SCR-01 복귀 버튼 이벤트
els.finishWorkoutBtn.addEventListener("click", handleFinishWorkout);

els.returnHomeBtn.addEventListener("click", () => {
  state.selectedExercise = null;
  els.scr01Alert.textContent = "";
  renderExerciseCards();
  showScreen("scr-01");
});

// 앱 초기 진입 시 상체 재활 운동 카드 렌더링
renderExerciseCards();

// =========================================
// 사용자 관리 API 연동
// =========================================
const userSelect = document.getElementById("userSelect");
const registerUserBtn = document.getElementById("registerUserBtn");
const videoConsent = document.getElementById("videoConsent");
const userAlert = document.getElementById("userAlert");

// 선택된 사용자 정보 반영
function selectUser() {
  state.userId = userSelect.value || null;

  const user = state.users.find(
    (u) => u.user_id === state.userId
  );

  videoConsent.disabled = !user;
  videoConsent.checked = Boolean(user?.consent_yn);
  userAlert.textContent = "";
}

// DB에서 사용자 목록 조회
async function loadUsers(selectedId = state.userId) {
  try {
    state.users = await apiRequest("/users");

    userSelect.replaceChildren(
      new Option("사용자를 선택해 주세요.", "")
    );

    state.users.forEach((user) => {
      userSelect.add(
        new Option(user.nickname, user.user_id)
      );
    });

    userSelect.value = selectedId || "";
    selectUser();

  } catch (err) {
    userAlert.textContent =
      "사용자 목록 조회 실패: " + err.message;
  }
}

// 신규 사용자 등록
registerUserBtn.addEventListener("click", async () => {
  const nickname = prompt("사용자 이름을 입력해 주세요.");

  if (!nickname || !nickname.trim()) return;

  registerUserBtn.disabled = true;

  try {
    const user = await apiRequest("/users", {
      method: "POST",
      body: JSON.stringify({
        nickname: nickname.trim(),
        consent_yn: false
      })
    });

    await loadUsers(user.user_id);

  } catch (err) {
    userAlert.textContent =
      "사용자 등록 실패: " + err.message;
  } finally {
    registerUserBtn.disabled = false;
  }
});

userSelect.addEventListener("change", selectUser);

// 페이지 접속 시 사용자 목록 조회
loadUsers();

// 영상 저장 동의 변경
videoConsent.addEventListener("change", async () => {
  const userId = state.userId;
  if (!userId) return;

  const consent = videoConsent.checked;

  videoConsent.disabled = true;
  userSelect.disabled = true;
  registerUserBtn.disabled = true;

  try {
    await apiRequest(`/users/${userId}/consent`, {
      method: "PATCH",
      body: JSON.stringify({
        consent_yn: consent
      })
    });

    const user = state.users.find(
      (u) => u.user_id === userId
    );

    if (user) {
      user.consent_yn = consent;
    }

    userAlert.textContent = consent
      ? "영상 저장에 동의하였습니다."
      : "영상 저장 동의를 철회하였습니다.";

  } catch (err) {
    videoConsent.checked = !consent;
    userAlert.textContent =
      "동의 설정 변경 실패: " + err.message;

  } finally {
    videoConsent.disabled = false;
    userSelect.disabled = false;
    registerUserBtn.disabled = false;
  }
});

// DB에 등록된 운동 종목 조회
async function loadExercises() {
  try {
    const exercises = await apiRequest("/exercises");

    ["upper", "lower"].forEach((category) => {
      EXERCISES[category] = EXERCISES[category]
        .filter((ex) =>
          exercises.some((db) => db.exercise_id === ex.id)
        )
        .map((ex) => {
          const db = exercises.find(
            (item) => item.exercise_id === ex.id
          );

          return {
            ...ex,
            name: db.name
          };
        });
    });

    renderExerciseCards();

  } catch (err) {
    els.scr01Alert.textContent =
      "운동 종목 조회 실패: " + err.message;
  }
}

loadExercises();
