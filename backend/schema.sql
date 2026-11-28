-- 1. 사용자 테이블
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY NOT NULL,
    nickname TEXT NOT NULL,
    created_at TEXT NOT NULL,
    consent_yn INTEGER NOT NULL DEFAULT 0
        CHECK (consent_yn IN (0, 1)),
    consented_at TEXT
);

-- 2. 운동 종류 테이블
CREATE TABLE IF NOT EXISTS exercises (
    exercise_id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    body_part TEXT,
    rule_version TEXT
);

-- 3. 운동 세션 테이블
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY NOT NULL,
    user_id TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'ready'
        CHECK (status IN ('ready', 'running', 'completed', 'error')),

    CHECK (
        ended_at IS NULL
        OR julianday(ended_at) >= julianday(started_at)
    ),

    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (exercise_id) REFERENCES exercises(exercise_id)
);


-- 4. 운동 결과 테이블
CREATE TABLE IF NOT EXISTS exercise_results (
    result_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL UNIQUE,
    rep_count INTEGER NOT NULL CHECK (rep_count >= 0),
    correct_rep_count INTEGER
        CHECK (correct_rep_count >= 0
               AND correct_rep_count <= rep_count),
    duration_sec REAL NOT NULL CHECK (duration_sec >= 0),
    created_at TEXT NOT NULL,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ON DELETE CASCADE
);


-- 5. 운동 영상 테이블
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    started_at TEXT NOT NULL,
    duration_sec REAL CHECK (duration_sec >= 0),
    deleted_at TEXT,
    deletion_status TEXT NOT NULL DEFAULT 'active'
        CHECK (deletion_status IN ('active', 'deleted', 'error')),

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ON DELETE CASCADE
);


-- 6. 자세 판정 기록 테이블
CREATE TABLE IF NOT EXISTS posture_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    rep_no INTEGER CHECK (rep_no >= 0),
    posture_status TEXT NOT NULL
        CHECK (posture_status IN (
            'correct', 'incorrect', 'undetermined'
        )),
    error_type TEXT,
    detected_at TEXT NOT NULL,
    feedback_msg TEXT,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ON DELETE CASCADE
);

-- 7. 운동 종목 기본 데이터 등록
INSERT OR IGNORE INTO exercises
    (exercise_id, name, body_part, rule_version)
VALUES
    ('elbow_flex_ext', '팔꿈치 굽혔다 펴기', 'upper', NULL),
    ('neck_side_pull', '목 옆으로 당기기', 'upper', NULL),
    ('seated_knee_ext', '앉아서 무릎 펴기', 'lower', NULL);