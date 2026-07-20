CREATE TABLE IF NOT EXISTS app_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    dataset_generation INTEGER NOT NULL,
    model_generation INTEGER NOT NULL,
    current_release_id TEXT NOT NULL,
    admin_password_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS installations (
    installation_id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL,
    app_release_id TEXT NOT NULL,
    model_generation INTEGER NOT NULL,
    client_platform TEXT NOT NULL DEFAULT 'desktop',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_seen_at TEXT
);

CREATE TABLE IF NOT EXISTS release_batches (
    batch_id TEXT PRIMARY KEY,
    model_generation INTEGER NOT NULL UNIQUE,
    dataset_generation INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('reserved', 'partial', 'published', 'expired')
    ),
    reserved_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    published_at TEXT
);

CREATE TABLE IF NOT EXISTS platform_releases (
    release_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    platform TEXT NOT NULL CHECK (platform IN ('desktop', 'android')),
    version_code INTEGER NOT NULL,
    version_name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0 CHECK (is_current IN (0, 1)),
    FOREIGN KEY (batch_id) REFERENCES release_batches(batch_id),
    UNIQUE (batch_id, platform)
);

CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id TEXT NOT NULL UNIQUE,
    installation_id TEXT NOT NULL,
    water_type TEXT NOT NULL,
    mode TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    app_release_id TEXT NOT NULL,
    model_generation INTEGER NOT NULL,
    dataset_generation INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL
    ,FOREIGN KEY (installation_id) REFERENCES installations(installation_id)
);

CREATE TABLE IF NOT EXISTS tube_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id TEXT NOT NULL,
    tube_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id) ON DELETE CASCADE,
    UNIQUE (upload_id, tube_index)
);

CREATE TABLE IF NOT EXISTS desktop_releases (
    release_id TEXT PRIMARY KEY,
    model_generation INTEGER NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS admin_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    source_ip TEXT,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS login_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_ip TEXT NOT NULL,
    successful INTEGER NOT NULL,
    attempted_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_uploads_generation_water
ON uploads(dataset_generation, water_type);

CREATE INDEX IF NOT EXISTS idx_login_attempts_ip_time
ON login_attempts(source_ip, attempted_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_release_current
ON platform_releases(platform) WHERE is_current = 1;
