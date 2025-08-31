"""Simple SQLite-backed job status store for pretrain jobs."""
import sqlite3
from pathlib import Path
import json
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[0]
DB_PATH = BASE / 'model_store' / 'jobs.db'
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS jobs (
      id TEXT PRIMARY KEY,
      job_file TEXT,
      status TEXT,
      created_at TEXT,
      started_at TEXT,
      finished_at TEXT,
      progress REAL,
      result_json TEXT
    )
    ''')
    conn.commit()
    conn.close()


def create_job(job_id: str, job_file: str):
    init_db()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('INSERT OR REPLACE INTO jobs (id, job_file, status, created_at) VALUES (?,?,?,?)',
                (job_id, job_file, 'queued', datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def update_job_status(job_id: str, status: str, progress: float = None, result: dict = None):
    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    if status == 'running':
        cur.execute('UPDATE jobs SET status=?, started_at=? WHERE id=?', (status, now, job_id))
    elif status in ('done', 'failed'):
        cur.execute('UPDATE jobs SET status=?, finished_at=?, progress=?, result_json=? WHERE id=?',
                    (status, now, progress, json.dumps(result) if result else None, job_id))
    else:
        cur.execute('UPDATE jobs SET status=?, progress=? WHERE id=?', (status, progress, job_id))
    conn.commit()
    conn.close()


def get_job(job_id: str):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM jobs WHERE id=?', (job_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)
