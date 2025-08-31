"""Simple background worker to execute pretrain jobs placed as JSON files in model_store/jobs/.

Job format (JSON): {"type": "pretrain", "files": [...], "epochs": 1, "job_file": "path"}
The worker writes progress into the provided job_file (job_file path should be absolute).
"""
import time
from pathlib import Path
import json
import logging
from tradingbot_ibkr import job_db
from tradingbot_ibkr import notifier
import contextlib
import sys
import os
import time as _time


MODEL_STORE = Path(__file__).resolve().parents[0] / 'model_store'
JOBS_DIR = MODEL_STORE / 'jobs'
ARCHIVE_DIR = MODEL_STORE / 'jobs_archive'
JOBS_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = MODEL_STORE / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)


def run_job(job_path: Path, max_retries: int = 2):
    try:
        job = json.loads(job_path.read_text())
    except Exception as e:
        logging.exception('Failed to read job file, removing: %s', job_path)
        try:
            job_path.unlink()
        except Exception:
            pass
        return

    jid = job_path.stem
    job_db.create_job(jid, str(job_path))
    typ = job.get('type')
    attempt = 0
    success = False
    while attempt <= max_retries and not success:
        attempt += 1
        try:
            logging.info('Starting job %s attempt %d', jid, attempt)
            job_db.update_job_status(jid, 'running')
            if typ == 'pretrain':
                from pretrain_online_trainer import run_pretrain
                files = job.get('files') or []
                epochs = int(job.get('epochs', 1))
                job_file = job.get('job_file')
                # per-job log capture
                log_path = LOGS_DIR / f'{jid}.log'
                start_ts = _time.time()
                try:
                    with open(log_path, 'w', encoding='utf-8') as lf:
                        with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
                            res = run_pretrain(files, epochs=epochs, job_file=job_file)
                except Exception:
                    logging.exception('Error running pretrain for job %s', jid)
                    # ensure failure bubbles to outer except for retry handling
                    raise
                duration = _time.time() - start_ts
                # ensure JSON-serializable result
                try:
                    json.dumps(res)
                    serial = res
                except Exception:
                    serial = {'summary': str(res)}
                payload = {'summary': serial, 'duration': duration, 'log_path': str(log_path)}
                job_db.update_job_status(jid, 'done', progress=100, result=payload)
                try:
                    notifier.notify({'job_id': jid, 'status': 'done', 'result': payload})
                except Exception:
                    logging.exception('Failed to send completion notification for job %s', jid)
            success = True
            logging.info('Job %s completed', jid)
        except Exception as e:
            logging.exception('Job %s failed on attempt %d: %s', jid, attempt, e)
            job_db.update_job_status(jid, 'failed', progress=None, result={'error': str(e), 'attempt': attempt})
            if attempt > max_retries:
                try:
                    notifier.notify({'job_id': jid, 'status': 'failed', 'error': str(e), 'attempt': attempt})
                except Exception:
                    logging.exception('Failed to send failure notification for job %s', jid)
            time.sleep(2 ** attempt)

    # archive job file
    try:
        target = ARCHIVE_DIR / job_path.name
        job_path.replace(target)
        logging.info('Archived job file to %s', target)
    except Exception:
        try:
            job_path.unlink()
        except Exception:
            pass


def run_forever(poll_seconds=5):
    while True:
        for f in list(JOBS_DIR.iterdir()):
            if f.suffix == '.json':
                run_job(f)
        time.sleep(poll_seconds)


if __name__ == '__main__':
    job_db.init_db()
    run_forever()
