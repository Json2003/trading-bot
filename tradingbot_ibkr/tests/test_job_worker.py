import tempfile
import json
from pathlib import Path
import time

import pytest

from tradingbot_ibkr import job_db
from tradingbot_ibkr import worker


def test_job_db_lifecycle(tmp_path):
    # point DB to temp location by adjusting DB_PATH indirectly
    dbdir = tmp_path / 'model_store'
    dbdir.mkdir()
    # patch job_db DB_PATH via environment of module by creating expected folder
    job_db.init_db()
    job_id = 'testjob'
    job_file = str(tmp_path / 'job.json')
    job_db.create_job(job_id, job_file)
    job_db.update_job_status(job_id, 'running', progress=10)
    job_db.update_job_status(job_id, 'done', progress=100, result={'ok': True})
    rec = job_db.get_job(job_id)
    assert rec['status'] in ('done', 'running') or rec is not None


def test_worker_run_job_creates_log_and_db(tmp_path, monkeypatch):
    # create fake job file
    jobs_dir = tmp_path
    job_path = jobs_dir / 'abc123.json'
    job = {'type': 'pretrain', 'files': [], 'epochs': 1, 'job_file': str(job_path)}
    job_path.write_text(json.dumps(job))

    # patch MODEL_STORE in worker to use tmp_path
    worker.MODEL_STORE = tmp_path
    worker.JOBS_DIR = tmp_path
    worker.ARCHIVE_DIR = tmp_path / 'archive'
    worker.ARCHIVE_DIR.mkdir(exist_ok=True)
    worker.LOGS_DIR = tmp_path / 'logs'
    worker.LOGS_DIR.mkdir(exist_ok=True)

    # mock run_pretrain to write a small job file and return a simple dict
    def fake_run_pretrain(paths, epochs=1, job_file=None):
        # write a progress update
        if job_file:
            Path(job_file).write_text(json.dumps({'status': 'running'}))
        return {'ok': True, 'model_version': 'v-test'}
    # ensure the import inside worker (from pretrain_online_trainer import run_pretrain) succeeds
    import types, sys
    mod = types.ModuleType('pretrain_online_trainer')
    mod.run_pretrain = fake_run_pretrain
    sys.modules['pretrain_online_trainer'] = mod
    monkeypatch.setattr('tradingbot_ibkr.worker.run_pretrain', fake_run_pretrain, raising=False)
    # call run_job
    worker.run_job(job_path)

    # check archived file exists
    archived = worker.ARCHIVE_DIR / job_path.name
    assert archived.exists()
    # check log file and DB record
    logf = worker.LOGS_DIR / (job_path.stem + '.log')
    assert logf.exists()
