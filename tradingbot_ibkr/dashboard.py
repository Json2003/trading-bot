from flask import Flask, render_template, jsonify, request
from pathlib import Path
import json
import threading
from pathlib import Path as P
from pathlib import Path as PP
import os
from functools import wraps
app = Flask(__name__, template_folder='templates', static_folder='static')
from flask import Response
from tradingbot_ibkr import job_db
import time

@app.route('/')
def index():
    # read model info if available
    model_dir = Path(__file__).resolve().parents[0] / 'models' / 'model_store'
    info = {}
    if model_dir.exists():
        for f in model_dir.iterdir():
            info[f.name] = str(f.stat().st_mtime)
    return render_template('index.html', models=info)

@app.route('/api/train', methods=['POST'])
def api_train():
    # trigger demo training (runs synchronously)
    from run_training_demo import run
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({'status': 'started'})


def _check_api_key():
    expected = os.environ.get('DASHBOARD_API_KEY')
    if not expected:
        return True
    # check header first
    from flask import request
    key = request.headers.get('X-API-KEY') or request.args.get('api_key')
    return key == expected


def require_auth(f):
    @wraps(f)
    def _wrapped(*args, **kwargs):
        if not _check_api_key():
            return jsonify({'error': 'unauthorized'}), 401
        return f(*args, **kwargs)
    return _wrapped


@app.route('/api/pretrain', methods=['POST'])
@require_auth
def api_pretrain():
    # enqueue a pretrain job for the worker
    data = request.get_json() or {}
    files = data.get('files', [])
    epochs = int(data.get('epochs', 1))
    model_store = P(__file__).resolve().parents[0] / 'model_store'
    jobs_dir = model_store / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job = {
        'type': 'pretrain',
        'files': files,
        'epochs': epochs,
    }
    # create a job file
    import uuid
    jid = uuid.uuid4().hex
    job_file = jobs_dir / f'{jid}.json'
    job['job_file'] = str(job_file)
    job_file.write_text(json.dumps(job))
    return jsonify({'status': 'queued', 'job_id': jid})


@app.route('/api/job_status')
@require_auth
def api_job_status():
    jid = request.args.get('job_id')
    if not jid:
        return jsonify({'error': 'job_id required'}), 400
    model_store = P(__file__).resolve().parents[0] / 'model_store'
    jobs_dir = model_store / 'jobs'
    job_file = jobs_dir / f'{jid}.json'
    if not job_file.exists():
        return jsonify({'status': 'not_found'})
    try:
        j = json.loads(job_file.read_text())
        return jsonify({'status': 'queued', 'job': j})
    except Exception:
        return jsonify({'status': 'error'})


@app.route('/api/job_record')
@require_auth
def api_job_record():
    jid = request.args.get('job_id')
    if not jid:
        return jsonify({'error': 'job_id required'}), 400
    job_db.init_db()
    rec = job_db.get_job(jid)
    if not rec:
        return jsonify({'status': 'not_found'})
    return jsonify({'status': 'ok', 'job': rec})


@app.route('/api/job_stream')
@require_auth
def api_job_stream():
    jid = request.args.get('job_id')
    if not jid:
        return jsonify({'error': 'job_id required'}), 400

    def gen():
        job_db.init_db()
        last = None
        while True:
            rec = job_db.get_job(jid)
            if rec != last:
                yield 'data: ' + json.dumps(rec) + '\n\n'
                last = rec
            if rec and rec.get('status') in ('done', 'failed'):
                break
            time.sleep(1)

    return Response(gen(), mimetype='text/event-stream')


@app.route('/api/model_version')
def api_model_version():
    model_dir = P(__file__).resolve().parents[0] / 'model_store'
    version_file = model_dir / 'model_version.json'
    if not version_file.exists():
        return jsonify({'model_version': None})
    try:
        data = json.loads(version_file.read_text())
        return jsonify({'model_version': data.get('last_version'), 'last_hash': data.get('last_hash')})
    except Exception:
        return jsonify({'model_version': None})

@app.route('/api/models')
def api_models():
    model_dir = Path(__file__).resolve().parents[0] / 'models' / 'model_store'
    out = []
    if model_dir.exists():
        for f in model_dir.iterdir():
            out.append({'name': f.name, 'mtime': f.stat().st_mtime})
    return jsonify(out)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
