"""Simple notifier: webhook POST and SMTP email for alerts.

Configure via environment variables:
 - ALERT_WEBHOOK_URL (POST JSON)
 - ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_FROM, ALERT_SMTP_TO
"""
import os
import json
import requests
import smtplib
from email.message import EmailMessage
from datetime import datetime
from typing import Dict, Any, List


def send_webhook(alerts):
    url = os.environ.get('ALERT_WEBHOOK_URL')
    if not url:
        return False
    payload = {'alerts': alerts}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print('Webhook send failed:', e)
        return False


def send_email(alerts, subject='TradingBot Alerts'):
    host = os.environ.get('ALERT_SMTP_HOST')
    port = int(os.environ.get('ALERT_SMTP_PORT', '25'))
    frm = os.environ.get('ALERT_SMTP_FROM')
    to = os.environ.get('ALERT_SMTP_TO')
    if not host or not frm or not to:
        return False
    body = json.dumps(alerts, indent=2)
    msg = EmailMessage()
    msg['From'] = frm
    msg['To'] = to
    msg['Subject'] = subject
    msg.set_content(body)
    try:
        s = smtplib.SMTP(host, port, timeout=10)
        s.send_message(msg)
        s.quit()
        return True
    except Exception as e:
        print('Email send failed:', e)
        return False


# note: unified notify implementation is below (supports Slack/webhook/email)


def send_slack_block(job_alert: dict):
    """Send a Slack Block message using ALERT_SLACK_WEBHOOK.

    job_alert should contain at least 'job_id' and 'status', and may include 'result' with duration/log_path/model_version.
    """
    url = os.environ.get('ALERT_SLACK_WEBHOOK')
    if not url:
        return False
    job_id = job_alert.get('job_id')
    status = job_alert.get('status')
    result = job_alert.get('result') or {}
    duration = result.get('duration')
    model_version = result.get('summary', {}).get('model_version') if isinstance(result.get('summary'), dict) else None
    log_path = result.get('log_path')

    job_id_str = str(job_id) if job_id is not None else 'unknown'
    status_str = str(status).upper() if status is not None else 'UNKNOWN'
    blocks: List[Dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Job:* {job_id}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Status:* {status}"}},
    ]
    if model_version:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"Model version: `{model_version}`"}})
    if duration:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"Duration: {duration:.1f}s"}]})
    if log_path:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"Log: `{log_path}`"}})

    job_alert: Dict[str, Any] = {"blocks": blocks}
    payload = {"blocks": blocks}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print('Slack send failed:', e)
        return False


def notify(alerts):
    """Enhanced notify: supports webhook, email, and Slack block webhook."""
    sent = False
    # If alerts is a dict representing a single job alert, try Slack block
    if isinstance(alerts, dict) and 'job_id' in alerts:
        if send_slack_block(alerts):
            sent = True
    # generic webhook/email
    if send_webhook(alerts):
        sent = True
    if send_email(alerts):
        sent = True
    return sent
