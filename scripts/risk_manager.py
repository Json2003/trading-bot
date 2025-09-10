#!/usr/bin/env python3
import os, time, json, datetime as dt

ART = os.getenv("ARTIFACTS_DIR","artifacts")
DAILY = float(os.getenv("DAILY_LOSS_HALT_PCT","-0.02"))
RUN_DD = float(os.getenv("RUNNING_DD_HALT_PCT","-0.08"))
MAX_LOSS_STREAK = int(os.getenv("MAX_CONSEC_LOSSES","4"))
SLACK = os.getenv("SLACK_WEBHOOK_URL","")

state = {"equity_hist": [], "loss_streak": 0, "halted": False}

def _post_slack(msg: str):
    if not SLACK:
        return
    # Lazy, guarded import to avoid local requests.py shadowing
    try:
        import importlib, sys as _sys
        try:
            req = importlib.import_module('requests')
            src = getattr(req, '__file__', '') or ''
            if os.path.abspath(os.getcwd()) in os.path.abspath(src):
                raise ImportError('shadowed')
        except Exception:
            original = _sys.path.copy()
            try:
                site = [p for p in original if ('site-packages' in (p or '')) or ('dist-packages' in (p or ''))]
                rest = [p for p in original if p not in site]
                _sys.path[:] = site + rest
                if 'requests' in _sys.modules:
                    del _sys.modules['requests']
                req = importlib.import_module('requests')
            finally:
                _sys.path[:] = original
        req.post(SLACK, json={"text": msg}, timeout=5)
    except Exception:
        pass

def notify(msg):
    print("[ALERT]", msg, flush=True)
    _post_slack(msg)

def _write_killswitch(reason: str):
    try:
        os.makedirs(ART, exist_ok=True)
        ks = os.path.join(ART, 'killswitch')
        with open(ks, 'w') as f:
            f.write(json.dumps({"ts": dt.datetime.utcnow().isoformat(), "reason": reason}))
    except Exception:
        pass

def check_killswitch():
    # Load latest portfolio equity (blend equal-weight from heartbeats if needed)
    # Here we just sum last_equity across crypto symbols as proxy and combine with equity rotation’s final equity
    ceq = 0.0; cpath = os.path.join(ART, "crypto_heartbeat.json")
    if os.path.exists(cpath):
        with open(cpath) as f: h = json.load(f)
        for row in h.get("equities", []): ceq += row.get("equity",1.0)
        if h.get("equities"): ceq /= max(len(h["equities"]),1)
    eeq = 0.0; epath = os.path.join(ART, "equity_rotation_financials_metrics.json")
    if os.path.exists(epath):
        with open(epath) as f: m = json.load(f); eeq = 1.0 + float(m.get("total_return",0.0))
    # crude combined equity (50/50)
    if ceq==0 and eeq==0: return
    eq = 0.5*(ceq if ceq>0 else 1.0) + 0.5*(eeq if eeq>0 else 1.0)

    ts = dt.datetime.utcnow().isoformat()
    state["equity_hist"].append({"ts": ts, "eq": eq})
    if len(state["equity_hist"])>1440: state["equity_hist"]=state["equity_hist"][-1440:]

    # daily change
    today = [x for x in state["equity_hist"] if x["ts"][:10]==ts[:10]]
    if today:
        open_eq = today[0]["eq"]; daily_ret = (eq/open_eq)-1.0
        if daily_ret <= DAILY and not state["halted"]:
            state["halted"]=True; notify(f"DAILY LOSS HALT {daily_ret:.2%}. Halting bots.")
            _write_killswitch('daily_loss')

    # running DD
    peak = max(x["eq"] for x in state["equity_hist"])
    dd = eq/peak - 1.0
    if dd <= RUN_DD and not state["halted"]:
        state["halted"]=True; notify(f"RUNNING DD HALT {dd:.2%}. Halting bots.")
        _write_killswitch('running_drawdown')

    # consecutive losses from crypto last trades
    h = None
    if os.path.exists(cpath):
        with open(cpath) as f: h=json.load(f)
    if h:
        pnl = 0.0
        for s in h.get("last_trades",[]):
            for tr in s.get("trades",[]):
                pnl += tr.get("pnl",0.0)
        if pnl < 0:
            state["loss_streak"] += 1
        else:
            state["loss_streak"] = 0
        if state["loss_streak"] >= MAX_LOSS_STREAK and not state["halted"]:
            state["halted"]=True; notify(f"LOSS STREAK HALT n={state['loss_streak']}. Halting bots.")
            _write_killswitch('loss_streak')

def main():
    notify("Risk manager started")
    while True:
        try:
            check_killswitch()
        except Exception as e:
            notify(f"Risk manager error: {e}")
        time.sleep(int(os.getenv('RISK_CHECK_SEC', '60')))

if __name__ == "__main__":
    main()
