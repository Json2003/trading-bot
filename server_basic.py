from fastapi import FastAPI, WebSocket
from datetime import datetime, timezone
import asyncio, json

app = FastAPI()

STATE = {
    "running": False,
    "metrics": {
        "equity": 100000,
        "daily_pnl": 0,
        "win_rate": 0.55,
        "sharpe": 1.2,
        "drawdown": 0.05,
        "timestamp": datetime.now(timezone.utc).isoformat()
    },
    "positions": [],
    "orders": []
}

@app.get("/status")
def status():
    return {"running": STATE["running"]}

@app.post("/control/start")
def start():
    STATE["running"] = True
    return {"ok": True}

@app.post("/control/stop")
def stop():
    STATE["running"] = False
    return {"ok": True}

clients = []

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await asyncio.sleep(2)
            STATE["metrics"]["timestamp"] = datetime.now(timezone.utc).isoformat()
            await ws.send_text(json.dumps({"metrics": STATE["metrics"]}))
    except Exception:
        clients.remove(ws)
