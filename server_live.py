import asyncio
import json
import logging
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
)
logger = logging.getLogger("server_live")

app = FastAPI(title="Trading Bot Live Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dashboard_dir = Path(__file__).parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")

MODEL_STORE = Path("tradingbot_ibkr") / "model_store"
MODEL_STORE.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = MODEL_STORE / "settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "PAPER": True,
    "CONTINUOUS_BACKTEST": False,
    "STRATEGY": "sma_cross",
    "BACKTEST_INTERVAL": 60,
    "RISK_PCT": 0.01,
    "STOP_LOSS_PCT": 0.02,
    "TAKE_PROFIT_PCT": 0.04,
    "EXCHANGE": "binance",
}


def load_settings() -> Dict[str, Any]:
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text())
            return {**DEFAULT_SETTINGS, **data}
    except Exception as e:
        logger.warning("Failed to load settings: %s", e)
    return dict(DEFAULT_SETTINGS)


def save_settings(data: Dict[str, Any]) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.error("Failed to save settings: %s", e)


STATE: Dict[str, Any] = {
    "running": False,
    "settings": load_settings(),
    "metrics": {
        "equity": 100000.0,
        "daily_pnl": 0.0,
        "sharpe": 1.2,
        "drawdown": 0.05,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": 0.0,
        "total_trades": 0,
        "active_positions": 0,
    },
    "markets": [],
    "news": [],
    "trades_current": [],
    "trades_proposed": [],
    "activity": [],
    "server_stats": {
        "start_time": time.time(),
        "active_connections": 0,
        "total_connections": 0,
        "requests_per_minute": 0,
        "error_count": 0,
    },
}

security = HTTPBearer(auto_error=False)

fake_users_db = {
    "admin": {"username": "admin", "hashed_password": "admin123_hashed", "permissions": ["read", "write", "control"]},
    "readonly": {"username": "readonly", "hashed_password": "readonly123_hashed", "permissions": ["read"]},
}


def verify_password(plain: str, hashed: str) -> bool:
    return f"{plain}_hashed" == hashed


def create_access_token(username: str, minutes: int = 30) -> str:
    exp = int((datetime.utcnow() + timedelta(minutes=minutes)).timestamp())
    return f"token_{username}_{exp}"


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        return None
    token = credentials.credentials
    if token.startswith("token_"):
        try:
            _, username, _ = token.split("_", 2)
            return fake_users_db.get(username)
        except Exception:
            pass
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def require_permission(permission: str):
    async def wrapper(user=Depends(get_current_user)):
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        if permission not in user.get("permissions", []):
            raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")
        return user

    return wrapper


@app.post("/auth/login")
async def login(payload: Dict[str, str]):
    username = payload.get("username", "")
    password = payload.get("password", "")
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(username)
    return {"access_token": token, "token_type": "bearer"}


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            "client_id": f"ws-{random.randint(1000, 9999)}",
            "connected_at": time.time(),
        }
        STATE["server_stats"]["active_connections"] = len(self.active_connections)
        STATE["server_stats"]["total_connections"] += 1

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_info.pop(websocket, None)
            STATE["server_stats"]["active_connections"] = len(self.active_connections)

    async def broadcast(self, data: Dict[str, Any]):
        if not self.active_connections:
            return
        msg = json.dumps(data)
        stale = []
        for ws in self.active_connections:
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


manager = ConnectionManager()

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


def _rand_price(base: float, vol: float = 0.02) -> float:
    return round(base * (1 + random.uniform(-vol, vol)), 2)


def simulate_markets() -> List[Dict[str, Any]]:
    base_prices = {
        "BTC/USDT": 68000.0,
        "ETH/USDT": 3500.0,
        "SOL/USDT": 160.0,
        "BNB/USDT": 580.0,
        "XRP/USDT": 0.58,
    }
    markets = []
    for sym in SYMBOLS:
        last = _rand_price(base_prices[sym])
        bid = round(last * (1 - random.uniform(0.0005, 0.002)), 2)
        ask = round(last * (1 + random.uniform(0.0005, 0.002)), 2)
        chg = round(random.uniform(-3, 3), 2)
        markets.append({"symbol": sym, "bid": bid, "ask": ask, "last": last, "change_pct": chg})
    return markets


def simulate_news() -> Optional[Dict[str, Any]]:
    headlines = [
        "Fed hints at policy shift amid inflation cool-down",
        "Binance expands services in new markets",
        "ETF inflows hit record highs for crypto funds",
        "Major protocol announces L2 scaling upgrade rollout",
        "Whale activity spikes across altcoins",
    ]
    if random.random() < 0.4:
        return None
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "headline": random.choice(headlines),
        "source": random.choice(["Reuters", "Bloomberg", "CoinDesk", "WSJ"]),
    }


def simulate_trades() -> Dict[str, List[Dict[str, Any]]]:
    current = []
    for sym in random.sample(SYMBOLS, k=random.randint(0, 3)):
        current.append(
            {
                "symbol": sym,
                "side": random.choice(["long", "short"]),
                "qty": round(random.uniform(0.1, 3.0), 3),
                "entry": round(random.uniform(50, 50000), 2),
                "pnl": round(random.uniform(-300, 500), 2),
            }
        )
    proposed = []
    for sym in random.sample(SYMBOLS, k=random.randint(1, 3)):
        proposed.append(
            {
                "symbol": sym,
                "side": random.choice(["buy", "sell"]),
                "confidence": round(random.uniform(0.5, 0.95), 2),
                "reason": random.choice(["sma_cross", "breakout", "rsi_rebound", "enhanced"]),
            }
        )
    return {"current": current, "proposed": proposed}


def log_activity(msg: str) -> None:
    STATE["activity"].append({"ts": datetime.now(timezone.utc).isoformat(), "message": msg})
    STATE["activity"] = STATE["activity"][-100:]


@app.get("/health")
async def health():
    uptime = time.time() - STATE["server_stats"]["start_time"]
    return {"status": "healthy", "uptime_seconds": uptime, "active_connections": len(manager.active_connections)}


@app.get("/metrics")
async def metrics():
    STATE["metrics"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    STATE["metrics"]["uptime_seconds"] = time.time() - STATE["server_stats"]["start_time"]
    return STATE["metrics"]


@app.get("/settings")
async def get_settings():
    return {"settings": STATE["settings"]}


@app.post("/settings")
async def update_settings(payload: Dict[str, Any]):
    allowed = set(DEFAULT_SETTINGS.keys())
    updates = {k: v for k, v in payload.items() if k in allowed}
    for k, default in DEFAULT_SETTINGS.items():
        if k in updates:
            v = updates[k]
            try:
                if isinstance(default, bool):
                    updates[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                elif isinstance(default, int):
                    updates[k] = int(v)
                elif isinstance(default, float):
                    updates[k] = float(v)
                else:
                    updates[k] = str(v)
            except Exception:
                updates[k] = default
    STATE["settings"].update(updates)
    save_settings(STATE["settings"]) 
    await manager.broadcast({"event": "settings_update", "settings": STATE["settings"]})
    return {"ok": True, "updated": updates}


@app.post("/control/start")
async def start_bot(user=Depends(require_permission("control"))):
    STATE["running"] = True
    log_activity(f"Bot started by {user['username']}")
    await manager.broadcast({"event": "bot_started", "user": user["username"]})
    return {"ok": True}


@app.post("/control/stop")
async def stop_bot(user=Depends(require_permission("control"))):
    STATE["running"] = False
    log_activity(f"Bot stopped by {user['username']}")
    await manager.broadcast({"event": "bot_stopped", "user": user["username"]})
    return {"ok": True}


@app.get("/feed/markets")
async def feed_markets():
    return {"markets": STATE["markets"]}


@app.get("/feed/news")
async def feed_news():
    return {"news": STATE["news"]}


@app.get("/trades/current")
async def trades_current():
    return {"trades": STATE["trades_current"]}


@app.get("/trades/proposed")
async def trades_proposed():
    return {"trades": STATE["trades_proposed"]}


@app.get("/activity/log")
async def activity_log():
    return {"activity": STATE["activity"]}


@app.get("/")
async def root():
    index_file = dashboard_dir / "manage.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/health", status_code=307)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_text(
            json.dumps(
                {
                    "event": "connection_established",
                    "metrics": STATE["metrics"],
                    "settings": STATE["settings"],
                    "markets": STATE["markets"],
                    "news": STATE["news"],
                    "trades_current": STATE["trades_current"],
                    "trades_proposed": STATE["trades_proposed"],
                    "activity": STATE["activity"],
                }
            )
        )
        while True:
            STATE["metrics"]["timestamp"] = datetime.now(timezone.utc).isoformat()
            STATE["metrics"]["uptime_seconds"] = time.time() - STATE["server_stats"]["start_time"]
            await ws.send_text(
                json.dumps(
                    {
                        "event": "tick",
                        "metrics": STATE["metrics"],
                        "markets": STATE["markets"],
                        "news": STATE["news"],
                        "trades_current": STATE["trades_current"],
                        "trades_proposed": STATE["trades_proposed"],
                        "activity": STATE["activity"],
                    }
                )
            )
            await asyncio.sleep(2)
    except Exception:
        pass
    finally:
        manager.disconnect(ws)


@app.on_event("startup")
async def on_startup():
    async def updater():
        while True:
            try:
                STATE["markets"] = simulate_markets()
                n = simulate_news()
                if n:
                    STATE["news"].append(n)
                    STATE["news"] = STATE["news"][-50:]
                tr = simulate_trades()
                STATE["trades_current"] = tr["current"]
                STATE["trades_proposed"] = tr["proposed"]
                if random.random() < 0.4:
                    log_activity(random.choice([
                        "heartbeat ok",
                        "rebalanced risk budget",
                        "refreshed signals",
                        "synced orders",
                        "updated exposure limits",
                    ]))
                await asyncio.sleep(3)
            except Exception as e:
                logger.error("Updater error: %s", e)
                await asyncio.sleep(5)

    asyncio.create_task(updater())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
