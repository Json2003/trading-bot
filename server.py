import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import random
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

APP_BRAND = os.getenv("APP_BRAND", "Splitstar Operations Console")
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "45"))
PBKDF2_ITERATIONS = 390_000

BASE_DIR = Path(__file__).parent
DASHBOARD_DIR = BASE_DIR / "dashboard"

MODEL_STORE = BASE_DIR / "tradingbot_ibkr" / "model_store"
MODEL_STORE.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = MODEL_STORE / "settings.json"
SECRET_FILE = MODEL_STORE / "secret.key"
USERS_FILE = MODEL_STORE / "users.json"

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

DEFAULT_METRICS: Dict[str, Any] = {
    "equity": 100_000.0,
    "daily_pnl": 0.0,
    "sharpe": 1.2,
    "drawdown": 0.04,
    "total_trades": 0,
    "active_positions": 0,
    "uptime_seconds": 0.0,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "status": "idle",
}

DEFAULT_ACCOUNT: Dict[str, Any] = {
    "equity": 100_000.0,
    "daily_pnl": 0.0,
    "total_pnl": 0.0,
    "total_trades": 0,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
)
logger = logging.getLogger("server")

app = FastAPI(title=f"{APP_BRAND} API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

security = HTTPBearer(auto_error=False)


def _urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _urlsafe_b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def load_secret_key() -> bytes:
    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes()
    key = secrets.token_bytes(32)
    SECRET_FILE.write_bytes(key)
    return key


SECRET_KEY = load_secret_key()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def load_settings() -> Dict[str, Any]:
    try:
        if SETTINGS_FILE.exists():
            stored = json.loads(SETTINGS_FILE.read_text())
            if isinstance(stored, dict):
                return {**DEFAULT_SETTINGS, **stored}
    except Exception as exc:
        logger.warning("Failed to load settings fallback to defaults: %s", exc)
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: Dict[str, Any]) -> None:
    try:
        _write_json(SETTINGS_FILE, settings)
    except Exception as exc:
        logger.error("Failed to persist settings: %s", exc)


def create_user_record(username: str, password: str, permissions: List[str]) -> Dict[str, Any]:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return {
        "username": username,
        "salt": _urlsafe_b64encode(salt),
        "hash": _urlsafe_b64encode(digest),
        "iterations": PBKDF2_ITERATIONS,
        "permissions": permissions,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_users() -> Dict[str, Any]:
    if USERS_FILE.exists():
        try:
            loaded = json.loads(USERS_FILE.read_text())
            if isinstance(loaded, dict):
                return loaded
        except Exception as exc:
            logger.error("User database unreadable, recreating: %s", exc)
    fallback_password = os.getenv("APP_ADMIN_PASSWORD", "change-me-now!")
    if fallback_password == "change-me-now!":
        logger.warning("Default admin password in use; set APP_ADMIN_PASSWORD for production.")
    admin_record = create_user_record("admin", fallback_password, ["read", "write", "control"])
    payload = {"admin": admin_record}
    _write_json(USERS_FILE, payload)
    return payload


USERS: Dict[str, Dict[str, Any]] = load_users()


def verify_password(password: str, user_record: Dict[str, Any]) -> bool:
    try:
        salt = _urlsafe_b64decode(user_record["salt"])
        expected = _urlsafe_b64decode(user_record["hash"])
        iterations = int(user_record.get("iterations") or PBKDF2_ITERATIONS)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(digest, expected)
    except Exception as exc:
        logger.error("Password verification failed: %s", exc)
        return False


def create_access_token(username: str, minutes: int = ACCESS_TOKEN_MINUTES) -> str:
    payload = {
        "sub": username,
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=minutes)).timestamp()),
    }
    body = _urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = _urlsafe_b64encode(hmac.new(SECRET_KEY, body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{signature}"


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        body, signature = token.split(".", 1)
        expected_sig = _urlsafe_b64encode(hmac.new(SECRET_KEY, body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected_sig):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")
        payload = json.loads(_urlsafe_b64decode(body))
        expiry = int(payload["exp"])
        if expiry < int(datetime.now(timezone.utc).timestamp()):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    payload = decode_access_token(credentials.credentials)
    username = payload.get("sub")
    user_record = USERS.get(username or "")
    if not user_record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown user")
    return user_record


def require_permission(permission: str):
    async def wrapper(user=Depends(get_current_user)):
        if permission not in user.get("permissions", []):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required")
        return user

    return wrapper


STATE: Dict[str, Any] = {
    "running": False,
    "settings": load_settings(),
    "metrics": dict(DEFAULT_METRICS),
    "account": dict(DEFAULT_ACCOUNT),
    "markets": [],
    "news": [],
    "trades_current": [],
    "trades_proposed": [],
    "activity": [],
    "server_stats": {
        "start_time": time.time(),
        "active_connections": 0,
        "total_connections": 0,
        "error_count": 0,
    },
}

RUN_EVENT = asyncio.Event()


def log_activity(message: str) -> None:
    STATE["activity"].append({"ts": datetime.now(timezone.utc).isoformat(), "message": message})
    STATE["activity"] = STATE["activity"][-100:]


def _refresh_run_event() -> None:
    should_run = STATE["running"] or STATE["settings"].get("CONTINUOUS_BACKTEST", False)
    if should_run:
        RUN_EVENT.set()
        STATE["metrics"]["status"] = "running" if STATE["running"] else "auto"
    else:
        RUN_EVENT.clear()
        STATE["metrics"]["status"] = "idle"


def _snapshot(event: str) -> Dict[str, Any]:
    metrics = dict(STATE["metrics"])
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    metrics["uptime_seconds"] = time.time() - STATE["server_stats"]["start_time"]
    metrics["running"] = STATE["running"]
    server_stats = {
        "uptime_seconds": metrics["uptime_seconds"],
        "active_connections": STATE["server_stats"]["active_connections"],
        "total_connections": STATE["server_stats"]["total_connections"],
        "error_count": STATE["server_stats"]["error_count"],
        "running": STATE["running"],
    }
    return {
        "event": event,
        "brand": APP_BRAND,
        "running": STATE["running"],
        "metrics": metrics,
        "account": STATE["account"],
        "settings": STATE["settings"],
        "markets": STATE["markets"],
        "news": STATE["news"],
        "trades_current": STATE["trades_current"],
        "trades_proposed": STATE["trades_proposed"],
        "activity": STATE["activity"],
        "server_stats": server_stats,
    }


async def broadcast_state(event: str) -> None:
    await manager.broadcast(_snapshot(event))


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self.connection_meta: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_meta[websocket] = {
            "client_id": f"ws-{random.randint(1000, 9999)}",
            "connected_at": time.time(),
        }
        STATE["server_stats"]["active_connections"] = len(self.active_connections)
        STATE["server_stats"]["total_connections"] += 1

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_meta.pop(websocket, None)
            STATE["server_stats"]["active_connections"] = len(self.active_connections)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        if not self.active_connections:
            return
        message = json.dumps(payload)
        stale: List[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_text(message)
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
        "BTC/USDT": 68_000.0,
        "ETH/USDT": 3_500.0,
        "SOL/USDT": 160.0,
        "BNB/USDT": 580.0,
        "XRP/USDT": 0.58,
    }
    markets = []
    for sym in SYMBOLS:
        last = _rand_price(base_prices[sym], vol=0.03)
        bid = round(last * (1 - random.uniform(0.0008, 0.0025)), 2)
        ask = round(last * (1 + random.uniform(0.0008, 0.0025)), 2)
        change = round(random.uniform(-2.5, 2.5), 2)
        markets.append({"symbol": sym, "bid": bid, "ask": ask, "last": last, "change_pct": change})
    return markets


def simulate_news() -> Optional[Dict[str, Any]]:
    headlines = [
        "Fed signals patience as inflation cools",
        "Major exchange launches new derivatives suite",
        "Large asset manager adds crypto exposure",
        "Protocol upgrade clears final governance vote",
        "Whale rotations prompt volatility spike",
    ]
    if random.random() < 0.5:
        return None
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "headline": random.choice(headlines),
        "source": random.choice(["Reuters", "Bloomberg", "CoinDesk", "WSJ"]),
    }


def simulate_trades() -> Dict[str, List[Dict[str, Any]]]:
    current: List[Dict[str, Any]] = []
    for sym in random.sample(SYMBOLS, k=random.randint(0, 3)):
        current.append(
            {
                "symbol": sym,
                "side": random.choice(["long", "short"]),
                "qty": round(random.uniform(0.1, 3.0), 3),
                "entry": round(random.uniform(50, 50_000), 2),
                "pnl": round(random.uniform(-250, 400), 2),
            }
        )
    proposed: List[Dict[str, Any]] = []
    for sym in random.sample(SYMBOLS, k=random.randint(1, 3)):
        proposed.append(
            {
                "symbol": sym,
                "side": random.choice(["buy", "sell"]),
                "confidence": round(random.uniform(0.45, 0.95), 2),
                "reason": random.choice(["sma_cross", "breakout", "rsi_rebound", "volatility_shift"]),
            }
        )
    return {"current": current, "proposed": proposed}


def update_account_snapshot() -> None:
    account = STATE["account"]
    metrics = STATE["metrics"]
    drift = random.uniform(-0.6, 0.85)
    account["daily_pnl"] = round(account.get("daily_pnl", 0.0) + drift, 2)
    account["total_pnl"] = round(account.get("total_pnl", 0.0) + drift, 2)
    account["equity"] = round(100_000.0 + account["total_pnl"], 2)
    account["total_trades"] = metrics.get("total_trades", 0)
    account["timestamp"] = datetime.now(timezone.utc).isoformat()


async def simulator_loop() -> None:
    while True:
        await RUN_EVENT.wait()
        sleep_for = max(2, min(int(STATE["settings"].get("BACKTEST_INTERVAL", 30)), 15))
        try:
            STATE["markets"] = simulate_markets()
            maybe_news = simulate_news()
            if maybe_news:
                STATE["news"].append(maybe_news)
                STATE["news"] = STATE["news"][-50:]
            trades = simulate_trades()
            STATE["trades_current"] = trades["current"]
            STATE["trades_proposed"] = trades["proposed"]
            STATE["metrics"]["equity"] = sum(item["last"] for item in STATE["markets"]) * 10
            STATE["metrics"]["daily_pnl"] = round(random.uniform(-500, 750), 2)
            STATE["metrics"]["sharpe"] = round(random.uniform(0.8, 2.1), 2)
            STATE["metrics"]["drawdown"] = round(random.uniform(0.01, 0.08), 4)
            STATE["metrics"]["total_trades"] = len(STATE["trades_current"]) + random.randint(0, 12)
            STATE["metrics"]["active_positions"] = len(STATE["trades_current"])
            STATE["metrics"]["status"] = "running"
            update_account_snapshot()
            await broadcast_state("tick")
        except Exception as exc:
            logger.error("Simulation loop error: %s", exc)
            STATE["server_stats"]["error_count"] += 1
            await asyncio.sleep(5)
            continue
        await asyncio.sleep(sleep_for)


@app.post("/auth/login")
async def login(payload: Dict[str, str]):
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    user_record = USERS.get(username)
    if not user_record or not verify_password(password, user_record):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token = create_access_token(username)
    logger.info("User %s authenticated", username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_MINUTES * 60,
        "user": {"username": username, "permissions": user_record["permissions"]},
    }


@app.get("/health")
async def health():
    uptime = time.time() - STATE["server_stats"]["start_time"]
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "active_connections": STATE["server_stats"]["active_connections"],
        "running": STATE["running"],
        "brand": APP_BRAND,
    }


@app.get("/metrics")
async def metrics():
    snap = _snapshot("metrics")
    return snap["metrics"]


@app.get("/account")
async def account():
    return STATE["account"]


@app.get("/settings")
async def get_settings():
    return {"settings": STATE["settings"]}


@app.post("/settings")
async def update_settings(
    payload: Dict[str, Any],
    user=Depends(require_permission("write")),
):
    allowed = set(DEFAULT_SETTINGS.keys())
    updates = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        default_value = DEFAULT_SETTINGS[key]
        try:
            if isinstance(default_value, bool):
                updates[key] = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
            elif isinstance(default_value, int):
                updates[key] = int(value)
            elif isinstance(default_value, float):
                updates[key] = float(value)
            else:
                updates[key] = str(value)
        except Exception:
            updates[key] = default_value
    if not updates:
        return {"ok": True, "updated": {}}
    STATE["settings"].update(updates)
    save_settings(STATE["settings"])
    log_activity(f"Settings updated by {user['username']}")
    _refresh_run_event()
    await broadcast_state("settings_update")
    return {"ok": True, "updated": updates}


@app.post("/control/start")
async def start_bot(user=Depends(require_permission("control"))):
    if STATE["running"]:
        return {"ok": True, "running": True, "message": "Bot already running"}
    STATE["running"] = True
    STATE["metrics"]["status"] = "running"
    log_activity(f"Bot started by {user['username']}")
    _refresh_run_event()
    await broadcast_state("bot_started")
    return {"ok": True, "running": True}


@app.post("/control/stop")
async def stop_bot(user=Depends(require_permission("control"))):
    if not STATE["running"]:
        return {"ok": True, "running": False, "message": "Bot already stopped"}
    STATE["running"] = False
    STATE["metrics"]["status"] = "paused"
    log_activity(f"Bot stopped by {user['username']}")
    _refresh_run_event()
    await broadcast_state("bot_stopped")
    return {"ok": True, "running": False}


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
    index_file = DASHBOARD_DIR / "manage.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/health", status_code=307)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps(_snapshot("connection_established")))
        while True:
            await websocket.send_text(json.dumps(_snapshot("tick")))
            await asyncio.sleep(2)
    except Exception:
        pass
    finally:
        manager.disconnect(websocket)


@app.on_event("startup")
async def on_startup():
    _refresh_run_event()
    asyncio.create_task(simulator_loop())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
