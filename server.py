<<<<<<< ours
=======
"""
Splitstar Operations Console FastAPI server with comprehensive features:

Features:
- JWT-based authentication for sensitive endpoints
- Rate limiting to prevent abuse  
- Robust error handling and connection management
- Real-time server health monitoring
- WebSocket connection pooling and management
- Detailed logging and metrics tracking
"""
from fastapi import FastAPI, WebSocket, HTTPException, Depends, status, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Set, Any
>>>>>>> theirs
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

# Security instance for authentication
security = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------------------
# Mock Authentication
# --------------------------------------------------------------------------------------
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "admin_password_hash",  # In production, use proper password hashing
        "permissions": ["control", "view", "settings"]
    },
    "trader": {
        "username": "trader", 
        "hashed_password": "trader_password_hash",
        "permissions": ["view"]
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Mock password verification - in production use proper hashing like bcrypt"""
    return plain_password == hashed_password.replace("_hash", "")

def create_access_token(username: str) -> str:
    """Mock token creation - in production use proper JWT"""
    timestamp = int(time.time())
    return f"token_{username}_{timestamp}"
# --------------------------------------------------------------------------------------
# Global state and settings
# --------------------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "auto_trade": False,
    "max_position_size": 1000.0,
    "risk_level": 0.02,
    "enable_notifications": True
}

STATE = {
    "running": False,
    "settings": DEFAULT_SETTINGS.copy(),
    "metrics": {
        "total_trades": 0,
        "profitable_trades": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat()
    },
    "markets": [],
    "news": [],
    "trades_current": [],
    "trades_proposed": [],
    "activity": [],
    "server_stats": {
        "start_time": time.time(),
        "active_connections": 0,
        "total_connections": 0
    }
}

dashboard_dir = Path(__file__).parent / "dashboard"

def save_settings(settings: Dict[str, Any]) -> None:
    """Mock settings persistence"""
    logger.info(f"Settings saved: {settings}")

# --------------------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
)
logger = logging.getLogger("server")


<<<<<<< ours
# --------------------------------------------------------------------------------------
# App and CORS
# --------------------------------------------------------------------------------------
app = FastAPI(title="Trading Bot Server", version="1.1.0")
=======
# Rate limiting configuration
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_WINDOW = 60  # seconds

# Brand configuration
APP_BRAND = os.getenv("APP_BRAND", "Splitstar Operations Console")

app = FastAPI(title=f"{APP_BRAND} API", version="1.0.0")
security = HTTPBearer(auto_error=False)

# Add CORS middleware
>>>>>>> theirs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Authentication functions
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Connection manager
# --------------------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, dict] = {}
        self.rate_limiter: Dict[str, deque] = defaultdict(lambda: deque())

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            "client_id": client_id or f"ws-{random.randint(1000,9999)}",
            "connected_at": time.time(),
            "message_count": 0,
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
        dead: List[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_text(msg)
            except Exception as e:
                logger.warning("WS send failed: %s", e)
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# --------------------------------------------------------------------------------------
# Simulated feeds
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# REST endpoints
# --------------------------------------------------------------------------------------
@app.get("/health")
async def health():
    uptime = time.time() - STATE["server_stats"]["start_time"]
    return {"status": "healthy", "uptime_seconds": uptime, "active_connections": len(manager.active_connections)}

<<<<<<< ours
=======
# Authentication endpoints
@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Authenticate user and return access token."""
    try:
        user = authenticate_user(user_credentials.username, user_credentials.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"]}, expires_delta=access_token_expires
        )
        
        logger.info(f"User {user['username']} logged in successfully")
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        STATE['server_stats']['error_count'] += 1
        raise HTTPException(status_code=500, detail="Login failed")

# Public endpoints
@app.get("/status")
async def get_status():
    """Get Splitstar Operations Console status."""
    return {
        "running": STATE["running"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
>>>>>>> theirs

@app.get("/metrics")
async def metrics():
    STATE["metrics"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    STATE["metrics"]["uptime_seconds"] = time.time() - STATE["server_stats"]["start_time"]
    return STATE["metrics"]


<<<<<<< ours
=======
# Simple management page (must be declared before the __main__ block)
@app.get("/manage", response_class=HTMLResponse)
async def manage_page():
        html = f"""
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{APP_BRAND} — Manage</title>
    <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; color: #111; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
        .card {{ border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
        h1 {{ margin-top: 0; font-size: 20px; }}
        .label {{ font-size: 12px; color: #555; }}
        .value {{ font-weight: 600; }}
        button {{ padding: 8px 12px; border-radius: 8px; border: 1px solid #cbd5e1; background: #f8fafc; cursor: pointer; }}
        input, select {{ width: 100%; padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 8px; }}
        .row {{ display: flex; align-items: center; gap: 8px; }}
    </style>
    <script>
        let ws;
        function connectWS() {{
            const proto = location.protocol === 'https:' ? 'wss' : 'ws';
            ws = new WebSocket(`${{proto}}://` + location.host + '/ws');
            ws.onmessage = (ev) => {{
                try {{
                    const msg = JSON.parse(ev.data);
                    if (msg.metrics) {{
                        document.getElementById('equity').textContent = msg.metrics.equity;
                        document.getElementById('pnl').textContent = msg.metrics.daily_pnl;
                        document.getElementById('sharpe').textContent = msg.metrics.sharpe;
                        document.getElementById('drawdown').textContent = msg.metrics.drawdown;
                    }}
                    if (msg.settings) {{
                        document.getElementById('paper').checked = !!msg.settings.PAPER;
                        document.getElementById('cont').checked = !!msg.settings.CONTINUOUS_BACKTEST;
                        document.getElementById('strategy').value = msg.settings.STRATEGY || 'sma_cross';
                        document.getElementById('interval').value = msg.settings.BACKTEST_INTERVAL || 60;
                    }}
                }} catch (e) {{ console.log('WS parse error', e); }}
            }}
            ws.onclose = () => setTimeout(connectWS, 2000);
        }}

        async function saveSettings() {{
            const payload = {{
                PAPER: document.getElementById('paper').checked,
                CONTINUOUS_BACKTEST: document.getElementById('cont').checked,
                STRATEGY: document.getElementById('strategy').value,
                BACKTEST_INTERVAL: parseInt(document.getElementById('interval').value || '60')
            }};
            try {{
                const res = await fetch('/settings', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(payload) }});
                const js = await res.json();
                console.log('Saved', js);
            }} catch (e) {{ console.error('Save failed', e); }}
        }}

        async function loadSettings() {{
            try {{
                const res = await fetch('/settings');
                const js = await res.json();
                const s = js.settings || {{}};
                document.getElementById('paper').checked = !!s.PAPER;
                document.getElementById('cont').checked = !!s.CONTINUOUS_BACKTEST;
                document.getElementById('strategy').value = s.STRATEGY || 'sma_cross';
                document.getElementById('interval').value = s.BACKTEST_INTERVAL || 60;
            }} catch (e) {{ console.log('Load settings failed', e); }}
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            connectWS();
            loadSettings();
        }});
    </script>
    </head>
    <body>
        <h1>{APP_BRAND} — Manage</h1>
        <p class="label" style="margin-top:-8px; margin-bottom:16px;">Live controls, telemetry, and configuration switches for Splitstar operations.</p>
        <div class="grid">
            <div class="card">
                <div class="row"><input type="checkbox" id="paper" /> <label for="paper">Paper Trading</label></div>
                <div class="row"><input type="checkbox" id="cont" /> <label for="cont">Continuous Backtest</label></div>
                <div class="row">
                    <label class="label" for="strategy">Strategy</label>
                    <select id="strategy">
                        <option value="sma_cross">sma_cross</option>
                        <option value="enhanced">enhanced</option>
                        <option value="breakout">breakout</option>
                    </select>
                </div>
                <div class="row">
                    <label class="label" for="interval">Backtest Interval (s)</label>
                    <input id="interval" type="number" min="5" step="5" value="60" />
                </div>
                <div style="margin-top:10px;" class="row">
                    <button onclick="saveSettings()">Save Settings</button>
                </div>
            </div>
            <div class="card">
                <div class="label">Equity</div>
                <div class="value" id="equity">-</div>
                <div class="label">Daily PnL</div>
                <div class="value" id="pnl">-</div>
                <div class="label">Sharpe</div>
                <div class="value" id="sharpe">-</div>
                <div class="label">Drawdown</div>
                <div class="value" id="drawdown">-</div>
            </div>
        </div>
    </body>
    </html>
        """
        return HTMLResponse(content=html)


# Settings endpoints
>>>>>>> theirs
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


<<<<<<< ours
@app.get("/feed/markets")
async def feed_markets():
    return {"markets": STATE["markets"]}


@app.get("/feed/news")
async def feed_news():
    return {"news": STATE["news"]}


@app.get("/trades/current")
async def trades_current():
    return {"trades": STATE["trades_current"]}

=======
@app.get("/mcp/signals")
async def mcp_fetch_signals():
    if not MCP_CLIENT:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP integration not configured")
    try:
        signals = await _run_in_executor(MCP_CLIENT.fetch_signal_batch)
        return {"signals": signals}
    except Exception as exc:
        logger.exception("Failed to fetch signals from MCP")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@app.post("/mcp/metrics")
async def mcp_push_metrics():
    if not MCP_CLIENT:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP integration not configured")
    payload = {
        "metrics": STATE.get("metrics", {}),
        "orders": STATE.get("orders", []),
        "positions": STATE.get("positions", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        response = await _run_in_executor(MCP_CLIENT.push_metrics, payload)
        return {"status": "ok", "response": response}
    except Exception as exc:
        logger.exception("Failed to push metrics to MCP")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

@app.post("/agents/run")
async def agents_run(
    request: AgentInvocationRequest,
    current_user: dict = Depends(require_permission("write")),
):
    if not LANGCHAIN_AGENT:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LangChain agent integration not configured",
        )
    try:
        run = await _run_in_executor(
            LANGCHAIN_AGENT.start_run,
            request.assistant_id,
            request.input,
            thread_id=request.thread_id,
            metadata=request.metadata,
        )
        return {"status": "submitted", "run": run}
    except httpx.HTTPStatusError as exc:
        logger.error(
            "LangChain agent invocation returned HTTP %s: %s",
            exc.response.status_code if exc.response else "unknown",
            exc.response.text if exc.response else str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LangChain agent call failed",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected LangChain agent error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

# Protected endpoints
@app.post("/control/start")
async def start(current_user: dict = Depends(require_permission("control"))):
    """Start the Splitstar Operations Console execution loop."""
    try:
        STATE["running"] = True
        logger.info(f"{APP_BRAND} started by user: {current_user['username']}")

        # Broadcast to all connected clients
        await manager.broadcast(json.dumps({
            "event": "bot_started",
            "brand_event": "console_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": current_user['username']
        }))

        return {"ok": True, "message": f"{APP_BRAND} started"}
        
    except Exception as e:
        logger.error(f"Failed to start console: {e}")
        STATE['server_stats']['error_count'] += 1
        raise HTTPException(status_code=500, detail="Failed to start console")

@app.post("/control/stop")
async def stop(current_user: dict = Depends(require_permission("control"))):
    """Stop the Splitstar Operations Console execution loop."""
    try:
        STATE["running"] = False
        logger.info(f"{APP_BRAND} stopped by user: {current_user['username']}")

        # Broadcast to all connected clients
        await manager.broadcast(json.dumps({
            "event": "bot_stopped",
            "brand_event": "console_stopped",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": current_user['username']
        }))

        return {"ok": True, "message": f"{APP_BRAND} stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop console: {e}")
        STATE['server_stats']['error_count'] += 1
        raise HTTPException(status_code=500, detail="Failed to stop console")
>>>>>>> theirs

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


# --------------------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------------------
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
    except Exception as e:
        logger.info("WS disconnected: %s", e)
    finally:
        manager.disconnect(ws)


# --------------------------------------------------------------------------------------
# Startup tasks
# --------------------------------------------------------------------------------------
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


@app.on_event("shutdown")
async def on_shutdown():
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

