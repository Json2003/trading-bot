"""
Enhanced WebSocket server for trading bot with comprehensive features:

Features:
- JWT-based authentication for sensitive endpoints
- Rate limiting to prevent abuse  
- Robust error handling and connection management
- Real-time server health monitoring
- WebSocket connection pooling and management
- Detailed logging and metrics tracking
"""
from fastapi import FastAPI, WebSocket, HTTPException, Depends, status, Request, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Set
import asyncio
import json
import time
import logging
from collections import defaultdict, deque
from pydantic import BaseModel
import uuid
import os
import glob
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Rate limiting configuration
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_WINDOW = 60  # seconds

app = FastAPI(title="Trading Bot Server", version="1.0.0")
security = HTTPBearer(auto_error=False)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state with enhanced tracking
STATE = {
    "running": False,
    "metrics": {
        "equity": 100000,
        "daily_pnl": 0,
        "win_rate": 0.55,
        "sharpe": 1.2,
        "drawdown": 0.05,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": 0,
        "total_trades": 0,
        "active_positions": 0
    },
    "positions": [],
    "orders": [],
    "server_stats": {
        "start_time": time.time(),
        "active_connections": 0,
        "total_connections": 0,
        "requests_per_minute": 0,
        "error_count": 0
    }
}

# Paths
REPO_ROOT = Path(__file__).resolve().parent
ART_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = REPO_ROOT / "models"
GATE_CFG = ART_DIR / "gate_config.json"
RUN_PID = REPO_ROOT / "run.pid"

def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except Exception:
        return ""

def _write_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt)

def _latest(glob_pattern: str) -> Optional[Path]:
    files = [Path(p) for p in glob.glob(str(ART_DIR / glob_pattern))]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def _json_load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def _tail_csv(path: Path, limit: int = 200):
    try:
        import csv
        rows = []
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows[-limit:]
    except Exception:
        return []

def _tail_text(path: Path, limit: int = 200):
    try:
        with path.open("r", errors="ignore") as f:
            lines = f.readlines()
        return lines[-limit:]
    except Exception:
        return []

def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

# Connection and rate limiting tracking
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, dict] = {}
        self.rate_limiter: Dict[str, deque] = defaultdict(lambda: deque())
        
    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Track connection info
        self.connection_info[websocket] = {
            'client_id': client_id or str(uuid.uuid4()),
            'connected_at': time.time(),
            'last_ping': time.time(),
            'message_count': 0
        }
        
        STATE['server_stats']['active_connections'] = len(self.active_connections)
        STATE['server_stats']['total_connections'] += 1
        
        logger.info(f"Client {self.connection_info[websocket]['client_id']} connected. Active: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
            client_info = self.connection_info.pop(websocket, {})
            client_id = client_info.get('client_id', 'unknown')
            connection_duration = time.time() - client_info.get('connected_at', time.time())
            
            STATE['server_stats']['active_connections'] = len(self.active_connections)
            
            logger.info(f"Client {client_id} disconnected after {connection_duration:.1f}s. Active: {len(self.active_connections)}")
    
    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        now = time.time()
        client_requests = self.rate_limiter[client_ip]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] < now - RATE_LIMIT_WINDOW:
            client_requests.popleft()
        
        # Check if over limit
        if len(client_requests) >= RATE_LIMIT_PER_MINUTE:
            return True
        
        # Add current request
        client_requests.append(now)
        return False
    
    async def broadcast(self, message: str):
        """Broadcast message to all active connections with error handling."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                self.connection_info[connection]['message_count'] += 1
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Authentication models
class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# Mock user database (in production, use proper database)
fake_users_db = {
    "admin": {
        "username": "admin", 
        "hashed_password": "admin123_hashed",  # In production, use proper password hashing
        "permissions": ["read", "write", "control"]
    },
    "readonly": {
        "username": "readonly",
        "hashed_password": "readonly123_hashed",
        "permissions": ["read"]
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Simplified for this example - use proper hashing in production
    return f"{plain_password}_hashed" == hashed_password

def get_user(username: str):
    return fake_users_db.get(username)

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    # Simplified JWT - in production use proper JWT library
    return f"token_{data['sub']}_{int(expire.timestamp())}"

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    
    # Simplified token validation - use proper JWT in production
    if credentials.credentials.startswith("token_"):
        username = credentials.credentials.split("_")[1]
        user = get_user(username)
        if user:
            return user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

def require_permission(permission: str):
    """Dependency to check if user has required permission."""
    async def check_permission(current_user: dict = Depends(get_current_user)):
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        if permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return check_permission

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else 'unknown'
    
    if manager.is_rate_limited(client_ip):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    response = await call_next(request)
    return response

# Health check and server info
@app.get("/health")
async def health_check():
    """Get server health and statistics."""
    uptime = time.time() - STATE['server_stats']['start_time']
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "server_stats": STATE['server_stats'],
        "active_connections": len(manager.active_connections)
    }

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
async def status():
    """Get trading bot status."""
    return {
        "running": STATE["running"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/metrics")  
async def metrics():
    """Get public metrics."""
    STATE["metrics"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    STATE["metrics"]["uptime_seconds"] = time.time() - STATE['server_stats']['start_time']
    return STATE["metrics"]

# Protected endpoints
@app.post("/control/start")
async def start(payload: dict | None = Body(default=None), current_user: dict = Depends(require_permission("control"))):
    """Start the trading bot (optionally spawn a process).

    Body example: { "cmd": ["bash", "scripts/start_paper.sh"] }
    Writes PID to run.pid for management.
    """
    try:
        STATE["running"] = True
        # Optional external command
        cmd = None
        if isinstance(payload, dict):
            cmd = payload.get("cmd")
        spawned = None
        if cmd and isinstance(cmd, list) and not RUN_PID.exists():
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            RUN_PID.write_text(str(proc.pid))
            spawned = {"pid": proc.pid, "cmd": cmd}
        logger.info(f"Trading bot started by user: {current_user['username']}; spawned={spawned}")

        await manager.broadcast(json.dumps({
            "event": "bot_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": current_user['username']
        }))
        return {"ok": True, "message": "Trading bot started", "spawned": spawned}
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        STATE['server_stats']['error_count'] += 1
        raise HTTPException(status_code=500, detail="Failed to start bot")

@app.post("/control/stop")
async def stop(current_user: dict = Depends(require_permission("control"))):
    """Stop the trading bot."""
    try:
        STATE["running"] = False
        # Try to terminate spawned process if tracked
        killed = False
        if RUN_PID.exists():
            try:
                pid = int(RUN_PID.read_text().strip())
                os.kill(pid, 15)
                killed = True
            except Exception:
                pass
            try:
                RUN_PID.unlink(missing_ok=True)
            except Exception:
                pass
        logger.info(f"Trading bot stopped by user: {current_user['username']}; killed={killed}")
        
        # Broadcast to all connected clients
        await manager.broadcast(json.dumps({
            "event": "bot_stopped", 
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": current_user['username']
        }))
        
        return {"ok": True, "message": "Trading bot stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop bot: {e}")
        STATE['server_stats']['error_count'] += 1
        raise HTTPException(status_code=500, detail="Failed to stop bot")

@app.get("/positions")
async def get_positions(current_user: dict = Depends(require_permission("read"))):
    """Get current positions."""
    return {"positions": STATE["positions"]}

@app.get("/orders") 
async def get_orders(current_user: dict = Depends(require_permission("read"))):
    """Get current orders."""
    return {"orders": STATE["orders"]}

# WebSocket endpoint with enhanced connection management
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None):
    """Enhanced WebSocket endpoint with connection management."""
    try:
        await manager.connect(websocket, client_id)
        
        # Send initial data
        initial_data = {
            "event": "connection_established",
            "client_id": manager.connection_info[websocket]['client_id'],
            "metrics": STATE["metrics"],
            "server_time": datetime.now(timezone.utc).isoformat()
        }
        await websocket.send_text(json.dumps(initial_data))
        
        # Main message loop
        while True:
            try:
                # Update metrics with current time and uptime
                STATE["metrics"]["timestamp"] = datetime.now(timezone.utc).isoformat()
                STATE["metrics"]["uptime_seconds"] = time.time() - STATE['server_stats']['start_time']
                
                # Send periodic updates
                message_data = {
                    "event": "metrics_update",
                    "metrics": STATE["metrics"],
                    "server_stats": {
                        "active_connections": STATE['server_stats']['active_connections'],
                        "uptime_seconds": STATE["metrics"]["uptime_seconds"]
                    }
                }
                
                await websocket.send_text(json.dumps(message_data))
                
                # Update last ping time
                manager.connection_info[websocket]['last_ping'] = time.time()
                
                # Wait before next update
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket message loop: {e}")
                STATE['server_stats']['error_count'] += 1
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        STATE['server_stats']['error_count'] += 1
    finally:
        manager.disconnect(websocket)

# Background task to update server statistics
@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup."""
    logger.info("Trading bot server starting up...")
    
    async def update_server_stats():
        while True:
            try:
                # Update requests per minute calculation
                # This is a simplified version - in production, implement proper metrics collection
                await asyncio.sleep(60)
                STATE['server_stats']['requests_per_minute'] = 0  # Reset counter
                
            except Exception as e:
                logger.error(f"Error updating server stats: {e}")
                await asyncio.sleep(10)
    
    # Start background task
    asyncio.create_task(update_server_stats())
    logger.info("Trading bot server startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Trading bot server shutting down...")
    
    # Notify all connected clients
    shutdown_message = {
        "event": "server_shutdown",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await manager.broadcast(json.dumps(shutdown_message))
    
    # Close all connections
    for connection in manager.active_connections.copy():
        try:
            await connection.close()
        except:
            pass
        manager.disconnect(connection)
    
    logger.info("Trading bot server shutdown complete")

# ------------------ Control/Config/Diagnostics endpoints ------------------
# ------------------ Control/Config/Diagnostics endpoints ------------------

class UpdateConfig(BaseModel):
    active_tag: Optional[str] = None
    gate_threshold: Optional[float] = None
    strategy: Optional[dict] = None  # written to artifacts/superbot/active_config.json


@app.get("/config")
async def get_config():
    active_tag = _read_text(MODELS_DIR / "active_tag.txt")
    gate = _json_load(GATE_CFG)
    strat = _json_load(ART_DIR / "superbot" / "active_config.json")
    return {
        "active_tag": active_tag,
        "gate_threshold": gate.get("threshold"),
        "strategy": strat or None,
    }


@app.post("/config")
async def post_config(cfg: UpdateConfig, current_user: dict = Depends(require_permission("control"))):
    before = await get_config()
    if cfg.active_tag:
        _write_text(MODELS_DIR / "active_tag.txt", cfg.active_tag)
    if cfg.gate_threshold is not None:
        GATE_CFG.parent.mkdir(parents=True, exist_ok=True)
        curr = _json_load(GATE_CFG)
        curr["threshold"] = float(cfg.gate_threshold)
        GATE_CFG.write_text(json.dumps(curr, indent=2))
    if cfg.strategy is not None:
        path = ART_DIR / "superbot" / "active_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg.strategy, indent=2))
    after = await get_config()
    return {"ok": True, "before": before, "after": after}


@app.get("/diagnostics/metrics")
async def diagnostics_metrics():
    # Prefer the latest *_metrics.json artifact; fallback to STATE metrics
    path = _latest("*_metrics.json")
    if path:
        try:
            return _json_load(path)
        except Exception:
            pass
    # Fallback
    STATE["metrics"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    return STATE["metrics"]


@app.get("/diagnostics/equity")
async def diagnostics_equity(limit: int = 500):
    path = _latest("*_equity.csv")
    return {"path": str(path) if path else None, "rows": _tail_csv(path, limit) if path else []}


@app.get("/diagnostics/trades")
async def diagnostics_trades(limit: int = 500):
    path = _latest("*_trades.csv")
    return {"path": str(path) if path else None, "rows": _tail_csv(path, limit) if path else []}


@app.get("/logs")
async def get_logs(file: str = "ml", limit: int = 200, current_user: dict = Depends(require_permission("read"))):
    """Tail text logs for UI: ml (artifacts/ml_infer.log) or server (server.log)."""
    if file == "ml":
        path = REPO_ROOT / "artifacts" / "ml_infer.log"
    else:
        path = REPO_ROOT / "server.log"
    return {"file": str(path), "lines": _tail_text(path, limit)}


# Aliases for compatibility
@app.post("/config/model")
async def post_config_model(body: dict, current_user: dict = Depends(require_permission("control"))):
    tag = (body or {}).get("active_tag")
    if not tag:
        raise HTTPException(status_code=400, detail="active_tag required")
    _write_text(MODELS_DIR / "active_tag.txt", tag)
    return await get_config()


@app.post("/config/strategy")
async def post_config_strategy(body: dict, current_user: dict = Depends(require_permission("control"))):
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="strategy body required")
    path = ART_DIR / "superbot" / "active_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2))
    return await get_config()


@app.get("/models/registry")
async def get_model_registry():
    """Return model registry tags and metadata."""
    reg = _read_json(MODELS_DIR / "registry.json")
    return reg or {}


@app.get("/models/tags")
async def get_model_tags(symbol: str | None = None, timeframe: str | None = None):
    reg = _read_json(MODELS_DIR / "registry.json")
    out = []
    for tag, meta in (reg or {}).items():
        if symbol and meta.get("symbol") != symbol:
            continue
        if timeframe and meta.get("timeframe") != timeframe:
            continue
        out.append({"tag": tag, **meta})
    return out


# Simple order/position actions for UI demo
@app.post("/orders/cancel")
async def cancel_order(body: dict, current_user: dict = Depends(require_permission("control"))):
    oid = (body or {}).get("id")
    if oid is None:
        raise HTTPException(status_code=400, detail="id required")
    before = len(STATE["orders"])
    STATE["orders"] = [o for o in STATE["orders"] if str(o.get("id")) != str(oid)]
    return {"ok": True, "removed": before - len(STATE["orders"]) }


@app.post("/positions/close")
async def close_position(body: dict, current_user: dict = Depends(require_permission("control"))):
    pid = (body or {}).get("id")
    if pid is None:
        raise HTTPException(status_code=400, detail="id required")
    # naive: remove from positions and increment total trades
    before = len(STATE["positions"])
    STATE["positions"] = [p for p in STATE["positions"] if str(p.get("id")) != str(pid)]
    STATE["metrics"]["total_trades"] = int(STATE["metrics"].get("total_trades", 0)) + 1
    return {"ok": True, "removed": before - len(STATE["positions"]) }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
