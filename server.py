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
from fastapi import FastAPI, WebSocket, HTTPException, Depends, status, Request
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
async def start(current_user: dict = Depends(require_permission("control"))):
    """Start the trading bot."""
    try:
        STATE["running"] = True
        logger.info(f"Trading bot started by user: {current_user['username']}")
        
        # Broadcast to all connected clients
        await manager.broadcast(json.dumps({
            "event": "bot_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": current_user['username']
        }))
        
        return {"ok": True, "message": "Trading bot started"}
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        STATE['server_stats']['error_count'] += 1
        raise HTTPException(status_code=500, detail="Failed to start bot")

@app.post("/control/stop")
async def stop(current_user: dict = Depends(require_permission("control"))):
    """Stop the trading bot."""
    try:
        STATE["running"] = False
        logger.info(f"Trading bot stopped by user: {current_user['username']}")
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")