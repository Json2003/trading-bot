"""Trading Bot Server with WebSocket support.

Features:
- Rate limiting for WebSocket connections
- Authentication for sensitive API endpoints
- Comprehensive error handling and logging
- Connection management and monitoring
"""
import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Bot API", version="1.0.0")

# Authentication
security = HTTPBearer()
API_KEY = "your-secret-api-key"  # In production, use environment variables

# Rate limiting storage
rate_limit_store = defaultdict(lambda: deque())
connection_timestamps = defaultdict(list)

STATE = {
    "running": False,
    "metrics": {
        "equity": 100000,
        "daily_pnl": 0,
        "win_rate": 0.55,
        "sharpe": 1.2,
        "drawdown": 0.05,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_trades": 0,
        "active_positions": 0
    },
    "positions": [],
    "orders": [],
    "last_updated": datetime.now(timezone.utc).isoformat()
}

# Connection management
clients: Set[WebSocket] = set()
connection_stats = {
    "total_connections": 0,
    "active_connections": 0,
    "total_messages_sent": 0,
    "total_errors": 0,
    "uptime_start": datetime.now(timezone.utc)
}


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify API key for protected endpoints.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        True if valid API key
        
    Raises:
        HTTPException: If invalid API key
    """
    if credentials.credentials != API_KEY:
        logger.warning(f"Invalid API key attempt: {credentials.credentials[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def rate_limit_check(client_id: str, max_requests: int = 10, window_seconds: int = 60) -> bool:
    """Check if client exceeds rate limit.
    
    Args:
        client_id: Unique identifier for client
        max_requests: Maximum requests allowed in window
        window_seconds: Time window in seconds
        
    Returns:
        True if within rate limit, False otherwise
    """
    now = time.time()
    client_requests = rate_limit_store[client_id]
    
    # Remove old requests outside the window
    while client_requests and client_requests[0] < now - window_seconds:
        client_requests.popleft()
    
    # Check if limit exceeded
    if len(client_requests) >= max_requests:
        return False
    
    # Add current request
    client_requests.append(now)
    return True


def connection_limit_check(client_ip: str, max_connections: int = 5) -> bool:
    """Check if client exceeds connection limit.
    
    Args:
        client_ip: Client IP address
        max_connections: Maximum concurrent connections per IP
        
    Returns:
        True if within connection limit
    """
    current_time = time.time()
    timestamps = connection_timestamps[client_ip]
    
    # Remove old connections (older than 1 hour)
    connection_timestamps[client_ip] = [
        ts for ts in timestamps if current_time - ts < 3600
    ]
    
    return len(connection_timestamps[client_ip]) < max_connections

# Public endpoints
@app.get("/status")
def status():
    """Get current bot status (public endpoint)."""
    try:
        return {
            "running": STATE["running"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - connection_stats["uptime_start"]).total_seconds()
        }
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_connections": len(clients),
        "total_connections": connection_stats["total_connections"]
    }


# Protected endpoints requiring API key authentication
@app.post("/control/start")
def start(authenticated: bool = Depends(verify_api_key)):
    """Start the trading bot (requires authentication)."""
    try:
        STATE["running"] = True
        STATE["last_updated"] = datetime.now(timezone.utc).isoformat()
        logger.info("Trading bot started via API")
        return {
            "ok": True,
            "message": "Trading bot started successfully",
            "timestamp": STATE["last_updated"]
        }
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail="Failed to start bot")


@app.post("/control/stop")
def stop(authenticated: bool = Depends(verify_api_key)):
    """Stop the trading bot (requires authentication)."""
    try:
        STATE["running"] = False
        STATE["last_updated"] = datetime.now(timezone.utc).isoformat()
        logger.info("Trading bot stopped via API")
        return {
            "ok": True,
            "message": "Trading bot stopped successfully",
            "timestamp": STATE["last_updated"]
        }
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop bot")


@app.get("/metrics")
def get_metrics(authenticated: bool = Depends(verify_api_key)):
    """Get detailed bot metrics (requires authentication)."""
    try:
        return {
            "metrics": STATE["metrics"],
            "positions": STATE["positions"],
            "orders": STATE["orders"],
            "connection_stats": connection_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


async def broadcast_to_clients(message: dict):
    """Broadcast message to all connected WebSocket clients.
    
    Args:
        message: Dictionary to broadcast as JSON
    """
    if not clients:
        return
    
    disconnected = set()
    message_json = json.dumps(message)
    
    for client in clients.copy():
        try:
            await client.send_text(message_json)
            connection_stats["total_messages_sent"] += 1
        except Exception as e:
            logger.warning(f"Failed to send message to client: {e}")
            disconnected.add(client)
            connection_stats["total_errors"] += 1
    
    # Remove disconnected clients
    for client in disconnected:
        clients.discard(client)
        connection_stats["active_connections"] = len(clients)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with rate limiting and error handling.
    
    Provides real-time updates of trading bot metrics with:
    - Connection rate limiting per IP
    - Request rate limiting per connection
    - Comprehensive error handling
    - Connection lifecycle management
    """
    client_host = websocket.client.host if websocket.client else "unknown"
    client_id = f"{client_host}:{websocket.client.port if websocket.client else 'unknown'}"
    
    # Check connection limit
    if not connection_limit_check(client_host):
        logger.warning(f"Connection limit exceeded for {client_host}")
        await websocket.close(code=1008, reason="Connection limit exceeded")
        return
    
    try:
        await websocket.accept()
        clients.add(websocket)
        connection_timestamps[client_host].append(time.time())
        connection_stats["total_connections"] += 1
        connection_stats["active_connections"] = len(clients)
        
        logger.info(f"WebSocket client connected: {client_id}")
        
        # Send initial state
        initial_message = {
            "type": "initial",
            "metrics": STATE["metrics"],
            "running": STATE["running"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await websocket.send_text(json.dumps(initial_message))
        
        # Main message loop
        last_heartbeat = time.time()
        message_count = 0
        
        while True:
            # Rate limiting check
            if not rate_limit_check(client_id, max_requests=60, window_seconds=60):  # 60 messages per minute
                logger.warning(f"Rate limit exceeded for {client_id}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Rate limit exceeded",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
                await asyncio.sleep(5)  # Throttle the client
                continue
            
            # Update metrics and send
            current_time = datetime.now(timezone.utc)
            STATE["metrics"]["timestamp"] = current_time.isoformat()
            STATE["last_updated"] = current_time.isoformat()
            
            # Prepare message
            message = {
                "type": "update",
                "metrics": STATE["metrics"],
                "running": STATE["running"],
                "positions": len(STATE["positions"]),
                "orders": len(STATE["orders"]),
                "timestamp": current_time.isoformat(),
                "sequence": message_count
            }
            
            await websocket.send_text(json.dumps(message))
            connection_stats["total_messages_sent"] += 1
            message_count += 1
            
            # Send heartbeat every 30 seconds
            if time.time() - last_heartbeat > 30:
                heartbeat_message = {
                    "type": "heartbeat",
                    "timestamp": current_time.isoformat(),
                    "uptime": (current_time - connection_stats["uptime_start"]).total_seconds()
                }
                await websocket.send_text(json.dumps(heartbeat_message))
                last_heartbeat = time.time()
            
            # Wait before next update
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected normally: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        connection_stats["total_errors"] += 1
    finally:
        # Cleanup
        clients.discard(websocket)
        connection_stats["active_connections"] = len(clients)
        logger.info(f"WebSocket client cleanup completed for {client_id}")


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    logger.info("Trading Bot Server starting up...")
    connection_stats["uptime_start"] = datetime.now(timezone.utc)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    logger.info("Trading Bot Server shutting down...")
    # Close all WebSocket connections
    for client in clients.copy():
        try:
            await client.close(code=1001, reason="Server shutdown")
        except Exception as e:
            logger.error(f"Error closing WebSocket connection: {e}")
    clients.clear()


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Trading Bot Server...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
