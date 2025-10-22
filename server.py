"""
Enhanced WebSocket server for trading bot with comprehensive features:

Features:
- JWT-based authentication for sensitive endpoints
        html = f"""
- Robust error handling and connection management
- Real-time server health monitoring
- WebSocket connection pooling and management
- Detailed logging and metrics tracking
"""
    <title>Trading Bot — Manage</title>
from fastapi.responses import HTMLResponse, RedirectResponse, Response
        :root {{
          --bg: #0b0f19; --card: #111827; --muted: #6b7280; --text: #e5e7eb; --accent: #06b6d4; --good: #10b981; --bad: #ef4444;
          --border: #1f2937; --chip: #0f172a;
        }}
        body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; color: var(--text); background: linear-gradient(180deg, #0b0f19, #0c111c); }}
        header {{ display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: rgba(11,15,25,0.9); backdrop-filter: blur(6px); }}
        .title {{ font-size: 18px; font-weight: 600; letter-spacing: 0.2px; }}
        .ws-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px; background: #f59e0b; }}
        .ws-ok {{ background: var(--good); }} .ws-bad {{ background: var(--bad); }}
        .container {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 16px; }}
        .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }}
        .card h2 {{ font-size: 14px; margin: 0 0 10px; color: #cbd5e1; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
        .label {{ font-size: 12px; color: var(--muted); }}
        .value {{ font-weight: 600; font-size: 16px; }}
        .row {{ display: flex; align-items: center; gap: 10px; }}
        .stack {{ display: grid; gap: 10px; }}
        button {{ padding: 8px 12px; border-radius: 10px; border: 1px solid var(--border); background: #0b1220; color: var(--text); cursor: pointer; transition: .2s; }}
        button:hover {{ border-color: #334155; transform: translateY(-1px); }}
        input, select {{ width: 100%; padding: 8px 10px; border: 1px solid var(--border); border-radius: 10px; background: #0b1220; color: var(--text); }}
        .chips {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .chip {{ font-size: 12px; padding: 4px 8px; border-radius: 999px; background: var(--chip); border: 1px solid var(--border); color: #9ca3af; }}
        .ok {{ color: var(--good); }} .bad {{ color: var(--bad); }} .muted {{ color: var(--muted); }}
        .footer {{ color: #64748b; font-size: 12px; padding: 10px 20px 20px; text-align: center; }}
import uuid
import os
import httpx
        let token = localStorage.getItem('authToken') || '';
        let currentUser = localStorage.getItem('authUser') || '';

        function setAuth(tk, user) {{
            token = tk || '';
            currentUser = user || '';
            if (token) {{ localStorage.setItem('authToken', token); }} else {{ localStorage.removeItem('authToken'); }}
            if (user) {{ localStorage.setItem('authUser', user); }} else {{ localStorage.removeItem('authUser'); }}
            document.getElementById('whoami').textContent = token ? `Logged in as ${currentUser}` : 'Not authenticated';
            document.getElementById('btnStart').disabled = !token;
            document.getElementById('btnStop').disabled = !token;
        }}

        function authHeaders() {{
            return token ? {{ 'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json' }} : {{ 'Content-Type': 'application/json' }};
        }}

from pydantic import BaseModel

try:
    from tradingbot_ibkr.services.mcp_client import MCPClient  # optional
except Exception:
    MCPClient = None  # type: ignore

                        renderMetrics(msg.metrics);

# Configure logging
                        applySettings(msg.settings);
        logging.FileHandler('server.log'),
                    if (msg.server_stats) {{
                        renderServerStats(msg.server_stats);
                    }}
        logging.StreamHandler()
    ]
            ws.onopen = () => document.getElementById('wsdot').classList.add('ws-ok');
            ws.onclose = () => {{
                document.getElementById('wsdot').classList.remove('ws-ok');
                document.getElementById('wsdot').classList.add('ws-bad');
                setTimeout(connectWS, 2000);
            }};
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

                BACKTEST_INTERVAL: parseInt(document.getElementById('interval').value || '60'),
                RISK_PCT: parseFloat(document.getElementById('risk').value || '0.01'),
                STOP_LOSS_PCT: parseFloat(document.getElementById('sl').value || '0.02'),
                TAKE_PROFIT_PCT: parseFloat(document.getElementById('tp').value || '0.04'),
                EXCHANGE: document.getElementById('exchange').value || 'binance'
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_WINDOW = 60  # seconds
                const res = await fetch('/settings', {{ method: 'POST', headers: authHeaders(), body: JSON.stringify(payload) }});
app = FastAPI(title="Trading Bot Server", version="1.0.0")
security = HTTPBearer(auto_error=False)
                toast('Settings saved');

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
                applySettings(js.settings || {{}});
    val = os.getenv(name)
    if val is None:
        return default
        function applySettings(s) {{
            document.getElementById('paper').checked = !!s.PAPER;
            document.getElementById('cont').checked = !!s.CONTINUOUS_BACKTEST;
            document.getElementById('strategy').value = s.STRATEGY || 'sma_cross';
            document.getElementById('interval').value = s.BACKTEST_INTERVAL ?? 60;
            document.getElementById('risk').value = s.RISK_PCT ?? 0.01;
            document.getElementById('sl').value = s.STOP_LOSS_PCT ?? 0.02;
            document.getElementById('tp').value = s.TAKE_PROFIT_PCT ?? 0.04;
            document.getElementById('exchange').value = s.EXCHANGE || 'binance';
        }

        function renderMetrics(m) {{
            document.getElementById('equity').textContent = fmtCurrency(m.equity);
            const pnlEl = document.getElementById('pnl');
            pnlEl.textContent = fmtNumber(m.daily_pnl);
            pnlEl.className = 'value ' + (m.daily_pnl >= 0 ? 'ok' : 'bad');
            document.getElementById('sharpe').textContent = (m.sharpe ?? 0).toFixed(2);
            document.getElementById('drawdown').textContent = fmtPct(m.drawdown);
        }

        function renderServerStats(s) {{
            document.getElementById('uptime').textContent = fmtDuration(s.uptime_seconds || 0);
            document.getElementById('conns').textContent = s.active_connections ?? 0;
        }

        function fmtCurrency(x) {{
            const v = Number(x||0);
            return new Intl.NumberFormat(undefined, {{ style: 'currency', currency: 'USD', maximumFractionDigits: 2 }}).format(v);
        }}
        function fmtNumber(x) {{
            const v = Number(x||0);
            return new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 2 }}).format(v);
        }}
        function fmtPct(x) {{
            const v = Number(x||0);
            return (v*100).toFixed(2) + '%';
        }}
        function fmtDuration(sec) {{
            const s = Math.floor(sec||0);
            const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), r = s%60;
            return `${h}h ${m}m ${r}s`;
        }}

        async function loadInitialMetrics() {{
            try {{
                const r = await fetch('/metrics');
                const m = await r.json();
                renderMetrics(m);
                // server_stats via WS; also fetch health for snapshot
                const h = await (await fetch('/health')).json();
                renderServerStats({{ uptime_seconds: h.uptime_seconds, active_connections: h.active_connections }});
            }} catch (e) {{ console.log('Initial metrics load failed', e); }}
        }}

        async function login() {{
            const username = document.getElementById('user').value.trim();
            const password = document.getElementById('pass').value;
            if (!username || !password) return toast('Enter username and password');
            try {{
                const res = await fetch('/auth/login', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ username, password }}) }});
                if (!res.ok) throw new Error('Login failed');
                const js = await res.json();
                setAuth(js.access_token, username);
                toast('Logged in');
            }} catch (e) {{ toast('Login failed'); console.error(e); }}
        }}

        function logout() {{ setAuth('', ''); toast('Logged out'); }}

        async function startBot() {{
            try {{
                const r = await fetch('/control/start', {{ method: 'POST', headers: authHeaders() }});
                const js = await r.json();
                toast(js.message || 'Started');
            }} catch (e) {{ toast('Start failed'); }}
        }}
        async function stopBot() {{
            try {{
                const r = await fetch('/control/stop', {{ method: 'POST', headers: authHeaders() }});
                const js = await r.json();
                toast(js.message || 'Stopped');
            }} catch (e) {{ toast('Stop failed'); }}
        }}

        function toast(msg) {{
            const el = document.getElementById('status');
            el.textContent = msg; el.style.opacity = 1;
            setTimeout(() => el.style.opacity = 0.0, 1800);
        }}
    return str(val).lower() in ("1", "true", "yes", "on")
    "CONTINUOUS_BACKTEST": _bool_env("CONTINUOUS_BACKTEST", False),
    "BACKTEST_INTERVAL": _int_env("BACKTEST_INTERVAL", 60),
    "RISK_PCT": float(os.getenv("RISK_PCT", "0.01")),
    "STOP_LOSS_PCT": float(os.getenv("STOP_LOSS_PCT", "0.02")),
            setAuth(token, currentUser);
    "TAKE_PROFIT_PCT": float(os.getenv("TAKE_PROFIT_PCT", "0.04")),
}

    <body>
        <header>
          <div class="title"><span id="wsdot" class="ws-dot"></span>Trading Bot Manager</div>
          <div class="chips">
            <span id="whoami" class="chip">Not authenticated</span>
            <button onclick="login()">Login</button>
            <button onclick="logout()">Logout</button>
          </div>
        </header>
        <div class="container">
          <div class="grid">
            <div class="card stack">
              <h2>Settings</h2>
              <div class="row"><input type="checkbox" id="paper" /> <label for="paper">Paper Trading</label></div>
              <div class="row"><input type="checkbox" id="cont" /> <label for="cont">Continuous Backtest</label></div>
              <div class="row">
                <label class="label" for="exchange" style="width: 160px;">Exchange</label>
                <input id="exchange" placeholder="binance" />
              </div>
              <div class="row">
                <label class="label" for="strategy" style="width: 160px;">Strategy</label>
                <select id="strategy">
                    <option value="sma_cross">sma_cross</option>
                    <option value="enhanced">enhanced</option>
                    <option value="breakout">breakout</option>
                </select>
              </div>
              <div class="row">
                <label class="label" for="interval" style="width: 160px;">Backtest Interval (s)</label>
                <input id="interval" type="number" min="5" step="5" value="60" />
              </div>
              <div class="row">
                <label class="label" for="risk" style="width: 160px;">Risk %</label>
                <input id="risk" type="number" step="0.001" min="0" max="1" />
              </div>
              <div class="row">
                <label class="label" for="sl" style="width: 160px;">Stop Loss %</label>
                <input id="sl" type="number" step="0.001" min="0" max="1" />
              </div>
              <div class="row">
                <label class="label" for="tp" style="width: 160px;">Take Profit %</label>
                <input id="tp" type="number" step="0.001" min="0" max="1" />
              </div>
              <div class="row" style="margin-top:8px;">
                <button onclick="saveSettings()">Save Settings</button>
              </div>
            </div>

            <div class="card stack">
              <h2>Controls</h2>
              <div class="row">
                <button id="btnStart" onclick="startBot()" disabled>Start Bot</button>
                <button id="btnStop" onclick="stopBot()" disabled>Stop Bot</button>
                <button onclick="loadInitialMetrics()">Refresh</button>
              </div>
              <div class="chips" style="margin-top:8px;">
                <span class="chip">WS: <span class="muted">watching metrics</span></span>
                <span class="chip">API: <span class="muted">/settings, /metrics, /health</span></span>
              </div>
              <div class="stack" style="margin-top:8px;">
                <div class="label">Uptime</div>
                <div class="value" id="uptime">-</div>
                <div class="label">Active Connections</div>
                <div class="value" id="conns">-</div>
              </div>
            </div>

            <div class="card stack">
              <h2>Key Metrics</h2>
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
        </div>
        <div id="status" style="position: fixed; left: 50%; transform: translateX(-50%); bottom: 16px; background: #0b1220; color: #cbd5e1; border: 1px solid var(--border); padding: 8px 12px; border-radius: 10px; opacity: 0; transition: opacity .3s;"></div>
        <div class="footer">UI is a lightweight control surface; sensitive endpoints require login. Default demo users: admin/admin123 or readonly/readonly123.</div>
    </body>
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": 0,
        "total_trades": 0,
        "active_positions": 0
    },
    "positions": [],
    "orders": [],
    "settings": load_settings(),
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
        
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
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

# Optional MCP integration
MCP_CLIENT = MCPClient.from_env() if MCPClient else None
LANGCHAIN_AGENT = LangChainAgentService.from_env() if LangChainAgentService else None


async def _run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# Authentication models
class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class AgentInvocationRequest(BaseModel):
    assistant_id: str
    input: Dict[str, Any]
    thread_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

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
async def get_status():
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


# Root redirect for convenience
@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/manage", status_code=307)

# Quiet the favicon 404s
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# Simple management page (must be declared before the __main__ block)
@app.get("/manage", response_class=HTMLResponse)
async def manage_page():
        html = f"""
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Trading Bot — Manage</title>
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

        async function loadInitialMetrics() {{
            try {{
                const r = await fetch('/metrics');
                const m = await r.json();
                document.getElementById('equity').textContent = m.equity ?? '-';
                document.getElementById('pnl').textContent = m.daily_pnl ?? '-';
                document.getElementById('sharpe').textContent = m.sharpe ?? '-';
                document.getElementById('drawdown').textContent = m.drawdown ?? '-';
            }} catch (e) {{ console.log('Initial metrics load failed', e); }}
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            connectWS();
            loadSettings();
            loadInitialMetrics();
        }});
    </script>
    </head>
    <body>
        <h1>Manage</h1>
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
@app.get("/settings")
async def get_settings():
    """Return current runtime settings."""
    return {"settings": STATE.get("settings", {})}


@app.post("/settings")
async def update_settings(payload: Dict[str, Any]):
    """Update runtime settings (partial updates allowed)."""
    allowed_keys = set(DEFAULT_SETTINGS.keys())
    updates = {k: v for k, v in payload.items() if k in allowed_keys}
    if not updates:
        return {"ok": True, "updated": {}, "message": "No valid keys provided"}

    # Basic type normalization
    for key, default_val in DEFAULT_SETTINGS.items():
        if key in updates:
            val = updates[key]
            try:
                if isinstance(default_val, bool):
                    updates[key] = bool(val) if isinstance(val, bool) else str(val).lower() in ("1","true","yes","on")
                elif isinstance(default_val, int):
                    updates[key] = int(val)
                elif isinstance(default_val, float):
                    updates[key] = float(val)
                else:
                    updates[key] = str(val)
            except Exception:
                updates[key] = default_val

    STATE["settings"].update(updates)
    save_settings(STATE["settings"])

    # Broadcast settings update
    try:
        await manager.broadcast(json.dumps({
            "event": "settings_update",
            "settings": STATE["settings"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
    except Exception as exc:
        logger.warning("WS broadcast failed for settings update: %s", exc)

    return {"ok": True, "updated": updates}


@app.get("/mcp/health")
async def mcp_health():
    if not MCP_CLIENT:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP integration not configured")
    try:
        data = await _run_in_executor(MCP_CLIENT.heartbeat)
        return {"status": "ok", "mcp": data}
    except Exception as exc:
        logger.exception("MCP health check failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


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
            "settings": STATE.get("settings", {}),
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
                    "settings": STATE.get("settings", {}),
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
