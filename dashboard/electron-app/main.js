const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

const DEFAULT_OPERATOR_URL = 'http://127.0.0.1:8765';

const OPERATIONS = Object.freeze({
  status: { method: 'GET', path: '/operator/status' },
  orders: { method: 'GET', path: '/operator/orders' },
  positions: { method: 'GET', path: '/operator/positions' },
  startPaper: { method: 'POST', path: '/operator/start-paper' },
  pause: { method: 'POST', path: '/operator/pause' },
  stop: { method: 'POST', path: '/operator/stop' },
  cancelAll: { method: 'POST', path: '/operator/cancel-all' },
  emergencyStop: { method: 'POST', path: '/operator/emergency-stop' }
});

function operatorBaseUrl() {
  const configured = process.env.TRADING_OPERATOR_URL || DEFAULT_OPERATOR_URL;
  const url = new URL(configured);
  const loopbackHosts = new Set(['127.0.0.1', 'localhost', '::1', '[::1]']);
  if (url.protocol !== 'http:' || !loopbackHosts.has(url.hostname)) {
    throw new Error('TRADING_OPERATOR_URL must use HTTP on a loopback address');
  }
  return url.toString().replace(/\/$/, '');
}

async function invokeOperator(operation) {
  const route = OPERATIONS[operation];
  if (!route) {
    throw new Error('Unsupported operator action');
  }

  const token = process.env.TRADING_OPERATOR_TOKEN;
  if (!token) {
    throw new Error('TRADING_OPERATOR_TOKEN is not configured');
  }

  const response = await fetch(`${operatorBaseUrl()}${route.path}`, {
    method: route.method,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/json'
    },
    cache: 'no-store'
  });

  let body = {};
  try {
    body = await response.json();
  } catch {
    body = {};
  }

  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : `Operator API returned HTTP ${response.status}`;
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return body;
}

function registerOperatorBridge() {
  ipcMain.handle('operator:invoke', async (_event, operation) => {
    try {
      return { ok: true, data: await invokeOperator(operation) };
    } catch (error) {
      return {
        ok: false,
        error: error instanceof Error ? error.message : String(error),
        status: Number(error && error.status) || 0
      };
    }
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1120,
    height: 780,
    minWidth: 860,
    minHeight: 620,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true
    }
  });

  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  win.removeMenu();
  win.loadFile(path.join(__dirname, 'renderer', 'operator.html'));
  win.once('ready-to-show', () => win.show());

  if (process.env.NODE_ENV === 'development') {
    win.webContents.openDevTools({ mode: 'detach' });
  }
}

app.whenReady().then(() => {
  registerOperatorBridge();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
