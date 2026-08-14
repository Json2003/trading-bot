const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

const DEFAULT_OPERATOR_URL = 'http://127.0.0.1:8765';
const JOB_ID_PATTERN = /^[a-f0-9]{32}$/;

const OPERATIONS = Object.freeze({
  status: { method: 'GET', path: '/operator/status' },
  orders: { method: 'GET', path: '/operator/orders' },
  positions: { method: 'GET', path: '/operator/positions' },
  startPaper: { method: 'POST', path: '/operator/start-paper' },
  pause: { method: 'POST', path: '/operator/pause' },
  stop: { method: 'POST', path: '/operator/stop' },
  cancelAll: { method: 'POST', path: '/operator/cancel-all' },
  emergencyStop: { method: 'POST', path: '/operator/emergency-stop' },
  researchDatasets: { method: 'GET', path: '/research/datasets' },
  researchJobs: { method: 'GET', path: '/research/jobs' },
  startResearch: { method: 'POST', path: '/research/jobs', body: validateResearchSpec },
  cancelResearch: {
    method: 'POST',
    path: (payload) => `/research/jobs/${validateJobId(payload && payload.jobId)}/cancel`
  }
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

function boundedInteger(value, name, minimum, maximum) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < minimum || number > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}`);
  }
  return number;
}

function validateResearchSpec(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('Research request is required');
  }
  const datasetId = String(payload.datasetId || '').trim();
  if (!datasetId || datasetId.length > 240 || datasetId.includes('..') || datasetId.startsWith('/')) {
    throw new Error('Select a valid local research dataset');
  }
  const holdout = Number(payload.finalHoldoutFraction);
  if (!Number.isFinite(holdout) || holdout < 0.20 || holdout > 0.40) {
    throw new Error('Final holdout must be between 20% and 40%');
  }
  return {
    dataset_id: datasetId,
    generations: boundedInteger(payload.generations, 'Generations', 1, 6),
    accounts_per_generation: boundedInteger(
      payload.accountsPerGeneration,
      'Accounts per generation',
      4,
      24
    ),
    final_holdout_fraction: holdout,
    seed: boundedInteger(payload.seed, 'Seed', 0, 2147483647)
  };
}

function validateJobId(value) {
  const jobId = String(value || '').trim();
  if (!JOB_ID_PATTERN.test(jobId)) {
    throw new Error('Invalid research job identifier');
  }
  return jobId;
}

async function invokeOperator(operation, payload = null) {
  const route = OPERATIONS[operation];
  if (!route) {
    throw new Error('Unsupported operator action');
  }

  const token = process.env.TRADING_OPERATOR_TOKEN;
  if (!token) {
    throw new Error('TRADING_OPERATOR_TOKEN is not configured');
  }

  const routePath = typeof route.path === 'function' ? route.path(payload) : route.path;
  const requestBody = route.body ? route.body(payload) : null;
  const headers = {
    Authorization: `Bearer ${token}`,
    Accept: 'application/json'
  };
  if (requestBody) headers['Content-Type'] = 'application/json';

  const response = await fetch(`${operatorBaseUrl()}${routePath}`, {
    method: route.method,
    headers,
    body: requestBody ? JSON.stringify(requestBody) : undefined,
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
  ipcMain.handle('operator:invoke', async (_event, operation, payload) => {
    try {
      return { ok: true, data: await invokeOperator(operation, payload) };
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
    width: 1240,
    height: 880,
    minWidth: 920,
    minHeight: 680,
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
