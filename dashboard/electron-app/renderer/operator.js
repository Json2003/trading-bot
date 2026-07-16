'use strict';

const bridge = window.tradingOperator;
const POLL_INTERVAL_MS = 5000;

const elements = {
  app: document.getElementById('app'),
  notice: document.getElementById('notice'),
  connection: document.getElementById('connection'),
  connectionText: document.getElementById('connectionText'),
  mode: document.getElementById('mode'),
  state: document.getElementById('state'),
  killSwitch: document.getElementById('killSwitch'),
  openOrders: document.getElementById('openOrders'),
  openPositions: document.getElementById('openPositions'),
  ordersBody: document.getElementById('ordersBody'),
  positionsBody: document.getElementById('positionsBody'),
  ordersEmpty: document.getElementById('ordersEmpty'),
  positionsEmpty: document.getElementById('positionsEmpty'),
  ordersCount: document.getElementById('ordersCount'),
  positionsCount: document.getElementById('positionsCount'),
  lastUpdated: document.getElementById('lastUpdated'),
  startButton: document.getElementById('startButton'),
  pauseButton: document.getElementById('pauseButton'),
  stopButton: document.getElementById('stopButton'),
  cancelButton: document.getElementById('cancelButton'),
  refreshButton: document.getElementById('refreshButton'),
  emergencyButton: document.getElementById('emergencyButton')
};

let latestStatus = null;
let busy = false;
let pollTimer = null;

function showNotice(message) {
  elements.notice.textContent = message || '';
  elements.notice.classList.toggle('visible', Boolean(message));
}

function setConnected(connected, message) {
  elements.connection.classList.toggle('online', connected);
  elements.connectionText.textContent = message;
}

function formatNumber(value, digits = 4) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function shortId(value) {
  const id = String(value || '—');
  return id.length > 14 ? `${id.slice(0, 7)}…${id.slice(-5)}` : id;
}

function appendCell(row, value, title = null) {
  const cell = document.createElement('td');
  cell.textContent = value == null || value === '' ? '—' : String(value);
  if (title) cell.title = title;
  row.appendChild(cell);
}

function renderOrders(orders) {
  const items = Array.isArray(orders) ? orders : [];
  elements.ordersBody.replaceChildren();
  elements.ordersEmpty.hidden = items.length > 0;
  elements.ordersCount.textContent = `${items.length} record${items.length === 1 ? '' : 's'}`;

  for (const order of items) {
    const row = document.createElement('tr');
    const fullId = String(order.id || order.client_order_id || '—');
    appendCell(row, shortId(fullId), fullId);
    appendCell(row, order.symbol);
    appendCell(row, order.side);
    appendCell(row, formatNumber(order.quantity ?? order.qty));
    appendCell(row, formatNumber(order.filled_quantity ?? order.filled_qty ?? 0));
    appendCell(row, formatNumber(order.price, 4));
    appendCell(row, order.status || 'open');
    elements.ordersBody.appendChild(row);
  }
}

function renderPositions(positions) {
  const items = Array.isArray(positions) ? positions : [];
  elements.positionsBody.replaceChildren();
  elements.positionsEmpty.hidden = items.length > 0;
  elements.positionsCount.textContent = `${items.length} record${items.length === 1 ? '' : 's'}`;

  for (const position of items) {
    const row = document.createElement('tr');
    appendCell(row, position.symbol);
    appendCell(row, formatNumber(position.quantity ?? position.qty));
    appendCell(row, formatNumber(position.average_price ?? position.avg_price, 4));
    elements.positionsBody.appendChild(row);
  }
}

function renderStatus(status) {
  latestStatus = status || null;
  if (!latestStatus) {
    elements.mode.textContent = '—';
    elements.state.textContent = 'offline';
    elements.state.className = 'metric-value small state-stopped';
    elements.killSwitch.textContent = 'unknown';
    elements.killSwitch.className = 'metric-value small state-stopped';
    elements.openOrders.textContent = '0';
    elements.openPositions.textContent = '0';
    updateControls();
    return;
  }

  const serviceState = String(latestStatus.state || 'unknown');
  elements.mode.textContent = `${String(latestStatus.mode || '—')} · ${latestStatus.engine_configured ? 'engine ready' : 'engine missing'}`;
  elements.state.textContent = serviceState;
  elements.state.className = `metric-value small ${serviceState === 'faulted' ? 'state-killed' : `state-${serviceState}`}`;
  elements.killSwitch.textContent = latestStatus.kill_switch_latched ? 'latched' : 'clear';
  elements.killSwitch.className = `metric-value small ${latestStatus.kill_switch_latched ? 'state-killed' : 'state-running'}`;
  elements.openOrders.textContent = String(Number(latestStatus.open_orders || 0));
  elements.openPositions.textContent = String(Number(latestStatus.open_positions || 0));

  if (latestStatus.kill_switch_latched) {
    showNotice(latestStatus.last_error || 'Kill switch is latched. New starts are blocked and manual recovery outside the operator console is required.');
  } else if (!latestStatus.engine_configured) {
    showNotice('Operator API is online, but no trading engine is configured. Start is disabled until the repaired paper engine is attached.');
  } else if (latestStatus.last_error) {
    showNotice(latestStatus.last_error);
  } else {
    showNotice('');
  }
  updateControls();
}

function updateControls() {
  const online = Boolean(latestStatus);
  const engineReady = Boolean(latestStatus && latestStatus.engine_configured);
  const killed = Boolean(latestStatus && latestStatus.kill_switch_latched);
  const state = latestStatus ? latestStatus.state : 'offline';
  const openOrders = Number(latestStatus && latestStatus.open_orders || 0);

  elements.startButton.disabled = busy || !online || !engineReady || killed || state === 'running';
  elements.pauseButton.disabled = busy || !online || killed || state !== 'running';
  elements.stopButton.disabled = busy || !online || state === 'stopped';
  elements.cancelButton.disabled = busy || !online || openOrders < 1;
  elements.refreshButton.disabled = busy;
  elements.emergencyButton.disabled = busy || !online || killed;
  elements.app.classList.toggle('busy', busy);
}

async function callBridge(methodName) {
  if (!bridge || typeof bridge[methodName] !== 'function') {
    throw new Error('Secure Electron operator bridge is unavailable');
  }
  const response = await bridge[methodName]();
  if (!response || response.ok !== true) {
    const error = new Error(response && response.error ? response.error : 'Operator request failed');
    error.status = Number(response && response.status) || 0;
    throw error;
  }
  return response.data || {};
}

function updateRefreshLabel(status) {
  const timestamp = new Date().toLocaleTimeString();
  if (!status) {
    elements.lastUpdated.textContent = `Refresh failed ${timestamp}`;
    return;
  }
  const cycles = Number(status.cycle_count || 0);
  const lastCycle = status.last_cycle_at ? new Date(status.last_cycle_at).toLocaleTimeString() : 'none';
  elements.lastUpdated.textContent = `Updated ${timestamp} · cycles ${cycles} · last cycle ${lastCycle}`;
}

async function refresh() {
  if (busy) return;
  try {
    const [statusPayload, ordersPayload, positionsPayload] = await Promise.all([
      callBridge('getStatus'),
      callBridge('getOrders'),
      callBridge('getPositions')
    ]);
    renderStatus(statusPayload.status);
    renderOrders(ordersPayload.orders);
    renderPositions(positionsPayload.positions);
    setConnected(true, latestStatus && latestStatus.engine_configured ? 'Operator API and engine ready' : 'Operator API connected');
    updateRefreshLabel(latestStatus);
  } catch (error) {
    latestStatus = null;
    renderStatus(null);
    renderOrders([]);
    renderPositions([]);
    setConnected(false, 'Operator API offline');
    showNotice(error instanceof Error ? error.message : String(error));
    updateRefreshLabel(null);
  }
}

function confirmationFor(action) {
  switch (action) {
    case 'stop':
      return 'Stop paper trading and cancel all open orders?';
    case 'cancelAll':
      return 'Cancel every open order?';
    case 'emergencyStop':
      return 'Trigger the emergency stop? This latches the kill switch and requires manual recovery.';
    default:
      return null;
  }
}

async function runAction(action) {
  if (action === 'refresh') {
    await refresh();
    return;
  }

  const promptText = confirmationFor(action);
  if (promptText && !window.confirm(promptText)) return;

  busy = true;
  updateControls();
  showNotice('');
  try {
    await callBridge(action);
  } catch (error) {
    showNotice(error instanceof Error ? error.message : String(error));
  } finally {
    busy = false;
    updateControls();
    await refresh();
  }
}

for (const button of document.querySelectorAll('[data-action]')) {
  button.addEventListener('click', () => runAction(button.dataset.action));
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refresh();
});

window.addEventListener('beforeunload', () => {
  if (pollTimer) window.clearInterval(pollTimer);
});

refresh();
pollTimer = window.setInterval(() => {
  if (!document.hidden) refresh();
}, POLL_INTERVAL_MS);
