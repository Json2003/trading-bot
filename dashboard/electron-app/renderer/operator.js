'use strict';

const bridge = window.tradingOperator;
const POLL_INTERVAL_MS = 5000;
const ACTIVE_LAB_STATES = new Set(['queued', 'running']);

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
  emergencyButton: document.getElementById('emergencyButton'),
  labState: document.getElementById('labState'),
  labDataset: document.getElementById('labDataset'),
  labGenerations: document.getElementById('labGenerations'),
  labAccounts: document.getElementById('labAccounts'),
  labHoldout: document.getElementById('labHoldout'),
  labSeed: document.getElementById('labSeed'),
  labStartButton: document.getElementById('labStartButton'),
  labCancelButton: document.getElementById('labCancelButton'),
  labRefreshButton: document.getElementById('labRefreshButton'),
  labJobState: document.getElementById('labJobState'),
  labGeneration: document.getElementById('labGeneration'),
  labCandidates: document.getElementById('labCandidates'),
  labLeaderboardType: document.getElementById('labLeaderboardType'),
  labLeaderboardCount: document.getElementById('labLeaderboardCount'),
  labLeaderboardBody: document.getElementById('labLeaderboardBody'),
  labLeaderboardEmpty: document.getElementById('labLeaderboardEmpty')
};

let latestStatus = null;
let latestLabJob = null;
let tradingBusy = false;
let labBusy = false;
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

function formatCurrency(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  return number.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  return `${(number * 100).toFixed(2)}%`;
}

function shortId(value) {
  const id = String(value || '—');
  return id.length > 14 ? `${id.slice(0, 7)}…${id.slice(-5)}` : id;
}

function appendCell(row, value, title = null, className = '') {
  const cell = document.createElement('td');
  cell.textContent = value == null || value === '' ? '—' : String(value);
  if (title) cell.title = title;
  if (className) cell.className = className;
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
    updateTradingControls();
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
  updateTradingControls();
}

function updateTradingControls() {
  const online = Boolean(latestStatus);
  const engineReady = Boolean(latestStatus && latestStatus.engine_configured);
  const killed = Boolean(latestStatus && latestStatus.kill_switch_latched);
  const state = latestStatus ? latestStatus.state : 'offline';
  const openOrders = Number(latestStatus && latestStatus.open_orders || 0);

  elements.startButton.disabled = tradingBusy || !online || !engineReady || killed || state === 'running';
  elements.pauseButton.disabled = tradingBusy || !online || killed || state !== 'running';
  elements.stopButton.disabled = tradingBusy || !online || state === 'stopped';
  elements.cancelButton.disabled = tradingBusy || !online || openOrders < 1;
  elements.refreshButton.disabled = tradingBusy;
  elements.emergencyButton.disabled = tradingBusy || !online || killed;
  elements.app.classList.toggle('busy', tradingBusy);
}

function renderDatasets(datasets) {
  const items = Array.isArray(datasets) ? datasets : [];
  const selected = elements.labDataset.value;
  elements.labDataset.replaceChildren();
  if (items.length === 0) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No CSV datasets found';
    elements.labDataset.appendChild(option);
    return;
  }
  for (const dataset of items) {
    const option = document.createElement('option');
    option.value = String(dataset.dataset_id || '');
    option.textContent = `${option.value} · ${formatNumber(Number(dataset.size_bytes || 0) / 1024, 1)} KB`;
    elements.labDataset.appendChild(option);
  }
  if (items.some((item) => String(item.dataset_id) === selected)) {
    elements.labDataset.value = selected;
  }
}

function selectCurrentJob(jobs) {
  const items = Array.isArray(jobs) ? jobs : [];
  return items.find((job) => ACTIVE_LAB_STATES.has(String(job.state))) || items[0] || null;
}

function renderLeaderboard(job) {
  const finalItems = Array.isArray(job && job.final_leaderboard) ? job.final_leaderboard : [];
  const developmentItems = Array.isArray(job && job.development_leaderboard) ? job.development_leaderboard : [];
  const items = finalItems.length > 0 ? finalItems : developmentItems;
  const type = finalItems.length > 0 ? 'final holdout' : (developmentItems.length > 0 ? 'development' : 'none');

  elements.labLeaderboardBody.replaceChildren();
  elements.labLeaderboardEmpty.hidden = items.length > 0;
  elements.labLeaderboardCount.textContent = `${items.length} candidate${items.length === 1 ? '' : 's'}`;
  elements.labLeaderboardType.textContent = type;

  items.forEach((candidate, index) => {
    const row = document.createElement('tr');
    appendCell(row, index + 1);
    appendCell(row, candidate.strategy);
    appendCell(row, candidate.execution_policy);
    appendCell(row, formatCurrency(candidate.ending_equity));
    const totalReturn = Number(candidate.total_return);
    appendCell(row, formatPercent(totalReturn), null, totalReturn >= 0 ? 'positive' : 'negative');
    appendCell(row, formatPercent(candidate.max_drawdown), null, 'negative');
    appendCell(row, formatNumber(candidate.sharpe, 2));
    appendCell(row, formatNumber(candidate.trade_count, 0));
    appendCell(row, formatCurrency(candidate.average_execution_cost));
    appendCell(row, formatNumber(candidate.score, 4));
    elements.labLeaderboardBody.appendChild(row);
  });
}

function renderLabJob(job) {
  latestLabJob = job || null;
  if (!latestLabJob) {
    elements.labState.textContent = 'Ready';
    elements.labJobState.textContent = 'idle';
    elements.labJobState.className = '';
    elements.labGeneration.textContent = '0 / 0';
    elements.labCandidates.textContent = '0';
    renderLeaderboard(null);
    updateLabControls();
    return;
  }

  const rawState = String(latestLabJob.state || 'unknown');
  const displayState = latestLabJob.cancellation_requested && ACTIVE_LAB_STATES.has(rawState) ? 'cancelling' : rawState;
  elements.labState.textContent = `Job ${shortId(latestLabJob.job_id)} · ${displayState}`;
  elements.labJobState.textContent = displayState;
  elements.labJobState.className = `state-${displayState}`;
  elements.labGeneration.textContent = `${Number(latestLabJob.generation || 0)} / ${Number(latestLabJob.total_generations || 0)}`;
  elements.labCandidates.textContent = String(Number(latestLabJob.candidates_evaluated || 0));
  renderLeaderboard(latestLabJob);
  if (latestLabJob.error) showNotice(latestLabJob.error);
  updateLabControls();
}

function updateLabControls() {
  const active = Boolean(latestLabJob && ACTIVE_LAB_STATES.has(String(latestLabJob.state)));
  const unavailable = elements.labDataset.options.length === 0 || !elements.labDataset.value;
  elements.labStartButton.disabled = labBusy || active || unavailable;
  elements.labCancelButton.disabled = labBusy || !active;
  elements.labRefreshButton.disabled = labBusy;
  elements.labDataset.disabled = labBusy || active;
  elements.labGenerations.disabled = labBusy || active;
  elements.labAccounts.disabled = labBusy || active;
  elements.labHoldout.disabled = labBusy || active;
  elements.labSeed.disabled = labBusy || active;
}

async function callBridge(methodName, ...args) {
  if (!bridge || typeof bridge[methodName] !== 'function') {
    throw new Error('Secure Electron operator bridge is unavailable');
  }
  const response = await bridge[methodName](...args);
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

async function refreshTrading() {
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
}

async function refreshLab() {
  try {
    const [datasetsPayload, jobsPayload] = await Promise.all([
      callBridge('getResearchDatasets'),
      callBridge('getResearchJobs')
    ]);
    renderDatasets(datasetsPayload.datasets);
    renderLabJob(selectCurrentJob(jobsPayload.jobs));
  } catch (error) {
    latestLabJob = null;
    elements.labState.textContent = 'Research lab unavailable';
    elements.labJobState.textContent = 'offline';
    elements.labJobState.className = 'state-failed';
    renderLeaderboard(null);
    updateLabControls();
    throw error;
  }
}

async function refresh() {
  if (tradingBusy || labBusy) return;
  try {
    await refreshTrading();
  } catch (error) {
    latestStatus = null;
    renderStatus(null);
    renderOrders([]);
    renderPositions([]);
    setConnected(false, 'Operator API offline');
    showNotice(error instanceof Error ? error.message : String(error));
    updateRefreshLabel(null);
    return;
  }

  try {
    await refreshLab();
  } catch (error) {
    showNotice(`Trading service is connected, but the Research Lab failed: ${error instanceof Error ? error.message : String(error)}`);
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

async function runTradingAction(action) {
  if (action === 'refresh') {
    await refresh();
    return;
  }

  const promptText = confirmationFor(action);
  if (promptText && !window.confirm(promptText)) return;

  tradingBusy = true;
  updateTradingControls();
  showNotice('');
  try {
    await callBridge(action);
  } catch (error) {
    showNotice(error instanceof Error ? error.message : String(error));
  } finally {
    tradingBusy = false;
    updateTradingControls();
    await refresh();
  }
}

function labSpecFromForm() {
  return {
    datasetId: elements.labDataset.value,
    generations: Number(elements.labGenerations.value),
    accountsPerGeneration: Number(elements.labAccounts.value),
    finalHoldoutFraction: Number(elements.labHoldout.value) / 100,
    seed: Number(elements.labSeed.value)
  };
}

async function startLab() {
  if (!window.confirm('Run isolated paper accounts against this local dataset? Results remain research-only and cannot activate live trading.')) return;
  labBusy = true;
  updateLabControls();
  showNotice('');
  try {
    const payload = await callBridge('startResearch', labSpecFromForm());
    renderLabJob(payload.job);
  } catch (error) {
    showNotice(error instanceof Error ? error.message : String(error));
  } finally {
    labBusy = false;
    updateLabControls();
    await refreshLab().catch((error) => showNotice(error instanceof Error ? error.message : String(error)));
  }
}

async function cancelLab() {
  if (!latestLabJob || !ACTIVE_LAB_STATES.has(String(latestLabJob.state))) return;
  if (!window.confirm('Cancel the active research run? Completed generation results will remain available for review.')) return;
  labBusy = true;
  updateLabControls();
  try {
    const payload = await callBridge('cancelResearch', latestLabJob.job_id);
    renderLabJob(payload.job);
  } catch (error) {
    showNotice(error instanceof Error ? error.message : String(error));
  } finally {
    labBusy = false;
    updateLabControls();
    await refreshLab().catch((error) => showNotice(error instanceof Error ? error.message : String(error)));
  }
}

for (const button of document.querySelectorAll('[data-action]')) {
  button.addEventListener('click', () => runTradingAction(button.dataset.action));
}
elements.labStartButton.addEventListener('click', startLab);
elements.labCancelButton.addEventListener('click', cancelLab);
elements.labRefreshButton.addEventListener('click', () => refreshLab().catch((error) => showNotice(error instanceof Error ? error.message : String(error))));

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
