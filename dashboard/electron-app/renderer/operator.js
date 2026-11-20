'use strict';

const api = window.tradingOperator;
const $ = (id) => document.getElementById(id);
const all = (selector) => [...document.querySelectorAll(selector)];
const instruments = Object.freeze({ IBM: 'International Business Machines', JPM: 'JPMorgan Chase', KO: 'The Coca-Cola Company' });
const workspaces = { sim: 'Local simulator', paper: 'IBKR Paper', live: 'IBKR Live' };
const headings = {
  overview: ['Trading desk', 'Monitor simulated positions, orders and market activity.'],
  positions: ['Positions', 'Review positions for the selected workspace.'],
  orders: ['Open orders', 'Review pending orders from the selected workspace.'],
  research: ['Research Lab', 'Review isolated experiments and chronological holdout results.'],
  connection: ['TWS connection', 'Verify the account and API version for each Trader Workstation session.']
};
const dateKey = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' });
const eastern = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false });
const shortDate = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
const session = {
  mode: 'sim', page: 'overview', ledger: 'positions', symbol: 'IBM', range: 'session',
  operator: null, settings: null, broker: { paper: null, live: null },
  brokerErrors: {}, checked: {}, refreshing: false, simulationBusy: false, twsBusy: false,
  labBusy: false, labLoaded: false, labJob: null, backendReady: false,
};
let chartHover = null;
let plotted = [];
let chartTransform = null;
let chartLayer = null;

const isNumber = (v) => typeof v === 'number' && Number.isFinite(v);
const safeArray = (v) => Array.isArray(v) ? v : [];
const number = (n, digits = 2) => isNumber(n) ? n.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
const signedPercent = (n) => isNumber(n) ? (n >= 0 ? '+' : '') + (n * 100).toFixed(2) + '%' : '—';
const currency = (v, unit = 'USD') => {
  if (!isNumber(v)) return '—';
  if (!/^[A-Z]{3}$/.test(unit) || unit === 'BASE') return number(v, 2) + ' ' + (unit || 'account currency');
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: unit }).format(v);
};
const timestamp = (raw, format = shortDate) => {
  const time = raw ? new Date(raw) : null;
  return time && Number.isFinite(time.getTime()) ? format.format(time) + ' ET' : '—';
};
const count = (items, label) => items.length + ' ' + label + (items.length === 1 ? '' : 's');
function set(id, value) { $(id).textContent = value; }
function tone(id, name) { $(id).dataset.tone = name; }
function pressed(selector, field, value) {
  for (const element of all(selector)) element.setAttribute('aria-pressed', String(element.dataset[field] === value));
}
function announce(message, color = 'warn') {
  $('notice').hidden = !message;
  set('notice', message || '');
  tone('notice', color);
}
function connection(label, color) { set('connectionText', label); tone('connection', color); }

async function request(method, ...args) {
  if (!api || typeof api[method] !== 'function') throw new Error('Local operator bridge is unavailable');
  const result = await api[method](...args);
  if (!result?.ok) {
    const error = new Error(result?.error || 'Operator request failed');
    error.status = result?.status;
    throw error;
  }
  return result.data || {};
}
function view() {
  const simulation = session.mode === 'sim';
  const snapshot = simulation ? session.operator : session.broker[session.mode];
  const currentCharts = simulation ? (snapshot?.workspace?.charts ?? {}) :
    (snapshot?.chart ? { [snapshot.chart.symbol]: snapshot.chart.bars } : {});
  return { simulation, snapshot, charts: currentCharts, label: workspaces[session.mode] };
}
function switchPage(page, focus = true) {
  if (!(page in headings)) return;
  session.page = page;
  if (session.backendReady) announce(null);
  if (['positions', 'orders'].includes(page)) session.ledger = page;
  render();
  if (focus) $('pageTitle').focus();
  if (page === 'research') void refreshLab();
  if (page === 'connection' && !session.settings) void refresh();
}
function switchMode(mode) {
  if (!(mode in workspaces) || mode === session.mode) return;
  session.mode = mode;
  chartHover = null;
  announce(null);
  render();
}
function render() {
  const { simulation, snapshot, charts, label } = view();
  const overview = session.page === 'overview';
  const title = overview && !simulation ? label + ' desk' : headings[session.page][0];
  const description = overview && !simulation ? 'Inspect your verified Trader Workstation account and historical market data.' : headings[session.page][1];
  set('pageTitle', title);
  set('pageDescription', description);
  set('pageKicker', session.page === 'research' ? 'STRATEGY RESEARCH' : session.page === 'connection' ? 'BROKER CONNECTOR' : (simulation ? 'NYSE WORKSPACE' : 'INTERACTIVE BROKERS'));
  for (const button of all('.nav-item[data-page]')) {
    if (button.dataset.page === session.page) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  }
  pressed('[data-mode]', 'mode', session.mode);
  pressed('[data-ledger]', 'ledger', session.ledger);
  $('overviewPanel').hidden = !overview;
  $('activityPanel').hidden = !['overview', 'positions', 'orders'].includes(session.page);
  $('researchPanel').hidden = session.page !== 'research';
  $('connectionPanel').hidden = session.page !== 'connection';
  renderBrokerSettings();
  if (overview) {
    renderSummary(simulation, snapshot);
    renderWatchlist(charts);
    renderChart(charts[session.symbol], simulation, snapshot);
  }
  if (!$('activityPanel').hidden) renderActivity(simulation, snapshot, label);
  if (!$('researchPanel').hidden) renderLabJob();
  updateControls();
}
function renderSummary(simulation, snapshot) {
  const balances = snapshot?.balances ?? {};
  const workspace = snapshot?.workspace;
  const status = snapshot?.status;
  const online = Boolean(snapshot);
  set('equityLabel', simulation ? 'Simulated equity' : 'Net liquidation value');
  set('cashLabel', simulation ? 'Simulated cash' : 'Settled cash');
  set('engineLabel', simulation ? 'Simulator status' : 'TWS account check');
  set('equity', simulation ? currency(workspace?.equity) : currency(balances.equity?.value, balances.equity?.currency));
  set('cash', simulation ? currency(workspace?.cash) : currency(balances.settled_cash?.value, balances.settled_cash?.currency));
  if (simulation) {
    set('equityDetail', isNumber(workspace?.starting_cash) ? 'Started at ' + currency(workspace.starting_cash) : 'Local simulator');
    set('cashDetail', 'USD · synthetic fills');
  } else {
    set('equityDetail', snapshot ? 'Account ' + snapshot.account : 'Awaiting verified snapshot');
    const funds = balances.available_funds;
    set('cashDetail', funds ? 'Available ' + currency(funds.value, funds.currency) : 'Reported by IBKR');
  }
  set('positionMetric', online ? number(safeArray(snapshot.positions).length, 0) : '—');
  set('positionDetail', simulation ? 'Local simulator' : 'Broker-reported snapshot');
  const check = session.checked[session.mode];
  const fault = session.brokerErrors[session.mode];
  if (simulation) {
    const state = status?.state;
    const statusText = status?.kill_switch_latched ? 'Stop latched' : state ? state[0].toUpperCase() + state.slice(1) : 'Offline';
    set('engineStatus', statusText);
    tone('engineStatus', status?.kill_switch_latched || state === 'faulted' ? 'bad' : state === 'running' ? 'good' : state === 'paused' ? 'warn' : 'neutral');
    set('engineDetail', status ? number(status.cycle_count, 0) + ' completed cycles' : 'Operator unavailable');
    $('simGuard').lastElementChild.textContent = status?.kill_switch_latched ? 'Safety stop latched · restart disabled' :
      status?.engine_configured ? 'Synthetic fills · no IBKR orders' : 'No simulation engine attached';
    tone('simGuard', status?.kill_switch_latched ? 'bad' : 'neutral');
  } else {
    set('engineStatus', online ? 'Verified' : fault ? 'Check failed' : 'Not checked');
    tone('engineStatus', online ? 'good' : fault ? 'bad' : 'neutral');
    set('engineDetail', online ? 'Snapshot · ' + timestamp(snapshot.captured_at) : check ? 'Last attempt · ' + timestamp(check) : 'Connection is checked on request');
  }
  $('simControls').hidden = !simulation;
  $('twsControls').hidden = simulation;
  if (!simulation) {
    const profile = session.settings?.profiles?.[session.mode];
    set('twsControlTitle', workspaces[session.mode] + ' connection');
    const localState = session.operator?.status?.state;
    set('twsControlDetail', 'Trader Workstation · port ' + (profile?.port ?? (session.mode === 'paper' ? 7497 : 7496)) +
      (localState === 'running' || localState === 'paused' ? ' · simulator ' + localState : ''));
  }
}
function validatedBars(raw) {
  return safeArray(raw).filter((b) =>
    b && Number.isFinite(Date.parse(b.time)) && [b.open, b.high, b.low, b.close].every(isNumber) &&
    b.low > 0 && b.low <= Math.min(b.open, b.close) && b.high >= Math.max(b.open, b.close)
  ).sort((a, b) => Date.parse(a.time) - Date.parse(b.time));
}
function sessionBars(bars) {
  if (!bars.length) return [];
  const date = dateKey.format(new Date(bars.at(-1).time));
  return bars.filter((bar) => dateKey.format(new Date(bar.time)) === date);
}
function sessionMove(bars) {
  const current = sessionBars(bars);
  return current.length ? current.at(-1).close / current[0].open - 1 : null;
}
function moveColour(n) { return !isNumber(n) || n === 0 ? 'neutral' : n > 0 ? 'good' : 'bad'; }
function renderWatchlist(charts) {
  pressed('[data-symbol]', 'symbol', session.symbol);
  for (const symbol of Object.keys(instruments)) {
    const bars = validatedBars(charts[symbol]);
    const value = document.querySelector('[data-last="' + symbol + '"]');
    value.textContent = bars.length ? currency(bars.at(-1).close) : '—';
    const move = document.querySelector('[data-change="' + symbol + '"]');
    const r = sessionMove(bars);
    move.textContent = signedPercent(r);
    move.dataset.tone = moveColour(r);
  }
}
function svgElement(tag, attrs, value) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, attr] of Object.entries(attrs)) el.setAttribute(key, String(attr));
  if (value !== undefined) el.textContent = value;
  return el;
}
function renderChart(raw, simulation, snapshot) {
  const bars = validatedBars(raw);
  set('symbolTitle', session.symbol);
  set('symbolMark', session.symbol);
  const mark = $('symbolMark');
  mark.className = 'instrument-mark tile-' + session.symbol.toLowerCase();
  set('symbolName', instruments[session.symbol]);
  pressed('[data-range]', 'range', session.range);
  set('lastPrice', currency(bars.at(-1)?.close));
  const move = sessionMove(bars);
  set('priceChange', signedPercent(move));
  tone('priceChange', moveColour(move));
  set('chartSource', simulation ? 'SYNTHETIC · 5 MIN' : 'TWS HISTORICAL · 5 MIN');
  const focusBars = session.range === 'all' ? bars : session.range === 'hour' ? bars.slice(-12) : sessionBars(bars);
  const graph = $('chart');
  graph.replaceChildren();
  $('chartEmpty').hidden = focusBars.length > 0;
  graph.hidden = !focusBars.length;
  chartLayer = null;
  chartTransform = null;
  plotted = focusBars;
  if (!focusBars.length) {
    const msg = simulation ? 'Start the local NYSE demo to generate bars.' :
      session.twsBusy ? 'Checking Trader Workstation…' :
        session.brokerErrors[session.mode] || (snapshot?.chart?.symbol === session.symbol ? snapshot.chart.message : null) ||
        'Check this TWS account to load historical bars for ' + session.symbol + '.';
    $('chartEmpty').lastElementChild.textContent = msg;
    set('barDetails', 'OHLC · Volume');
    return;
  }
  const height = 320;
  const width = Math.max(300, Math.round(graph.getBoundingClientRect().width) || 800);
  graph.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
  graph.setAttribute('aria-label', session.symbol + ' · ' + focusBars.length + ' five-minute bars · ' + (simulation ? 'synthetic' : 'TWS historical'));
  const left = 14, right = width - (width < 450 ? 62 : 71), top = 12, bottom = 248, volumeY = 286;
  const min = Math.min(...focusBars.map(b => b.low));
  const max = Math.max(...focusBars.map(b => b.high));
  const padding = Math.max((max - min) * 0.08, max * 0.0008, 0.01);
  const lower = min - padding, upper = max + padding;
  const priceY = (v) => top + ((upper - v) / (upper - lower)) * (bottom - top);
  const dx = (right - left) / focusBars.length;
  const x = (i) => left + dx * (i + 0.5);
  const vMax = Math.max(1, ...focusBars.map(b => isNumber(b.volume) && b.volume > 0 ? b.volume : 0));
  const nodes = [];
  for (let i = 0; i <= 5; i++) {
    const p = upper - (upper - lower) * i / 5, y = priceY(p);
    nodes.push(svgElement('line', { x1: left, x2: right, y1: y, y2: y, class: 'gridline' }));
    nodes.push(svgElement('text', { x: right + 7, y: y + 4, class: 'axis-label' }, p.toFixed(2)));
  }
  const steps = width < 450 ? 3 : 5;
  for (let i = 0; i <= steps; i++) {
    const j = Math.round(i * (focusBars.length - 1) / steps);
    const x0 = x(j);
    nodes.push(svgElement('line', { x1: x0, x2: x0, y1: top, y2: volumeY, class: 'gridline' }));
    nodes.push(svgElement('text', { x: x0, y: 308, class: 'axis-label', 'text-anchor': i === 0 ? 'start' : i === steps ? 'end' : 'middle' },
      (session.range === 'all' && steps === 5 ? shortDate : eastern).format(new Date(focusBars[j].time))));
  }
  focusBars.forEach((bar, i) => {
    const xp = x(i), y1 = priceY(bar.open), y2 = priceY(bar.close);
    const style = bar.close >= bar.open ? 'up' : 'down';
    const candle = Math.max(1, Math.min(13, dx * 0.64));
    const vol = isNumber(bar.volume) ? 31 * Math.max(0, bar.volume) / vMax : 0;
    nodes.push(svgElement('line', { x1: xp, x2: xp, y1: priceY(bar.high), y2: priceY(bar.low), class: style + ' wick' }));
    nodes.push(svgElement('rect', { x: xp - candle / 2, width: candle, y: Math.min(y1, y2), height: Math.max(1, Math.abs(y2 - y1)), class: style, rx: .55 }));
    if (vol) nodes.push(svgElement('rect', { x: xp - candle / 2, width: candle, y: volumeY - vol, height: vol, class: style + ' volume' }));
  });
  const latest = priceY(focusBars.at(-1).close);
  nodes.push(svgElement('line', { x1: left, x2: right, y1: latest, y2: latest, class: 'reference' }));
  chartLayer = svgElement('g', {});
  graph.append(...nodes, chartLayer);
  chartTransform = { width, left, x, dx, top, volumeY };
  renderCrosshair();
}
function renderCrosshair() {
  if (!chartLayer || !chartTransform || !plotted.length) return;
  const i = chartHover === null ? plotted.length - 1 : Math.min(plotted.length - 1, Math.max(0, chartHover));
  chartLayer.replaceChildren();
  if (chartHover !== null) {
    const x = chartTransform.x(i);
    chartLayer.append(svgElement('line', { x1: x, x2: x, y1: chartTransform.top, y2: chartTransform.volumeY, class: 'focus-line' }));
  }
  const b = plotted[i];
  set('barDetails', timestamp(b.time) + '  ·  O ' + number(b.open) + '  H ' + number(b.high) +
    '  L ' + number(b.low) + '  C ' + number(b.close) + '  ·  V ' + number(b.volume, 0));
}
function row(cells) {
  const tr = document.createElement('tr');
  for (const [value, className, title] of cells.map(v => Array.isArray(v) ? v : [v])) {
    const td = document.createElement('td');
    td.textContent = value == null || value === '' ? '—' : String(value);
    if (className) td.className = className;
    if (title) td.title = title;
    tr.append(td);
  }
  return tr;
}
function renderActivity(simulation, snapshot, label) {
  const ps = safeArray(snapshot?.positions);
  const os = simulation ? safeArray(snapshot?.orders) : [];
  const id = snapshot?.account ? ' · ' + snapshot.account : '';
  set('ledgerSource', label + id + (simulation ? ' · hypothetical activity' : ' · read-only snapshot'));
  $('positionTable').hidden = session.ledger !== 'positions';
  $('orderTable').hidden = session.ledger !== 'orders';
  set('positionsCount', snapshot ? number(ps.length, 0) : '—');
  set('ordersCount', simulation && snapshot ? number(os.length, 0) : '—');
  $('positionsEmpty').hidden = Boolean(ps.length);
  $('ordersEmpty').hidden = Boolean(os.length);
  set('positionsEmpty', snapshot ? 'No open positions in this snapshot.' : simulation ? 'Simulator is unavailable.' : 'Check this TWS account to see positions.');
  set('ordersEmpty', simulation ? (snapshot ? 'No pending simulated orders.' : 'Simulator is unavailable.') : 'Broker orders are not retrieved in read-only TWS mode.');
  $('positionsBody').replaceChildren(...ps.map(p => row([
    p.symbol, p.security_type || (simulation && snapshot?.workspace?.source === 'synthetic_nyse' ? 'Stock' : '—'),
    number(p.quantity ?? p.qty, 4), number(p.average_price ?? p.avg_price), p.currency || (simulation ? 'USD' : '—'),
  ])));
  $('ordersBody').replaceChildren(...os.map(o => {
    const id = String(o.id || o.client_order_id || '—');
    const color = o.side === 'buy' ? 'positive' : o.side === 'sell' ? 'negative' : '';
    return row([[id.length > 18 ? id.slice(0, 8) + '…' + id.slice(-6) : id, '', id], o.symbol, [o.side, color],
      number(o.quantity ?? o.qty, 4), number(o.filled_quantity ?? o.filled_qty, 4), number(o.price), o.status]);
  }));
}
function renderBrokerSettings() {
  for (const mode of ['paper', 'live']) {
    const profile = session.settings?.profiles?.[mode];
    const snapshot = session.broker[mode];
    const error = session.brokerErrors[mode];
    set(mode + 'Port', profile?.port ?? (mode === 'paper' ? '7497' : '7496'));
    set(mode + 'Account', profile?.account || 'Not configured');
    const installed = session.settings?.sdk_version;
    set(mode + 'Sdk', [installed ? 'ib_async ' + installed : 'Not detected',
      snapshot?.server_version ? 'TWS API v' + snapshot.server_version : null].filter(Boolean).join(' · '));
    set(mode + 'Checked', snapshot ? timestamp(snapshot.captured_at) : timestamp(session.checked[mode]));
    set(mode + 'State', snapshot ? 'Verified snapshot' : error ? 'Check failed' : profile?.configured ? 'Ready to check' : 'Not configured');
    tone(mode + 'State', snapshot ? 'good' : error ? 'bad' : 'neutral');
  }
}
function updateControls() {
  const st = session.operator?.status;
  const inhibited = session.simulationBusy || !st || session.mode !== 'sim';
  $('startButton').disabled = inhibited || st.mode !== 'paper' || !st.engine_configured || st.kill_switch_latched || st.state === 'running';
  $('startButton').lastElementChild.textContent = st?.state === 'paused' ? 'Resume simulation' : 'Start simulation';
  $('pauseButton').disabled = inhibited || st.state !== 'running' || st.kill_switch_latched;
  $('stopButton').disabled = inhibited || st.state === 'stopped';
  $('cancelButton').disabled = inhibited || st.open_orders < 1;
  $('emergencyButton').disabled = inhibited || st.kill_switch_latched;
  $('checkTws').disabled = session.twsBusy || !session.backendReady || session.mode === 'sim';
  $('checkTws').lastElementChild.textContent = session.twsBusy ? 'Checking TWS…' : 'Check TWS account';
  for (const button of all('[data-tws-check]')) button.disabled = session.twsBusy || !session.backendReady;
  $('refreshButton').disabled = session.refreshing || session.twsBusy;
  const labActive = ['queued', 'running'].includes(session.labJob?.state);
  $('labStartButton').disabled = session.labBusy || !session.labLoaded || labActive || !$('labDataset').value;
  $('labCancelButton').disabled = session.labBusy || !session.labLoaded || !labActive;
  for (const field of all('#labForm input, #labForm select')) field.disabled = session.labBusy || labActive || !session.labLoaded;
}
function option(value, title) {
  const item = document.createElement('option');
  item.value = value;
  item.textContent = title;
  return item;
}
function renderLabDatasets(datasets) {
  const items = safeArray(datasets).filter(d => d.dataset_id);
  const select = $('labDataset'), selected = select.value;
  select.replaceChildren(...(items.length ? items.map(d =>
    option(d.dataset_id, d.dataset_id + ' · ' + number(d.size_bytes / 1024, 1) + ' KB')) : [option('', 'No local CSV datasets found')]));
  if (items.some(d => d.dataset_id === selected)) select.value = selected;
}
function renderLabJob() {
  const job = session.labJob;
  const final = safeArray(job?.final_leaderboard);
  const development = safeArray(job?.development_leaderboard);
  const list = final.length ? final : development;
  const state = job?.cancellation_requested && ['queued','running'].includes(job.state) ? 'cancelling' : (job?.state || (session.labLoaded ? 'Idle' : 'Unavailable'));
  set('labState', job ? String(job.job_id || '').slice(0, 8) + ' · ' + (job.dataset_id || 'Local dataset') : session.labLoaded ? 'No experiment running' : 'Research data not loaded');
  set('labJobState', state);
  tone('labJobState', state === 'completed' ? 'good' : state === 'failed' ? 'bad' : 'neutral');
  set('labGeneration', job ? number(job.generation, 0) + ' / ' + number(job.total_generations, 0) : '—');
  set('labCandidates', number(job?.candidates_evaluated, 0));
  set('labLeaderboardType', final.length ? 'Final holdout' : development.length ? 'Development' : 'No scored run');
  set('labLeaderboardCount', count(list, 'candidate'));
  $('labLeaderboardEmpty').hidden = Boolean(list.length);
  set('labLeaderboardEmpty', job?.error || 'No research results yet.');
  $('labLeaderboardBody').replaceChildren(...list.map((c, i) => row([
    i + 1, c.strategy, c.execution_policy, currency(c.ending_equity),
    [signedPercent(c.total_return), isNumber(c.total_return) ? c.total_return >= 0 ? 'positive' : 'negative' : ''],
    isNumber(c.max_drawdown) ? (100 * c.max_drawdown).toFixed(2) + '%' : '—', number(c.sharpe),
    number(c.trade_count, 0), currency(c.average_execution_cost), number(c.score, 4),
  ])));
}
async function refreshLab() {
  if (session.labBusy || !session.backendReady) return;
  session.labBusy = true;
  updateControls();
  try {
    const [datasets, jobs] = await Promise.all([request('getResearchDatasets'), request('getResearchJobs')]);
    renderLabDatasets(datasets.datasets);
    const items = safeArray(jobs.jobs);
    session.labJob = items.find(j => ['queued', 'running'].includes(j.state)) || items[0] || null;
    session.labLoaded = true;
  } catch (error) {
    session.labLoaded = false;
    session.labJob = null;
    renderLabDatasets([]);
    if (session.page === 'research') announce(error.message, 'bad');
  } finally {
    session.labBusy = false;
    if (session.page === 'research') renderLabJob();
    updateControls();
  }
}
async function refresh() {
  if (session.refreshing || session.simulationBusy) return;
  if (typeof api?.isUnlocked === 'function' && !api.isUnlocked()) {
    $('access').hidden = false;
    connection('Operator locked', 'warn');
    return;
  }
  session.refreshing = true;
  updateControls();
  try {
    const status = await request('getStatus');
    if (!status?.status) throw new Error('Invalid operator status');
    session.operator = status;
    if (!session.backendReady) announce(null);
    session.backendReady = true;
    connection('Operator online', 'good');
    $('access').hidden = true;
    set('lastUpdated', 'Updated ' + timestamp(new Date().toISOString(), eastern));
  } catch (error) {
    session.operator = null;
    session.backendReady = false;
    session.labLoaded = false;
    session.labJob = null;
    session.settings = null;
    session.broker = { paper: null, live: null };
    connection(error.status === 401 || error.status === 403 ? 'Operator locked' : 'Operator offline', 'bad');
    set('lastUpdated', 'No current operator data');
    announce(error.message, 'bad');
    if (typeof api?.lock === 'function' && (error.status === 401 || error.status === 403)) {
      api.lock();
      $('access').hidden = false;
    }
  } finally {
    session.refreshing = false;
    render();
  }
  if (session.backendReady && !session.settings) {
    try { session.settings = await request('getTwsSettings'); }
    catch { /* An older operator can still serve simulation; TWS check will explain unavailable routes. */ }
    renderBrokerSettings();
    if (session.settings?.error && session.page === 'connection') announce(session.settings.error, 'bad');
  }
  if (session.page === 'research') await refreshLab();
}
async function checkTws(mode) {
  if (!session.backendReady || session.twsBusy || !['paper', 'live'].includes(mode)) return;
  session.broker[mode] = null;
  delete session.brokerErrors[mode];
  session.twsBusy = true;
  announce(null);
  render();
  try {
    const response = await request('checkTws', mode, session.symbol);
    if (response.mode !== mode || !response.account || !response.captured_at) throw new Error('TWS response does not match this workspace');
    if (session.backendReady) session.broker[mode] = response;
  } catch (error) {
    session.brokerErrors[mode] = error.message;
    if (session.mode === mode || session.page === 'connection') announce(workspaces[mode] + ': ' + error.message, 'bad');
  } finally {
    session.twsBusy = false;
    session.checked[mode] = new Date().toISOString();
    render();
  }
}
const actions = new Set(['startPaper', 'pause', 'stop', 'cancelAll', 'emergencyStop']);
const confirmations = {
  stop: 'Stop the simulator and cancel all simulated open orders?',
  cancelAll: 'Cancel all open orders in the local simulator?',
  emergencyStop: 'Latch the local simulator stop and cancel open simulated orders? This does not affect TWS.',
};
async function runAction(action) {
  if (session.mode !== 'sim' || !session.operator || session.simulationBusy || !actions.has(action)) return;
  if (confirmations[action] && !window.confirm(confirmations[action])) return;
  session.simulationBusy = true;
  announce(null);
  updateControls();
  try { await request(action); }
  catch (error) { announce(error.message, 'bad'); }
  finally {
    session.simulationBusy = false;
    await refresh();
  }
}
async function runLab(event) {
  event?.preventDefault();
  if ($('labStartButton').disabled || !$('labForm').reportValidity()) return;
  if (!window.confirm('Run this local research experiment? It will not affect a broker account.')) return;
  session.labBusy = true;
  updateControls();
  announce(null);
  try {
    const payload = await request('startResearch', {
      datasetId: $('labDataset').value,
      generations: Number($('labGenerations').value),
      accountsPerGeneration: Number($('labAccounts').value),
      finalHoldoutFraction: Number($('labHoldout').value) / 100,
      seed: Number($('labSeed').value),
    });
    session.labJob = payload.job;
  } catch (error) { announce(error.message, 'bad'); }
  finally { session.labBusy = false; await refreshLab(); }
}
async function cancelLab() {
  if ($('labCancelButton').disabled || !session.labJob?.job_id) return;
  if (!window.confirm('Cancel this research run? Completed results remain available.')) return;
  session.labBusy = true;
  updateControls();
  try { await request('cancelResearch', session.labJob.job_id); }
  catch (error) { announce(error.message, 'bad'); }
  finally { session.labBusy = false; await refreshLab(); }
}
for (const el of all('[data-page]')) el.addEventListener('click', () => switchPage(el.dataset.page));
for (const el of all('[data-mode]')) el.addEventListener('click', () => switchMode(el.dataset.mode));
for (const el of all('[data-ledger]')) el.addEventListener('click', () => {
  session.ledger = el.dataset.ledger;
  if (['positions', 'orders'].includes(session.page)) switchPage(session.ledger, false);
  else render();
});
for (const el of all('[data-range]')) el.addEventListener('click', () => { session.range = el.dataset.range; chartHover = null; render(); });
for (const el of all('[data-symbol]')) el.addEventListener('click', () => { session.symbol = el.dataset.symbol; chartHover = null; render(); });
for (const el of all('[data-tws-check]')) el.addEventListener('click', () => checkTws(el.dataset.twsCheck));
for (const el of all('[data-action]')) el.addEventListener('click', () => runAction(el.dataset.action));
$('checkTws').addEventListener('click', () => checkTws(session.mode));
$('refreshButton').addEventListener('click', () => refresh());
$('labForm').addEventListener('submit', runLab);
$('labCancelButton').addEventListener('click', cancelLab);
$('labDataset').addEventListener('change', updateControls);
$('accessForm').addEventListener('submit', event => {
  event.preventDefault();
  if (!api?.unlock) return;
  api.unlock($('accessToken').value);
  $('accessToken').value = '';
  announce(null);
  void refresh();
});
$('chart').addEventListener('pointermove', event => {
  if (!chartTransform || !plotted.length) return;
  const rect = $('chart').getBoundingClientRect();
  const offset = (event.clientX - rect.left) / rect.width * chartTransform.width;
  chartHover = Math.floor((offset - chartTransform.left) / chartTransform.dx);
  renderCrosshair();
});
$('chart').addEventListener('pointerleave', () => { chartHover = null; renderCrosshair(); });
const layoutObserver = new ResizeObserver(() => {
  if (session.page === 'overview') {
    const { simulation, snapshot, charts } = view();
    renderChart(charts[session.symbol], simulation, snapshot);
  }
});
layoutObserver.observe($('chartStage'));
$('pageTitle').tabIndex = -1;
function updateClock() { set('clock', timestamp(new Date().toISOString(), eastern)); }
updateClock();
render();
void refresh();
window.setInterval(updateClock, 15_000);
window.setInterval(() => { if (!document.hidden) void refresh(); }, 4000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) void refresh(); });
