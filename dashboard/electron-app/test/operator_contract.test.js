const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const root = path.resolve(__dirname, '..');
const read = (relativePath) => fs.readFileSync(path.join(root, relativePath), 'utf8');

test('packaged renderer is the restricted operator console', () => {
  const packageJson = JSON.parse(read('package.json'));
  const packagedFiles = packageJson.build.files;

  assert.ok(packagedFiles.includes('renderer/operator.html'));
  assert.ok(packagedFiles.includes('renderer/operator.js'));
  assert.ok(!packagedFiles.includes('renderer/index.html'));
});

test('renderer has no network, token storage, arbitrary orders, or live activation', () => {
  const html = read('renderer/operator.html');
  const script = read('renderer/operator.js');

  assert.match(html, /connect-src 'none'/);
  assert.match(html, /Research Lab/);
  assert.match(html, /manual review/i);
  assert.doesNotMatch(script, /\bfetch\s*\(/);
  assert.doesNotMatch(script, /localStorage|sessionStorage/);
  assert.doesNotMatch(script, /Authorization|Bearer|TRADING_OPERATOR_TOKEN/);
  assert.doesNotMatch(script, /submitOrder|placeOrder|sendOrder|resetKillSwitch/);
  assert.doesNotMatch(script, /enableLive|startLive|promoteLive|activateLive/);
});

test('preload exposes only approved trading and bounded research actions', () => {
  const preload = read('preload.js');
  const expectedMethods = [
    'getStatus',
    'getOrders',
    'getPositions',
    'startPaper',
    'pause',
    'stop',
    'cancelAll',
    'emergencyStop',
    'getResearchDatasets',
    'getResearchJobs',
    'startResearch',
    'cancelResearch'
  ];

  for (const method of expectedMethods) {
    assert.match(preload, new RegExp(`\\b${method}\\b`));
  }
  assert.doesNotMatch(preload, /submitOrder|placeOrder|resetKillSwitch|setRisk/);
  assert.doesNotMatch(preload, /startLive|enableLive|promoteLive|activateLive/);
});

test('main process enforces loopback, fixed routes, and bounded lab inputs', () => {
  const main = read('main.js');

  assert.match(main, /127\.0\.0\.1:8765/);
  assert.match(main, /loopback address/);
  assert.match(main, /Unsupported operator action/);
  assert.match(main, /accounts_per_generation/);
  assert.match(main, /final_holdout_fraction/);
  assert.match(main, /JOB_ID_PATTERN/);
  assert.doesNotMatch(main, /\/operator\/order(?:s\/submit)?['"`]/);
  assert.doesNotMatch(main, /reset-kill|risk-limit|live-mode/);
  assert.doesNotMatch(main, /eval\s*\(|new Function/);
});
