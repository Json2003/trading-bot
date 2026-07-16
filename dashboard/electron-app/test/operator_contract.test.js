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

test('renderer has no network, token storage, or arbitrary order capability', () => {
  const html = read('renderer/operator.html');
  const script = read('renderer/operator.js');

  assert.match(html, /connect-src 'none'/);
  assert.doesNotMatch(script, /\bfetch\s*\(/);
  assert.doesNotMatch(script, /localStorage|sessionStorage/);
  assert.doesNotMatch(script, /Authorization|Bearer|TRADING_OPERATOR_TOKEN/);
  assert.doesNotMatch(script, /submitOrder|placeOrder|sendOrder|resetKillSwitch/);
});

test('preload exposes only the approved operator actions', () => {
  const preload = read('preload.js');
  const expectedMethods = [
    'getStatus',
    'getOrders',
    'getPositions',
    'startPaper',
    'pause',
    'stop',
    'cancelAll',
    'emergencyStop'
  ];

  for (const method of expectedMethods) {
    assert.match(preload, new RegExp(`\\b${method}\\b`));
  }
  assert.doesNotMatch(preload, /submitOrder|placeOrder|resetKillSwitch|setRisk/);
});

test('main process enforces loopback API and a fixed route allowlist', () => {
  const main = read('main.js');

  assert.match(main, /127\.0\.0\.1:8765/);
  assert.match(main, /loopback address/);
  assert.match(main, /Unsupported operator action/);
  assert.doesNotMatch(main, /\/operator\/order(?:s\/submit)?['"`]/);
  assert.doesNotMatch(main, /reset-kill|risk-limit|live-mode/);
});
