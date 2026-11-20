'use strict';

(() => {
  // The browser UI lives on the operator's loopback origin. This bridge is
  // omitted from Electron. Keep the operator token in memory for this tab only.
  let token = '';
  async function invoke(path, options = {}) {
    if (!token) return { ok: false, status: 401, error: 'Enter the local operator token' };
    try {
      const response = await fetch(path, {
        method: options.method || 'GET',
        credentials: 'omit',
        redirect: 'error',
        cache: 'no-store',
        headers: {
          Authorization: 'Bearer ' + token, Accept: 'application/json',
          ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        },
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: AbortSignal.timeout(options.timeout || 12_000),
      });
      const body = await response.json();
      return response.ok ? { ok: true, data: body } :
        { ok: false, status: response.status, error: typeof body.detail === 'string' ? body.detail : 'Operator request failed (' + response.status + ')' };
    } catch {
      return { ok: false, error: 'Local operator connection failed or timed out' };
    }
  }
  const post = (path, body) => invoke(path, { method: 'POST', body });
  const checked = (mode, symbol) => {
    if (!['paper', 'live'].includes(mode) || !['IBM', 'JPM', 'KO'].includes(symbol)) {
      return { ok: false, error: 'Invalid TWS workspace or chart symbol' };
    }
    return invoke('/operator/tws/' + mode + '/' + symbol, { timeout: 18_000 });
  };
  window.tradingOperator = Object.freeze({
    isUnlocked: () => Boolean(token),
    unlock: (value) => { token = String(value || '').trim(); },
    lock: () => { token = ''; },
    getStatus: () => invoke('/operator/status'),
    getTwsSettings: () => invoke('/operator/tws-settings'),
    checkTws: checked,
    startPaper: () => post('/operator/start-paper'),
    pause: () => post('/operator/pause'),
    stop: () => post('/operator/stop'),
    cancelAll: () => post('/operator/cancel-all'),
    emergencyStop: () => post('/operator/emergency-stop'),
    getResearchDatasets: () => invoke('/research/datasets'),
    getResearchJobs: () => invoke('/research/jobs'),
    startResearch: (spec) => post('/research/jobs', {
      dataset_id: spec.datasetId,
      generations: spec.generations,
      accounts_per_generation: spec.accountsPerGeneration,
      final_holdout_fraction: spec.finalHoldoutFraction,
      seed: spec.seed,
    }),
    cancelResearch: (jobId) => /^[a-f0-9]{32}$/.test(jobId) ?
      post('/research/jobs/' + jobId + '/cancel') : { ok: false, error: 'Invalid research job identifier' },
  });
})();
