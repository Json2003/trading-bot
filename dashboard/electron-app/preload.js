const { contextBridge, ipcRenderer } = require('electron');

function invoke(operation, payload = null) {
  return ipcRenderer.invoke('operator:invoke', operation, payload);
}

contextBridge.exposeInMainWorld('tradingOperator', Object.freeze({
  getStatus: () => invoke('status'),
  getOrders: () => invoke('orders'),
  getPositions: () => invoke('positions'),
  startPaper: () => invoke('startPaper'),
  pause: () => invoke('pause'),
  stop: () => invoke('stop'),
  cancelAll: () => invoke('cancelAll'),
  emergencyStop: () => invoke('emergencyStop'),
  getResearchDatasets: () => invoke('researchDatasets'),
  getResearchJobs: () => invoke('researchJobs'),
  startResearch: (spec) => invoke('startResearch', spec),
  cancelResearch: (jobId) => invoke('cancelResearch', { jobId })
}));
