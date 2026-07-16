const { contextBridge, ipcRenderer } = require('electron');

function invoke(operation) {
  return ipcRenderer.invoke('operator:invoke', operation);
}

contextBridge.exposeInMainWorld('tradingOperator', Object.freeze({
  getStatus: () => invoke('status'),
  getOrders: () => invoke('orders'),
  getPositions: () => invoke('positions'),
  startPaper: () => invoke('startPaper'),
  pause: () => invoke('pause'),
  stop: () => invoke('stop'),
  cancelAll: () => invoke('cancelAll'),
  emergencyStop: () => invoke('emergencyStop')
}));
