const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  // Use absolute path to ensure proper file loading
  const indexPath = path.join(__dirname, 'renderer', 'index.html');
  console.log('Loading file from:', indexPath);
  
  win.loadFile(indexPath).catch(err => {
    console.error('Failed to load file:', err);
    // Fallback: try loading from URL
    win.loadURL(`file://${indexPath}`);
  });
  
  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    win.webContents.openDevTools();
  }
}

app.whenReady().then(() => {
  console.log('Electron app is ready');
  console.log('Current directory:', __dirname);
  console.log('Renderer path exists:', require('fs').existsSync(path.join(__dirname, 'renderer', 'index.html')));
  
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
