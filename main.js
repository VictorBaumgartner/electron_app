const { app, BrowserWindow } = require('electron');
const path = require('path');
const { PythonShell } = require('python-shell');

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  win.loadFile('index.html');

  // Specify the path to script.py within the app bundle
  const scriptPath = path.join(__dirname, 'script.py');

  // Run Python script and send output to renderer
  PythonShell.run(scriptPath, { pythonPath: '/opt/homebrew/bin/python3' }, (err, results) => {
    if (err) {
      console.error('Python error:', err);
      win.webContents.send('python-output', 'Error running Python script');
      return;
    }
    win.webContents.send('python-output', results[0]);
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});