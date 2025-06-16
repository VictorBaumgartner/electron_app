const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const log = require("electron-log");
const fs = require("fs");

let pythonProcess = null;
let mainWindow = null;

function getPythonPath() {
    if (app.isPackaged) {
        return path.join(process.resourcesPath, 'python-portable', 'bin', 'python3');
    }
    return path.join(__dirname, 'python-portable', 'bin', 'python3');
}

function getScriptPath() {
    if (app.isPackaged) {
        return path.join(process.resourcesPath, 'app', 'python', 'main.py');
    }
    return path.join(__dirname, 'python', 'main.py');
}

function getPlaywrightBrowsersPath() {
    if (app.isPackaged) {
        return path.join(process.resourcesPath, 'python-portable', 'lib', 'playwright');
    }
    return path.join(__dirname, 'python-portable', 'lib', 'playwright');
}

function startPythonBackend() {
    const pythonExecutable = getPythonPath();
    const scriptPath = getScriptPath();
    const scriptDir = path.dirname(scriptPath);
    const playwrightBrowsersPath = getPlaywrightBrowsersPath();

    log.info(`Starting Python backend...`);
    log.info(`Python Executable: ${pythonExecutable}`);
    log.info(`Script Path: ${scriptPath}`);
    log.info(`Working Directory: ${scriptDir}`);
    log.info(`Playwright Browsers Path: ${playwrightBrowsersPath}`);

    // Validate paths
    if (!fs.existsSync(pythonExecutable)) {
        const errorMsg = `Python executable not found at: ${pythonExecutable}`;
        log.error(errorMsg);
        if (mainWindow) {
            mainWindow.webContents.send('python-error', errorMsg);
        }
        return false;
    }
    if (!fs.existsSync(scriptPath)) {
        const errorMsg = `Python script not found at: ${scriptPath}`;
        log.error(errorMsg);
        if (mainWindow) {
            mainWindow.webContents.send('python-error', errorMsg);
        }
        return false;
    }
    if (!fs.existsSync(playwrightBrowsersPath) || !fs.existsSync(path.join(playwrightBrowsersPath, 'chromium'))) {
        const errorMsg = `Playwright browsers not found at: ${playwrightBrowsersPath}/chromium`;
        log.warn(errorMsg);
        if (mainWindow) {
            mainWindow.webContents.send('python-error', errorMsg);
        }
    }

    // Set environment variables
    const env = { ...process.env, PLAYWRIGHT_BROWSERS_PATH: playwrightBrowsersPath };

    pythonProcess = spawn(pythonExecutable, [scriptPath], {
        cwd: scriptDir,
        env: env,
        stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on('data', (data) => {
        const message = data.toString();
        log.info(`Python stdout: ${message}`);
        if (mainWindow) {
            mainWindow.webContents.send('python-message', message);
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        const message = data.toString();
        log.error(`Python stderr: ${message}`);
        if (mainWindow) {
            mainWindow.webContents.send('python-error', message);
        }
    });

    pythonProcess.on('close', (code) => {
        const message = `Python process exited with code ${code}`;
        log.info(message);
        pythonProcess = null;
        if (mainWindow) {
            mainWindow.webContents.send('python-error', message);
        }
    });

    pythonProcess.on('error', (err) => {
        const errorMsg = `Failed to start Python process: ${err.message}`;
        log.error(errorMsg);
        if (mainWindow) {
            mainWindow.webContents.send('python-error', errorMsg);
        }
    });

    return true;
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1000,
        height: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, "preload.js"),
        },
    });

    mainWindow.loadFile("index.html");
}

app.whenReady().then(() => {
    const pythonStarted = startPythonBackend();
    if (!pythonStarted) {
        mainWindow.webContents.send('python-error', 'Failed to start Python backend. Check logs for details.');
    }
    createWindow();

    app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

function killPythonProcess() {
    if (pythonProcess) {
        log.info('Terminating Python process...');
        pythonProcess.kill('SIGTERM');
        pythonProcess = null;
    }
}

app.on("window-all-closed", () => {
    killPythonProcess();
    if (process.platform !== "darwin") {
        app.quit();
    }
});

app.on("before-quit", () => {
    killPythonProcess();
});