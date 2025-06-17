// main.js - Using CommonJS require() syntax

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const log = require('electron-log');
const fs = require('fs');

let pythonProcess = null;
let mainWindow = null;

function startPythonBackend() {
    const isProd = app.isPackaged;
    let executablePath;
    let playwrightBrowsersPath;
    let spawnArgs = [];

    // --- Determine Paths and Arguments based on Environment ---
    if (isProd) {
        // --- PRODUCTION ---
        // In the packaged app, we run the compiled single-file executable.
        executablePath = path.join(process.resourcesPath, 'python-bin', 'main');

        // The Playwright browsers were bundled into this specific folder.
        playwrightBrowsersPath = path.join(process.resourcesPath, 'playwright-browsers');

        // No extra arguments are needed for the executable.
        spawnArgs = [];

    } else {
        // --- DEVELOPMENT ---
        // In development, we use the Python interpreter from our virtual environment.
        executablePath = path.join(app.getAppPath(), 'python', 'venv', 'bin', 'python');

        // Playwright browsers were installed here by our setup script.
        playwrightBrowsersPath = path.join(app.getAppPath(), 'python-portable', 'lib', 'playwright');

        // We need to tell the interpreter which script to run.
        spawnArgs = [path.join(app.getAppPath(), 'python', 'main.py')];
    }

    console.log("TEST")
    log.info(`--- Python Backend Setup (${isProd ? 'PRODUCTION' : 'DEVELOPMENT'}) ---`);
    log.info(`Executable Path: ${executablePath}`);
    log.info(`Spawn Arguments: [${spawnArgs.join(', ')}]`);
    log.info(`Playwright Browsers Path: ${playwrightBrowsersPath}`);

    // --- Validate Paths ---
    if (!fs.existsSync(executablePath)) {
        const errorMsg = `Python backend executable not found at: ${executablePath}`;
        log.error(errorMsg);
        if (mainWindow) {
            mainWindow.webContents.send('python-error', errorMsg);
        }
        return false; // Critical error, cannot continue.
    }
    if (!fs.existsSync(playwrightBrowsersPath)) {
        const errorMsg = `Playwright browsers directory not found at: ${playwrightBrowsersPath}`;
        log.warn(errorMsg); // Log as a warning, as the Python script might handle it.
    }

    // --- Set Environment Variable and Spawn Process ---
    const env = { ...process.env, PLAYWRIGHT_BROWSERS_PATH: playwrightBrowsersPath };

    pythonProcess = spawn(executablePath, spawnArgs, {
        env: env,
        stdio: ['pipe', 'pipe', 'pipe']
    });

    // --- Process Event Listeners ---
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
            preload: path.join(__dirname, "preload.js"), // In CJS, __dirname is available and reliable
        },
    });

    mainWindow.loadFile("index.html");
}

app.whenReady().then(() => {
    createWindow();
    
    const pythonStarted = startPythonBackend();
    if (!pythonStarted) {
        mainWindow.webContents.send('python-error', 'Fatal: Could not start Python backend. Check main process logs.');
    }

    app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

function killPythonProcess() {
    if (pythonProcess) {
        log.info('Terminating Python process...');
        pythonProcess.kill('SIGKILL');
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