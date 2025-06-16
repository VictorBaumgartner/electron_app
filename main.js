const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const log = require("electron-log");

let pythonProcess = null;
let mainWindow = null;

function getPythonPath() {
    // If we are packaged, the python executable is in the resources path
    if (app.isPackaged) {
        return path.join(process.resourcesPath, 'python-portable', 'bin', 'python3');
    }
    // In development, use the portable Python in the project directory
    return path.join(__dirname, 'python-portable', 'bin', 'python3');
}

function getScriptPath() {
    // If we are packaged, the script is in 'app/python' inside the resources path
    if (app.isPackaged) {
        return path.join(process.resourcesPath, 'app', 'python', 'main.py');
    }
    // In development, it's in the 'python' folder of your project
    return path.join(__dirname, 'python', 'main.py');
}

function getPlaywrightBrowsersPath() {
    // Set the path where Playwright stores its browser binaries
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

    // Set PLAYWRIGHT_BROWSERS_PATH environment variable for Playwright
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
        log.info(`Python process exited with code ${code}`);
        pythonProcess = null;
    });

    pythonProcess.on('error', (err) => {
        log.error('Failed to start Python process:', err.message);
    });
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
    // mainWindow.webContents.openDevTools(); // Uncomment for debugging
}

app.whenReady().then(() => {
    startPythonBackend();
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