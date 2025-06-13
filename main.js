const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

let pythonProcess = null;
let mainWindow = null;

function getPythonPath() {
    // If we are packaged, the python executable is in the resources path
    if (app.isPackaged) {
        // The path will be /Applications/YourApp.app/Contents/Resources/python-portable/bin/python3
        return path.join(process.resourcesPath, 'python-portable', 'bin', 'python3');
    }
    // In development, use the system's python3
    // Make sure 'python3' is in your system's PATH and has the required packages installed for development.
    // Or point to a local venv: path.join(__dirname, 'venv', 'bin', 'python')
    // return 'python3';
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

function startPythonBackend() {
    const pythonExecutable = getPythonPath();
    const scriptPath = getScriptPath();
    const scriptDir = path.dirname(scriptPath);

    console.log(`Starting Python backend...`);
    console.log(`Executable: ${pythonExecutable}`);
    console.log(`Script: ${scriptPath}`);
    console.log(`CWD: ${scriptDir}`);

    pythonProcess = spawn(pythonExecutable, [scriptPath], {
        cwd: scriptDir, // Set the working directory to the script's location
        stdio: ['pipe', 'pipe', 'pipe'] // Use pipes for stdin, stdout, stderr
    });

    pythonProcess.stdout.on('data', (data) => {
        const message = data.toString();
        console.log(`Python stdout: ${message}`);
        // You can forward messages to the renderer process if needed
        if (mainWindow) {
            mainWindow.webContents.send('python-message', message);
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        const message = data.toString();
        console.error(`Python stderr: ${message}`);
        if (mainWindow) {
            mainWindow.webContents.send('python-error', message);
        }
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        pythonProcess = null;
    });

    pythonProcess.on('error', (err) => {
        console.error('Failed to start Python process:', err);
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

// Gracefully kill the python process when the app closes
function killPythonProcess() {
    if (pythonProcess) {
        console.log('Terminating Python process...');
        pythonProcess.kill('SIGTERM'); // Send termination signal
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