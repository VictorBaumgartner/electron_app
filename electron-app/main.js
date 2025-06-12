// main.js
const { app, BrowserWindow } = require("electron");
const path = require("path");
const { spawn } = require("child_process"); // Import child_process for spawning Python

// Determine the correct paths for packaged vs. development environments
const isDev = !app.isPackaged;

// Path to the Python executable (ensure this points to a Python installation with FastAPI/Uvicorn)
const pythonExecutable = isDev
  ? "python" // Assumes 'python' is in PATH in development
  : path.join(process.resourcesPath, "python-portable", "python.exe"); // Adjust for packaged Python

// Path to the FastAPI main.py script
const fastapiScriptPath = isDev
  ? path.join(__dirname, "main.py") // In dev, main.py is in the root
  : path.join(process.resourcesPath, "main.py"); // In packaged app, it's in Resources

let pythonProcess = null; // Global variable to hold the Python process

function startPythonBackend() {
  if (pythonProcess) {
    console.log("Python backend is already running.");
    return;
  }

  console.log("Starting Python backend...");
  console.log(`Python Executable: ${pythonExecutable}`);
  console.log(`FastAPI Script: ${fastapiScriptPath}`);

  pythonProcess = spawn(pythonExecutable, [fastapiScriptPath]);

  pythonProcess.stdout.on("data", (data) => {
    console.log(`Python stdout: ${data.toString('utf8')}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`Python stderr: ${data.toString('utf8')}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`Python process exited with code ${code}`);
    pythonProcess = null; // Clear the reference
  });

  pythonProcess.on("error", (err) => {
    console.error("Failed to start Python process:", err);
    pythonProcess = null; // Clear the reference on error
  });
}

function killPythonBackend() {
  if (pythonProcess) {
    console.log("Killing Python backend...");
    pythonProcess.kill(); // Send a SIGTERM signal to the process
    pythonProcess = null; // Clear the reference immediately
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
      nodeIntegration: false, // Node.js integration disabled for renderer processes for security
      contextIsolation: true, // Context isolation enabled for security
      preload: path.join(__dirname, 'preload.js') // Preload script for secure IPC
    },
    icon: path.join(__dirname, 'build', 'icon.png')
  });

  win.loadFile("index.html");

  // Open the DevTools.
  // win.webContents.openDevTools();

  startPythonBackend(); // Start the FastAPI backend when the window is created
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    killPythonBackend(); // Kill Python backend when all windows are closed
    app.quit();
  }
});

app.on("before-quit", () => {
  killPythonBackend(); // Ensure Python backend is killed before app quits
});
