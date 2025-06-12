// main.js
const { app, BrowserWindow } = require("electron");
const path = require("path");
const { spawn } = require("child_process"); // Use spawn from child_process

let pythonProcess = null; // To hold the Python process

// Determine the correct paths for packaged vs. development environments
const isDev = !app.isPackaged;

// Path to the Python executable
const pythonExecutable = isDev
  ? path.join(__dirname, "python-portable", "bin", "python3")
  : path.join(process.resourcesPath, "python-portable", "bin", "python3");

// Path to the script's directory
const scriptDir = isDev
  ? __dirname // In dev, script.py is in the root
  : process.resourcesPath; // In packaged app, it's in Resources

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  win.loadFile("index.html");

  // Wait for the window to be ready before running the Python script
  win.webContents.on("did-finish-load", () => {
    // Launch the FastAPI server using uvicorn
    const args = [
      "-m",
      "uvicorn",
      "script:app",
      "--host",
      "127.0.0.1",
      "--port",
      "8000",
    ];
    pythonProcess = spawn(pythonExecutable, args, {
      cwd: scriptDir, // Run the command from the script's directory
      stdio: ["pipe", "pipe", "pipe"], // Pipe stdout, stderr
    });

    console.log("Starting Python FastAPI server...");

    // Listen for data from the Python server's stdout
    pythonProcess.stdout.on("data", (data) => {
      const message = data.toString();
      console.log("Python stdout:", message);
      win.webContents.send("python-output", message);
    });

    // Listen for data from the Python server's stderr
    pythonProcess.stderr.on("data", (data) => {
      const message = data.toString();
      console.error("Python stderr:", message);
      win.webContents.send("python-output", `ERROR: ${message}`);
    });

    // Listen for process exit
    pythonProcess.on("close", (code) => {
      const message = `Python process exited with code ${code}`;
      console.log(message);
      win.webContents.send("python-output", message);
    });

    // Listen for process errors
    pythonProcess.on("error", (err) => {
      console.error("Failed to start Python process:", err);
      win.webContents.send(
        "python-output",
        `Failed to start Python process: ${err.message}`,
      );
    });
  });
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

// Make sure to kill the python process when the app quits
app.on("will-quit", () => {
  if (pythonProcess) {
    console.log("Terminating Python process...");
    pythonProcess.kill();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
