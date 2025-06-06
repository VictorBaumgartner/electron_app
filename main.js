// main.js
const { app, BrowserWindow } = require("electron");
const path = require("path");
const { PythonShell } = require("python-shell");

// Determine the correct paths for packaged vs. development environments
const isDev = !app.isPackaged;

// Path to the Python executable
const pythonExecutable = isDev
  ? path.join(__dirname, "python-portable", "bin", "python3")
  : path.join(process.resourcesPath, "python-portable", "bin", "python3");

// Path to the Python script
const scriptPath = isDev
  ? path.join(__dirname) // In dev, script.py is in the root
  : path.join(process.resourcesPath); // In packaged app, it's in Resources

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
    const options = {
      mode: "text",
      pythonPath: pythonExecutable, // Use our determined path
      pythonOptions: ["-u"], // get print results in real-time
      scriptPath: scriptPath, // Tell python-shell where to find the script
    };

    const pyShell = new PythonShell("script.py", options);

    // Listen for messages from the Python script
    pyShell.on("message", (message) => {
      // Received a message from the Python script
      console.log("Python output:", message);
      win.webContents.send("python-output", message);
    });

    // Listen for any errors
    pyShell.on("error", (err) => {
      console.error("Python error:", err);
      win.webContents.send(
        "python-output",
        `Error: ${err.message}\nTraceback: ${err.stack}`,
      );
    });

    // End the stream when the script finishes
    pyShell.end((err, code, signal) => {
      if (err) {
        console.error("Python error during script execution:", err);
        return;
      }
      console.log(
        `Python script finished with code ${code} and signal ${signal}`,
      );
      win.webContents.send("python-output", "Script finished.");
    });
  });
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
