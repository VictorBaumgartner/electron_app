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

  const options = {
    mode: "text",
    pythonPath: pythonExecutable, // Use our determined path
    pythonOptions: ["-u"], // get print results in real-time
    scriptPath: scriptPath, // Tell python-shell where to find the script
  };

  PythonShell.run("script.py", options, (err, results) => {
    if (err) {
      console.error("Python error:", err);
      // Send a more detailed error for debugging
      win.webContents.send(
        "python-output",
        `Error: ${err.message}\nTraceback: ${err.stack}`,
      );
      return;
    }
    console.log("Python results:", results);
    win.webContents.send(
      "python-output",
      results ? results.join("\n") : "Script finished with no output.",
    );
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
