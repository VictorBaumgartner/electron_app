const { contextBridge } = require('electron');

// Expose a limited API to the renderer process if needed
contextBridge.exposeInMainWorld('electronAPI', {
    // Example: You might want to expose a way to log to main process console
    // log: (message) => console.log(`[Renderer Log]: ${message}`),
}); 