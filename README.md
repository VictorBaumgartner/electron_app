🚀 My Electron + Python Application

This project combines an Electron frontend with a Python backend, bundled into a cross-platform desktop app (macOS .dmg). Python is fully portable using a virtual environment (venv) created and managed from npm scripts.
🛠️ Project Structure

bash

/python                 # Python backend code
/python/venv            # Virtual environment (auto-created)
/python/requirements.txt# Python dependencies list
/scripts/setup_python.sh# Python environment setup script
/main.js                # Electron main process
/package.json           # Build & run scripts
/dist, /build           # Build outputs

✅ Prerequisites

    Node.js (v16+ recommended)

    Python Portable (included in python-portable/, no need for system Python)

    macOS for .dmg creation (for cross-platform builds, adjust Electron Builder config)

🚀 Setup Instructions
1. Clone the repository:

bash

git clone https://github.com/your-repo.git
cd your-repo

2. Install dependencies and setup Python:

bash

npm install

This will automatically:

    Create Python virtual environment (python/venv),

    Install all Python dependencies from requirements.txt,

    Install Playwright browsers in portable mode,

    Install Electron dependencies.

3. (Optional) Manually re-setup Python environment:

bash

npm run setup-python

Equivalent to running:

bash

npm run setup-venv
npm run install-python-deps
npm run install-playwright-browsers

💻 Development Mode

Run both the Electron app and the Python backend in parallel:

bash

npm run dev

📦 Build Production Version (macOS DMG)

To build the Python backend + Electron frontend and generate a .dmg:

bash

npm run build

Includes:

    PyInstaller build of Python backend,

    Electron packaging using electron-builder.

Output: /dist/*.dmg
🧹 Clean Build (Reset everything)

bash

rm -rf node_modules python/venv dist build
npm install

📝 Available npm scripts
Command	Description
npm install	Install Node + Python dependencies (auto setup)
npm run setup-python	Manually setup Python venv + dependencies
npm run dev	Run Electron + Python backend in development mode
npm run build	Build Python (PyInstaller) + Electron app (DMG)
npm run build-python	Build only the Python backend via PyInstaller
npm run build-electron	Build only the Electron app (DMG)
npm run clean (if added)	Remove node_modules, python/venv, dist, build

⚠️ Notes

    Uses portable Python, no system Python required.

    Playwright browsers are stored locally (python-portable/lib/playwright).

    PyInstaller produces the Python executable before Electron builds the final .dmg.
