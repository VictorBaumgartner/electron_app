
# 🚀 Electron + Python Portable Desktop App

This project combines an **Electron frontend** with a **Python backend**, bundled into a **cross-platform desktop application** (macOS `.dmg`).  
Python runs fully portable via a virtual environment (`venv`) created and managed using **npm scripts**.

## 🛠️ Project Structure

```
/python/                 # Python backend code
/python/venv/            # Python virtual environment (auto-created)
/python/requirements.txt # Python dependencies list
/scripts/setup_python.sh # Python environment setup script
/main.js                 # Electron main process
/package.json            # Build & run scripts
/dist/, /build/          # Build outputs
```

## ✅ Prerequisites

- **Node.js** (v16+ recommended)
- **Python Portable** (included in `/python-portable/`, no system Python required)
- **macOS** (for `.dmg` creation; adjust Electron Builder config for cross-platform builds)

## 🚀 Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/VictorBaumgartner/electron_app.git
cd electron_app
```

2. **Install dependencies and set up Python:**

```bash
npm install
```

This will automatically:

- Create the Python virtual environment (`/python/venv`)
- Install Python dependencies (`requirements.txt`)
- Install Playwright browsers (in portable mode)
- Install Electron dependencies

3. **(Optional) Manually set up the Python environment:**

```bash
npm run setup-python
```

Equivalent to running:

```bash
npm run setup-venv
npm run install-python-deps
npm run install-playwright-browsers
```

## 💻 Development Mode

Run both the Electron app and the Python backend in parallel:

```bash
npm run dev
```

## 📦 Build Production Version (macOS `.dmg`)

To build the Python backend and Electron frontend and generate a macOS `.dmg` file:

```bash
npm run build
```

Includes:

- PyInstaller build of the Python backend
- Electron packaging using `electron-builder`

Output located in: `/dist/*.dmg`

## 🧹 Clean Build (Reset Everything)

```bash
rm -rf node_modules python/venv dist build
npm install
```

## 📝 Available npm Scripts

| Command                        | Description                                             |
|-------------------------------|---------------------------------------------------------|
| `npm install`                 | Install Node + Python dependencies (auto setup)          |
| `npm run setup-python`        | Manually set up Python venv + dependencies              |
| `npm run dev`                | Run Electron + Python backend in development mode        |
| `npm run build`              | Build Python (PyInstaller) + Electron app (.dmg)         |
| `npm run build-python`       | Build only the Python backend via PyInstaller            |
| `npm run build-electron`     | Build only the Electron app (.dmg)                        |
| `npm run clean` (if added)   | Remove `node_modules`, `python/venv`, `dist`, `build`     |

## ⚠️ Notes

- Uses **portable Python**; no system Python required.
- **Playwright browsers** are stored locally (`/python-portable/lib/playwright`).
- **PyInstaller** produces the Python executable before Electron builds the final `.dmg`.

---

**GitHub Repository:** [https://github.com/VictorBaumgartner/electron_app](https://github.com/VictorBaumgartner/electron_app)
