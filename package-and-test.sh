#!/bin/bash

# Exit immediately if a command exits with a non-zero status, except where explicitly handled.
set -e

# --- Configuration ---
PYTHON_EXECUTABLE="/opt/homebrew/bin/python3.10"  # Homebrew Python 3.10
VENV_DIR="python-portable/venv"
PLAYWRIGHT_BROWSERS_PATH="python-portable/lib/playwright"
CHROMIUM_VERSION="chromium-1169"  # Adjust if Playwright uses a different version

echo "🚀 Starting build process for ElectronCrawler..."

# 1. Verify Python executable
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    echo "❌ Error: Python 3.10 not found at $PYTHON_EXECUTABLE. Install it via: brew install python@3.10"
    exit 1
fi
echo "🐍 Using Python: $PYTHON_EXECUTABLE"
$PYTHON_EXECUTABLE --version

# 2. Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    echo "Architecture: Apple Silicon (arm64)"
elif [ "$ARCH" = "x86_64" ]; then
    echo "Architecture: Intel (x86_64)"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# 3. Clean up previous builds
echo "🧹 Cleaning up old directories..."
rm -rf python-portable python/dist
mkdir -p python-portable/lib/playwright
echo "✅ Cleanup complete."

# 4. Create a virtual environment for dependency isolation
echo "🔧 Creating virtual environment..."
$PYTHON_EXECUTABLE -m venv $VENV_DIR
source $VENV_DIR/bin/activate
echo "✅ Virtual environment created."

# 5. Install Python dependencies
echo "🔧 Installing Python dependencies..."
if [ ! -f "python/requirements.txt" ]; then
    echo "⚠️ python/requirements.txt not found. Creating minimal requirements.txt."
    echo "playwright>=1.47.0" > python/requirements.txt
    echo "pyinstaller>=6.14.1" >> python/requirements.txt
fi
pip install --upgrade pip
pip install -r python/requirements.txt
echo "✅ Python dependencies installed."

# 6. Build Python executable with pyinstaller
echo "🔨 Building Python executable with pyinstaller..."
cd python
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -f "dist/main" ]; then
    echo "✅ Python executable already exists, skipping pyinstaller."
else
    pyinstaller --onefile \
        --add-data "$SITE_PACKAGES/playwright:playwright" \
        --add-data "$SITE_PACKAGES/playwright/driver:playwright/driver" \
        main.py
    chmod +x dist/main
    echo "✅ Python executable created at python/dist/main."
fi
cd ..

# 7. Install Playwright browsers
echo "🌐 Checking Playwright browsers..."
CHROMIUM_PATH="$PLAYWRIGHT_BROWSERS_PATH/$CHROMIUM_VERSION/chrome-mac/Chromium.app/Contents/MacOS/Chromium"
if [ -f "$CHROMIUM_PATH" ]; then
    echo "✅ Playwright browsers already installed, skipping."
else
    echo "🌐 Installing Playwright browsers..."
    export PLAYWRIGHT_BROWSERS_PATH=$PLAYWRIGHT_BROWSERS_PATH
    set +e  # Temporarily disable exit-on-error
    PLAYWRIGHT_INSTALL_OUTPUT=$(python -m playwright install --with-deps chromium 2>&1)
    PLAYWRIGHT_INSTALL_STATUS=$?
    set -e  # Re-enable exit-on-error
    echo "Playwright install output:"
    echo "$PLAYWRIGHT_INSTALL_OUTPUT"
    if [ $PLAYWRIGHT_INSTALL_STATUS -ne 0 ]; then
        echo "❌ Failed to install Playwright browsers! Exit code: $PLAYWRIGHT_INSTALL_STATUS"
        exit 1
    fi
    if [ ! -f "$CHROMIUM_PATH" ]; then
        echo "❌ Playwright browsers not found after installation at: $CHROMIUM_PATH"
        exit 1
    fi
    echo "✅ Playwright browsers installed."
fi

# 8. Deactivate virtual environment
deactivate
echo "✅ Virtual environment deactivated."

# 9. Run electron-builder to package the application
echo "📦 Packaging the Electron app..."
if [ -d "dist/ElectronCrawler.app" ]; then
    echo "✅ Electron app already packaged, skipping."
else
    echo "🧹 Cleaning up old 'dist' directory..."
    rm -rf dist
    npx electron-builder
    echo "✅ Electron app packaged."
fi

# 9. Remove macOS quarantine attributes from the packaged app
if [ -f "dist/ElectronCrawler.app/Contents/Info.plist" ]; then
    echo "🔓 Removing macOS quarantine attributes..."
    xattr -cr ./dist/electroncrawler.app
    echo "✅ Quarantine attributes removed."
else
    echo "⚠️ No ElectronCrawler.app found, skipping xattr."
fi

echo "🎉 Build process completed successfully!"
echo "📦 Your application can be found in the 'dist' directory."

# Optional: Open the output directory
open dist