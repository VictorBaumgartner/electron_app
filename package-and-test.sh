#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_VERSION="3.9.18"
RELEASE_DATE="20231002"
PYTHON_TARBALL="python-mac.tar.gz"

# --- Architecture Detection for macOS ---
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    echo "Architecture: Apple Silicon (aarch64)"
    PYTHON_ARCH="aarch64-apple-darwin"
elif [ "$ARCH" = "x86_64" ]; then
    echo "Architecture: Intel (x86_64)"
    PYTHON_ARCH="x86_64-apple-darwin"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

PYTHON_DOWNLOAD_URL="https://github.com/indygreg/python-build-standalone/releases/download/${RELEASE_DATE}/cpython-${PYTHON_VERSION}+${RELEASE_DATE}-${PYTHON_ARCH}-install_only.tar.gz"

echo "🚀 Starting build process for ElectronCrawler..."

# 1. Check if python-portable exists
if [ -d "python-portable" ] && [ -f "python-portable/bin/python3" ]; then
    echo "✅ python-portable already exists, skipping download and extraction."
else
    # 2. Clean up previous builds
    echo "🧹 Cleaning up old 'python-portable' directory..."
    rm -rf python-portable
    echo "✅ Cleanup complete."

    # 3. Download portable Python
    echo "🐍 Downloading Python portable for your architecture..."
    curl -L "$PYTHON_DOWNLOAD_URL" -o "$PYTHON_TARBALL"
    echo "✅ Download complete."

    # 4. Extract Python
    echo "📦 Extracting Python..."
    mkdir -p python-portable
    tar -xzf "$PYTHON_TARBALL" --strip-components=1 -C python-portable
    echo "✅ Extraction complete."

    # 5. Clean up the downloaded tarball
    rm "$PYTHON_TARBALL"
fi

# 6. Install Python dependencies from requirements.txt
echo "🔧 Checking Python dependencies..."
if [ -f "python-portable/requirements-installed.txt" ]; then
    echo "✅ Python dependencies already installed, skipping."
else
    if [ ! -f "python/requirements.txt" ]; then
        echo "❌ Error: requirements.txt not found! Please create it."
        exit 1
    fi
    echo "🔧 Installing Python dependencies into portable environment..."
    python-portable/bin/python3 -m pip install --upgrade pip
    python-portable/bin/python3 -m pip install -r python/requirements.txt
    touch python-portable/requirements-installed.txt
    echo "✅ Python dependencies installed."
fi

# 7. Install Playwright browsers
echo "🌐 Checking Playwright browsers..."
if [ -d "python-portable/lib/playwright" ] && [ -d "python-portable/lib/playwright/chromium" ]; then
    echo "✅ Playwright browsers already installed, skipping."
else
    echo "🌐 Installing Playwright browsers..."
    python-portable/bin/python3 -m playwright install --with-deps
    echo "✅ Playwright browsers installed."
fi

# 8. Run electron-builder to package the application
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
    xattr -cr ./dist/ElectronCrawler.app
    echo "✅ Quarantine attributes removed."
else
    echo "⚠️ No ElectronCrawler.app found, skipping xattr."
fi

echo "🎉 Build process completed successfully!"
echo "📦 Your application can be found in the 'dist' directory."

# Optional: Open the output directory
open dist