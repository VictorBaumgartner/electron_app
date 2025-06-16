#!/bin/bash

set -e

# --- Configuration ---
PYTHON_VERSION="3.9.18"
RELEASE_DATE="20231002"

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
PYTHON_TARBALL="python-mac.tar.gz"

echo "🚀 Starting build process for ElectronCrawler..."

# 1. Clean up previous builds
echo "🧹 Cleaning up old 'dist' and 'python-portable' directories..."
rm -rf dist
rm -rf python-portable
echo "✅ Cleanup complete."

# 2. Download portable Python
echo "🐍 Downloading Python portable for your architecture..."
curl -L "$PYTHON_DOWNLOAD_URL" -o "$PYTHON_TARBALL"
echo "✅ Download complete."

# 3. Extract Python
echo "📦 Extracting Python..."
mkdir -p python-portable
tar -xzf "$PYTHON_TARBALL" --strip-components=1 -C python-portable
echo "✅ Extraction complete."

# 4. Clean up the downloaded tarball
rm "$PYTHON_TARBALL"

# 5. Install Python dependencies from requirements.txt
echo "🔧 Installing Python dependencies into portable environment..."
if [ ! -f "python/requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found! Please create it."
    exit 1
fi
python-portable/bin/python3 -m pip install --upgrade pip
python-portable/bin/python3 -m pip install -r python/requirements.txt
echo "✅ Python dependencies installed."

# 6. Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
python-portable/bin/python3 -m playwright install --with-deps
echo "✅ Playwright browsers installed."

# 7. Run electron-builder to package the application
echo "📦 Packaging the Electron app..."
npm run package

# 8. Remove macOS quarantine attributes from the packaged app
echo "🔓 Removing macOS quarantine attributes..."
xattr -cr ./dist/ElectronCrawler.app
echo "✅ Quarantine attributes removed."

echo "🎉 Build process completed successfully!"
echo "📦 Your application can be found in the 'dist' directory."

# Optional: Open the output directory
open dist