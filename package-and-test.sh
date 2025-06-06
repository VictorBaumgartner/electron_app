#!/bin/bash

# Exit on any error
set -e

# If the dist directory exists, delete it
if [ -d "dist" ]; then
    rm -rf dist
    echo "🔍 Deleted dist directory"
fi

# Download python portable
curl -L "https://github.com/astral-sh/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-aarch64-apple-darwin-install_only.tar.gz" -o python-mac.tar.gz
echo "🔍 Downloaded python-mac.tar.gz"

# Extract the tar.gz file into python-portable
mkdir -p python-portable
tar -xzf python-mac.tar.gz --strip-components=1 -C python-portable
echo "🔍 Extracted python-mac.tar.gz into python-portable"

# Delete the tar.gz file
rm python-mac.tar.gz
echo "🔍 Deleted python-mac.tar.gz"

# Install the dependencies using the python-portable
python-portable/bin/python3 -m pip install -r requirements.txt
echo "🔍 Installed the dependencies using the python-portable"

echo "🚀 Starting packaging process with electron-builder..."

# Run electron-builder. It will create the DMG in the 'dist' folder.
# The command comes from the "scripts" section of package.json.
npm run package

echo "✅ Process completed successfully!"
echo "📦 Your DMG can be found in the 'dist' directory."

# Optional: Open the output directory
open dist