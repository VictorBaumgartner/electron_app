#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting packaging process with electron-builder..."

# Run electron-builder. It will create the DMG in the 'dist' folder.
# The command comes from the "scripts" section of package.json.
npm run package

echo "✅ Process completed successfully!"
echo "📦 Your DMG can be found in the 'dist' directory."

# Optional: Open the output directory
open dist