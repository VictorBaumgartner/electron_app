#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting packaging process..."

# Run npm package command
echo "📦 Running npm package..."
npm run package

# Create DMG file
echo "📦 Creating DMG file..."
create-dmg --no-sign 'dist/HelloWorld-darwin-x64/HelloWorld.app' dist

# Open the app for testing
echo "🔍 Opening app for testing..."
open dist/HelloWorld-darwin-x64/HelloWorld.app

echo "✅ Process completed successfully!" 