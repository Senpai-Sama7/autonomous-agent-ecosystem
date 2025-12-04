#!/bin/bash
# ASTRO v2.0 - Build Windows Executable
# Requires: PyInstaller, Wine (for cross-compilation) or run on Windows

set -e

echo "============================================"
echo "  ASTRO v2.0 - Building Windows Executable"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous builds
rm -rf build dist

# Build the executable
echo "Building executable..."
pyinstaller astro.spec --clean

echo ""
echo "============================================"
echo "  Build Complete!"
echo "============================================"
echo "  Executable: dist/ASTRO.exe (Windows)"
echo "             dist/ASTRO.app (macOS)"
echo ""
echo "  Note: For Windows builds, run this on Windows"
echo "        or use a cross-compilation tool."
echo "============================================"
