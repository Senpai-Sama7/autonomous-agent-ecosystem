#!/bin/bash
# ASTRO - Build Debian Package (.deb)
# Run this script to create astro_2.0.0_amd64.deb

set -e

echo "============================================"
echo "  ASTRO v2.0 - Building Debian Package"
echo "============================================"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Clean previous build
rm -rf debian/usr/share/astro/*
mkdir -p debian/usr/share/astro

# Copy application files
echo "Copying application files..."
cp -r src debian/usr/share/astro/
cp -r config debian/usr/share/astro/
cp requirements.txt debian/usr/share/astro/
cp README.md debian/usr/share/astro/
cp LICENSE debian/usr/share/astro/

# Create workspace directory
mkdir -p debian/usr/share/astro/workspace/knowledge

# Set permissions
echo "Setting permissions..."
chmod 755 debian/DEBIAN/postinst
chmod 755 debian/DEBIAN/prerm
chmod 755 debian/usr/bin/astro

# Build the package
echo "Building .deb package..."
dpkg-deb --build debian astro_2.0.0_amd64.deb

echo ""
echo "============================================"
echo "  Build Complete!"
echo "============================================"
echo "  Package: astro_2.0.0_amd64.deb"
echo ""
echo "  To install:"
echo "    sudo dpkg -i astro_2.0.0_amd64.deb"
echo "    sudo apt-get install -f  # Fix dependencies"
echo ""
echo "  To run:"
echo "    astro (from terminal)"
echo "    or find 'ASTRO' in Applications menu"
echo "============================================"
