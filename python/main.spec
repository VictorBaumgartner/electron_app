# python/main.spec

# -*- mode: python ; coding: utf-8 -*-
import os
import site
from pathlib import Path

# Dynamically find the site-packages directory of the virtual environment
# This makes the script robust and portable.
try:
    # This works when run from a virtual environment
    site_packages_path = site.getsitepackages()[0]
except IndexError:
    # Fallback for other scenarios, though the venv should be primary
    from distutils.sysconfig import get_python_lib
    site_packages_path = get_python_lib()

print(f"--- Using site-packages from: {site_packages_path} ---")

# This list directly translates the `--add-data` flags from your shell script.
# It tells PyInstaller to bundle Playwright's necessary Python components.
playwright_datas = [
    (os.path.join(site_packages_path, 'playwright'), 'playwright'),
    (os.path.join(site_packages_path, 'playwright', 'driver'), os.path.join('playwright', 'driver'))
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=playwright_datas,  # <-- This is where we add the Playwright data
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    onefile=True,
    name='main'
)

exe = EXE(
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # Important: No console window on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)