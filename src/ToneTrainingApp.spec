# -*- mode: python ; coding: utf-8 -*-
import sys
import os

# 1. Detect the Operating System
is_mac = sys.platform == 'darwin'

# 2. Define Resources to Include
# Maps (source_folder, destination_folder_inside_app)
# This ensures your sound files and models are available after compilation
added_files = []

# Check if model_training exists before adding to prevent "Unable to find" error
if os.path.exists('model_training'):
    added_files.append(('model_training', 'model_training'))

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=added_files, #
    hiddenimports=[
        'sounddevice', 
        'soundfile', 
        'pandas', 
        'seaborn', 
        'sklearn', 
        'scipy', 
        'scipy._cyutility'
    ], # 
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ToneTrainingApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # Set to True if you need a terminal for debugging [cite: 5]
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 3. macOS Bundle Configuration
# This section only runs when compiling on a Mac and handles microphone permissions
if is_mac:
    app = BUNDLE(
        exe,
        name='ToneTrainingApp.app',
        icon=None, # Path to a .icns file if available
        bundle_identifier='com.oliver.tonetraining',
        info_plist={
            'NSMicrophoneUsageDescription': 'This app requires microphone access to record and analyze your voice during training.',
            'com.apple.security.device.audio-input': True, # Required for Hardened Runtime
        },
    )