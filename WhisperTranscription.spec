# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Get the site-packages directory
import site
site_packages = site.getsitepackages()[0]

# Collect all lightning_fabric data files
lightning_datas = collect_data_files('lightning_fabric')
pytorch_lightning_datas = collect_data_files('pytorch_lightning')
pyannote_datas = collect_data_files('pyannote.audio')

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('fonts/*.ttf', 'fonts'),
        *lightning_datas,
        *pytorch_lightning_datas,
        *pyannote_datas,
        (os.path.join(site_packages, 'lightning_fabric/version.info'), 'lightning_fabric'),
    ],
    hiddenimports=[
        'torch',
        'torchaudio',
        'whisper',
        'pyannote.audio',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtMultimedia',
        'librosa',
        'soundfile',
        'matplotlib',
        'numpy',
        'scipy',
        'pydub',
        'requests',
        'transformers',
        'pytorch_lightning',
        'lightning_fabric',
        'src.ui.app',
        'src.ui.fonts',
        'src.transcription.transcriber',
        'src.transcription.recorder',
        'src.diarization.local',
        *collect_submodules('lightning_fabric'),
        *collect_submodules('pytorch_lightning'),
        *collect_submodules('pyannote.audio'),
    ],
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
    [],
    exclude_binaries=True,
    name='WhisperTranscription',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for debugging
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WhisperTranscription'
)

app = BUNDLE(
    coll,
    name='WhisperTranscription.app',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSMicrophoneUsageDescription': 'This app needs access to the microphone to record audio for transcription.',
        'NSHighResolutionCapable': 'True',
    },
)