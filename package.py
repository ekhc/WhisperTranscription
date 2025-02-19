import os
import subprocess
from pathlib import Path

def build_app():
    # Clean previous builds
    os.system('rm -rf build dist')
    
    # Install required packages
    os.system('poetry install')
    
    print("Building app with PyInstaller...")
    result = os.system('poetry run pyinstaller WhisperTranscription.spec')
    if result != 0:
        print("Error: PyInstaller failed to build the app")
        return
    
    # Check if the app was built successfully
    app_path = 'dist/WhisperTranscription.app'
    if not os.path.exists(app_path):
        print(f"Error: {app_path} was not created successfully")
        return
    
    print("Creating DMG settings...")
    # Create DMG settings file
    dmg_settings = '''
from pathlib import Path
import os.path

application = defines.get('app', 'dist/WhisperTranscription.app')
appname = os.path.basename(application)
volname = "WhisperTranscription Installer"
format = defines.get('format', 'UDBZ')
size = defines.get('size', '1G')
files = [application]
symlinks = {'Applications': '/Applications'}
'''
    
    with open('dmg_settings.py', 'w') as f:
        f.write(dmg_settings)
    
    print("Building DMG...")
    # Build the DMG using poetry
    result = os.system('poetry run dmgbuild -s dmg_settings.py "WhisperTranscription" dist/WhisperTranscription.dmg')
    if result != 0:
        print("Error: Failed to create DMG")
        return
    
    print("Build completed successfully!")
    print("DMG file created at: dist/WhisperTranscription.dmg")

if __name__ == '__main__':
    build_app()