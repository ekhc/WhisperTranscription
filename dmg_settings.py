
from pathlib import Path
import os.path

application = defines.get('app', 'dist/WhisperTranscription.app')
appname = os.path.basename(application)
volname = "WhisperTranscription Installer"
format = defines.get('format', 'UDBZ')
size = defines.get('size', '1G')
files = [application]
symlinks = {'Applications': '/Applications'}
