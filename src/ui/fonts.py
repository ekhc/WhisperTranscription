# src/ui/fonts.py

from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtWidgets import QApplication

def load_font(resource_path: str) -> str:
    """
    Load a font from the given resource path and return the first family name.
    If loading fails, returns None.
    """
    font_id = QFontDatabase.addApplicationFont(resource_path)
    if font_id != -1:
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return families[0]
    return None

def setup_application_font(primary_resource=":/fonts/NotoSansKR.ttf"):
    """
    Loads the primary application font from the specified resource path.
    If the font cannot be loaded, it falls back to a default system font ("Arial").
    Sets the resulting font as the default for the entire application.
    """
    primary_family = load_font(primary_resource)
    if not primary_family:
        primary_family = "Arial"  # Fallback to a system font if loading fails
    
    font = QFont(primary_family)
    QApplication.instance().setFont(font)
