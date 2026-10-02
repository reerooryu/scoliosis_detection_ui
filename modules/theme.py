# Dark clinical theme: the color palette and stylesheet for the toolbar and
# workspace only. Menus and dialogs keep the native OS look, so the theme is
# applied to specific widgets (apply_clinical_theme), never app-wide.

# ---- Palette ---------------------------------------------------------------
BG_APP = "#1c2024"           # Main window / canvas background
BG_PANEL = "#24292e"         # Toolbar / side panel background
BG_PANEL_RAISED = "#2b3136"  # Cards, inputs, hover states
BORDER = "#3a4048"           # Subtle dividers
TEXT_PRIMARY = "#e6e9eb"     # Primary readable text
TEXT_MUTED = "#9aa4ab"       # Secondary labels
ACCENT = "#4f83a3"           # Desaturated steel blue - primary actions
ACCENT_HOVER = "#5d93b3"
ACCENT_TEXT = "#ffffff"
DANGER = "#b3564a"           # Muted red - destructive/reject states
SUCCESS = "#4f9d78"          # Muted green - confirmations

STYLESHEET = f"""
QWidget {{
    background-color: {BG_APP};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', Arial;
    font-size: 11.5pt;
}}

/* Labels and frames stay transparent so they blend into their parent
   instead of showing as mismatched boxes. DropZone and MeasurementPanel
   set their own background below (the more specific selector wins). */
QLabel, QFrame {{
    background-color: transparent;
}}

QToolBar {{
    background-color: {BG_PANEL};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 4px 6px;
    spacing: 4px;
}}
QToolButton {{
    color: {TEXT_PRIMARY};
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px 10px;
}}
QToolButton:hover {{
    background-color: {BG_PANEL_RAISED};
    border-color: {BORDER};
}}
QToolButton:checked {{
    background-color: {ACCENT};
    color: {ACCENT_TEXT};
}}
QToolButton:disabled {{
    color: {TEXT_MUTED};
}}

QFrame#DropZone {{
    border: 2px dashed {BORDER};
    border-radius: 8px;
    background-color: {BG_PANEL};
}}
QFrame#DropZone[dragActive="true"] {{
    border-color: {ACCENT};
    background-color: {BG_PANEL_RAISED};
}}

QFrame#MeasurementPanel {{
    background-color: {BG_PANEL};
    border-left: 1px solid {BORDER};
}}

QPushButton {{
    background-color: {BG_PANEL_RAISED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 7px 16px;
}}
QPushButton:hover {{
    background-color: {BORDER};
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
}}
QPushButton#PrimaryButton {{
    background-color: {ACCENT};
    color: {ACCENT_TEXT};
    border: none;
    font-weight: 600;
}}
QPushButton#PrimaryButton:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton#PrimaryButton:disabled {{
    background-color: {BG_PANEL_RAISED};
    color: {TEXT_MUTED};
}}

QLabel#MetricValue {{
    color: {TEXT_PRIMARY};
    font-size: 14pt;
    font-weight: 600;
}}
QLabel#MetricLabel {{
    color: {TEXT_MUTED};
    font-size: 10pt;
}}

QSplitter::handle {{
    background-color: {BORDER};
}}

QStatusBar {{
    background-color: {BG_PANEL};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}

QProgressBar {{
    background-color: {BG_PANEL_RAISED};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 2px;
}}
"""


def apply_clinical_theme(*widgets):
    """Apply the stylesheet to specific widgets only (toolbar, workspace
    stack, status bar). Never pass the QApplication or the QMainWindow: the
    theme would spread to the menu bar and every dialog."""
    for widget in widgets:
        widget.setStyleSheet(STYLESHEET)
