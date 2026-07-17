# Clinical visual theme for the Scoliosis Detection & Measurement workspace.
#
# Centralizes the color palette and stylesheet for the app's *content*
# areas -- the toolbar and the workspace (Load page / canvas / measurement
# panel). Deliberately NOT applied at the QApplication level: the menu bar
# (File/Edit/View) and all dialogs (Settings, file/color pickers, message
# boxes) are intentionally left with native OS chrome, the same way the
# original wizard-based app behaved. Call apply_clinical_theme(*widgets) on
# only the specific container widgets that should carry the custom look;
# anything that isn't a descendant of one of those (e.g. a QDialog, which
# is always its own top-level window) is untouched and renders natively.

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

/* Labels and plain frames must not paint their own opaque background --
   once a widget has any QSS background-color (even inherited from the
   generic QWidget rule above), Qt fills its own rect with it. Without
   this, every label/frame nested inside a differently-shaded container
   (DropZone, MeasurementPanel, a MetricRow) shows up as a visible
   mismatched box instead of blending into its parent. Named frames below
   (DropZone, MeasurementPanel) still get their own real background --
   an ID selector is more specific and wins over this one. */
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
"""


def apply_clinical_theme(*widgets):
    """Applies the clinical stylesheet to specific container widgets only
    (e.g. the toolbar, the central stacked workspace, the status bar).

    Never pass the QApplication or the QMainWindow itself here -- that
    would cascade into the native menu bar and leak into every dialog the
    window ever parents (Settings, QMessageBox, QFileDialog, QColorDialog),
    which is exactly the look we don't want for those.
    """
    for widget in widgets:
        widget.setStyleSheet(STYLESHEET)
