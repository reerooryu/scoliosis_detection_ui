# QWizard implementation coordinating page transitions and clinical state

from PySide6.QtWidgets import QWizard
from PySide6.QtGui import QIcon
from config import APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT
from modules.pages import ImageLoaderPage, InferencePage, AdjustmentPage, ExportPage
from modules.model_mock import ScoliosisModelEngine

class ScoliosisWizard(QWizard):
    """The central workflow coordinator subclassing QWizard."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Central state variables
        self.image_path = ""
        
        # Central clinical data-state engine
        self.model_engine = ScoliosisModelEngine()

        # Define Wizard options
        self.setWizardStyle(QWizard.ClassicStyle)
        self.setOption(QWizard.HaveHelpButton, False)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.setOption(QWizard.NoBackButtonOnLastPage, False)

        # Instantiate modular pages
        self.page_image_loader = ImageLoaderPage(self)
        self.page_inference = InferencePage(self)
        self.page_adjustment = AdjustmentPage(self)
        self.page_export = ExportPage(self)

        # Map steps to consecutive ID indices
        self.addPage(self.page_image_loader)
        self.addPage(self.page_inference)
        self.addPage(self.page_adjustment)
        self.addPage(self.page_export)
        
        # Set background layout style parameters
        self.setStyleSheet("""
            QWizard {
                background-color: #f5f6f8;
            }
            QLabel {
                font-family: 'Segoe UI', Arial;
            }
            QPushButton {
                font-family: 'Segoe UI', Arial;
                padding: 6px 14px;
                min-width: 75px;
            }
        """)
