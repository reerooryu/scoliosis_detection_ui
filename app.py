# Entry point for the desktop app.

import sys
import os

# Let `config` and `modules` import no matter where the app is launched from.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PySide6.QtWidgets import QApplication
from modules.main_window import MainWindow

def main():
    app = QApplication(sys.argv)

    # MainWindow themes only its toolbar and workspace; menus and dialogs stay native.
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
