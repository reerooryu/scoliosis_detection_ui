# Main desktop application launcher for Scoliosis Detection UI

import sys
import os

# Insert the current directory to sys.path to guarantee modules load correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PySide6.QtWidgets import QApplication
from modules.wizard import ScoliosisWizard

def main():
    # Initialize the desktop application context (Qt6 handles High-DPI scaling automatically)
    app = QApplication(sys.argv)
    
    # Instantiate and display the step-by-step clinical workflow wizard
    wizard = ScoliosisWizard()
    wizard.show()
    
    # Run the desktop application main execution loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
