from PyQt6.QtWidgets import QApplication
from gui.gui import ModelTrainerApp
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    trainer_app = ModelTrainerApp()
    trainer_app.show()
    sys.exit(app.exec())