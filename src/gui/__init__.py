import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.gui.main_window import QuantResearchDashboard

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    ex = QuantResearchDashboard()
    ex.showMaximized()
    sys.exit(app.exec())
