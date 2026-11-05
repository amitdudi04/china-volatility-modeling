"""Optional legacy PyQt GUI entry point.

The research data adapter and offline dashboard must import without PyQt.
Only the explicitly selected *legacy* graphical UI needs this dependency.
"""
def main():
    import sys
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from src.gui.main_window import QuantResearchDashboard
    except ImportError as error:
        raise RuntimeError(
            "Legacy PyQt6 dashboard is optional. Use 'python main.py --run-gui' "
            "for the verified offline research dashboard."
        ) from error
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app=QApplication.instance() or QApplication(sys.argv)
    window=QuantResearchDashboard()
    window.showMaximized()
    return app.exec()
