from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout

class KPICard(QFrame):
    def __init__(self, title, value, is_positive=None):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 8px;
                border: 1px solid #2a2a2a;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        
        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet("color: #aaaaaa; font-size: 13px; font-family: Segoe UI; font-weight: bold; background: transparent; border: none;")
        
        color = "#ffffff"
        if is_positive is True or is_positive == "good": 
            color = "#34a853"
        elif is_positive is False or is_positive == "bad": 
            color = "#ea4335"
        elif is_positive == "warning":
            color = "#fbbc05"
            
        val_lbl = QLabel(str(value))
        val_lbl.setStyleSheet(f"color: {color}; font-size: 26px; font-weight: bold; font-family: Consolas; background: transparent; border: none;")
        
        layout.addWidget(title_lbl)
        layout.addWidget(val_lbl)

class KPIStrip(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("QFrame { background-color: transparent; border: none; }")
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(15)

    def add_kpi(self, title, value, is_positive=None):
        card = KPICard(title, value, is_positive)
        self.layout.addWidget(card)
        return card

    def add_stretch(self):
        self.layout.addStretch()
