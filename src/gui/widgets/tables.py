import pandas as pd
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt

class DataTable(QFrame):
    def __init__(self, title, df):
        super().__init__()
        self.setStyleSheet("""
            QFrame { background-color: #1e1e1e; border-radius: 8px; border: 1px solid #2a2a2a; }
            QTableWidget { 
                background-color: #1e1e1e; 
                alternate-background-color: #1a1a1a; 
                color: #ffffff; 
                gridline-color: #333; 
                border: none;
                font-size: 14px;
            }
            QHeaderView::section { 
                background-color: #2c2c2c; 
                color: #ffffff; 
                border: 1px solid #333; 
                padding: 8px; 
                font-weight: bold; 
                font-family: Consolas;
                font-size: 14px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #ff9900; background: transparent; border: none; padding-bottom: 5px;")
        layout.addWidget(title_lbl)

        if df is None or df.empty:
            empty_lbl = QLabel("Data Unavailable")
            empty_lbl.setStyleSheet("color: #e6e6e6; background: transparent; border: none;")
            layout.addWidget(empty_lbl)
            return

        table = QTableWidget()
        table.setSortingEnabled(True)
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        
        for i, col_name in enumerate(df.columns):
            item = QTableWidgetItem(str(col_name))
            item.setToolTip(str(col_name))
            table.setHorizontalHeaderItem(i, item)
            
        table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setWordWrap(True)
        table.horizontalHeader().setFixedHeight(50)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = str(df.iat[i, j])
                if pd.api.types.is_numeric_dtype(df.dtypes.iloc[j]):
                    try:
                        val = f"{float(val):.4f}"
                    except ValueError:
                        pass
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, j, item)
                
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(False)
        
        for col in range(table.columnCount()):
            table.setColumnWidth(col, max(140, table.columnWidth(col)))
            
        table.verticalHeader().setDefaultSectionSize(35)
        layout.addWidget(table)
