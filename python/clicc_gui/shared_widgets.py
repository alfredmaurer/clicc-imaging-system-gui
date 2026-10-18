from PyQt6.QtWidgets import QWidget, QToolButton, QVBoxLayout
from PyQt6.QtCore import Qt

class CollapsibleGroup(QWidget):
    def __init__(self, title, content_layout, expanded=True, parent=None):
        super().__init__(parent)

        self.title = title  

        self.toggle_btn = QToolButton(checkable=True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )

        arrow = "▼" if expanded else "▶"
        self.toggle_btn.setText(f"{arrow}  {self.title}")

        self.toggle_btn.clicked.connect(self._on_toggled)

        self.content = QWidget()
        self.content.setObjectName("groupContent")
        self.content.setLayout(content_layout)
        self.content.setVisible(expanded)

        self.content.setStyleSheet("""
    QWidget#groupContent {
        border: 1px solid #C0C0C0;
        border-radius: 6px;
        padding: 6px;
        background-color: transparent;
    }
""")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.content)

        self._update_style(expanded)

    def _on_toggled(self, checked):
        self.content.setVisible(checked)

        arrow = "▼" if checked else "▶"
        self.toggle_btn.setText(f"{arrow}  {self.title}")

        self._update_style(checked)

    def _update_style(self, expanded):
        if expanded:
            self.toggle_btn.setStyleSheet("""
                QToolButton,
                QToolButton:checked,
                QToolButton:hover,
                QToolButton:pressed {
                    color: #a184f5;      
                    font-weight: bold;
                    background-color: transparent;
                    border: none;
                }
            """)
        else:
            self.toggle_btn.setStyleSheet("""
                QToolButton,
                QToolButton:hover,
                QToolButton:pressed {
                    color: #7a7a7a;       
                    font-weight: normal;
                    background-color: transparent;
                    border: none;
                }
            """)
        
    def set_expanded(self, expanded: bool):
        self.toggle_btn.blockSignals(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.blockSignals(False)

        arrow = "▼" if expanded else "▶"
        self.toggle_btn.setText(f"{arrow}  {self.title}")

        self.content.setVisible(expanded)
        self._update_style(expanded)



