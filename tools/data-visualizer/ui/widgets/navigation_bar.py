# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

from pyqtgraph.Qt import QtWidgets


class NavigationBarWidget(QtWidgets.QWidget):

    def __init__(self) -> None:
        super().__init__()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label_file = QtWidgets.QLabel()

        self.btn_prev = QtWidgets.QPushButton("<- Prev.")
        self.btn_next = QtWidgets.QPushButton("Next ->")

        layout.addWidget(self.label_file)
        layout.addStretch()
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.btn_next)


