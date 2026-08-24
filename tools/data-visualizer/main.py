# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import sys
from pathlib import Path

# Add visualizer dir and src dir to path
VISUALIZER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALIZER_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(VISUALIZER_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZER_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pyqtgraph as pg  # noqa: E402
from ui.windows.main_window import MainWindow  # noqa: E402

app = pg.mkQApp()

window = MainWindow()
window.show()

app.exec()

