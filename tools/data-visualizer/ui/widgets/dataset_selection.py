# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import logging
from pathlib import Path
from typing import Final

from pyqtgraph.Qt import QtWidgets
from PySide6.QtCore import Signal

logger: Final = logging.getLogger("visualizer.ui.dataset_selection")


class DatasetSelectionWidget(QtWidgets.QWidget):

    dataset_selected = Signal(Path)

    def __init__(self) -> None:
        super().__init__()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Dataset selection button with hierarchical popup menu
        self.btn_dataset = QtWidgets.QPushButton("📂 Select Dataset ▾")
        self.menu_dataset = QtWidgets.QMenu(self)
        self.btn_dataset.setMenu(self.menu_dataset)

        layout.addWidget(self.btn_dataset)

    def set_available_datasets(
        self, datasets: list[Path], data_root: Path, current_path: Path | None = None
    ) -> None:
        """Build hierarchical nested QMenu from list of dataset directory paths."""
        self.menu_dataset.clear()

        if not datasets:
            logger.warning("No datasets available to populate selection menu.")
            action = self.menu_dataset.addAction("No CSV folders found")
            action.setEnabled(False)
            return

        logger.info("Populating dataset menu with %d folders", len(datasets))

        # Dictionary to store created sub-menus: path_tuple -> QMenu
        submenus: dict[tuple[str, ...], QtWidgets.QMenu] = {}

        for path in datasets:
            try:
                rel_parts = path.relative_to(data_root).parts
            except ValueError:
                rel_parts = (path.name,)

            # Create or find parent menus for all but the last component
            parent_menu = self.menu_dataset
            for i in range(1, len(rel_parts)):
                sub_key = rel_parts[:i]
                if sub_key not in submenus:
                    new_menu = parent_menu.addMenu(f"📁 {rel_parts[i - 1]}")
                    submenus[sub_key] = new_menu
                parent_menu = submenus[sub_key]

            # The leaf action
            leaf_name = rel_parts[-1] if rel_parts else path.name
            action = parent_menu.addAction(leaf_name)
            # Capture path in lambda default arg
            action.triggered.connect(
                lambda _checked=False, p=path: self._on_action_triggered(p)
            )

        if current_path is not None:
            self.set_current_dataset_label(current_path, data_root)

    def _on_action_triggered(self, path: Path) -> None:
        logger.info("Dataset menu action clicked: %s", path)
        self.dataset_selected.emit(path)

    def set_current_dataset_label(self, current_path: Path, data_root: Path) -> None:
        """Update button label to show current dataset relative path."""
        try:
            rel = current_path.relative_to(data_root)
            display_text = " / ".join(rel.parts)
        except ValueError:
            display_text = current_path.name
        self.btn_dataset.setText(f"📂 {display_text} ▾")
