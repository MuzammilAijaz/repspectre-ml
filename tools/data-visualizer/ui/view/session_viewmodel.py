# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

from pathlib import Path

from model.session_repository import Session, SessionRepository
from PySide6.QtCore import QObject, Signal


class SessionViewModel(QObject):

    session_changed = Signal(object)  # Emits Session | None
    dataset_changed = Signal(Path)    # Emits current Path

    def __init__(self, repository: SessionRepository) -> None:
        super().__init__()

        self.repository = repository
        self.current_idx: int = 0

        # caching
        self.session_count: int = self.repository.get_sessions_count()

        # initial session load
        self.current_session: Session | None = self.repository.load_session(self.current_idx)

    def select_dataset(self, data_dir: Path) -> None:
        """Change the active dataset directory and load the first session."""
        self.repository.set_data_dir(data_dir)
        self.session_count = self.repository.get_sessions_count()
        self.current_idx = 0
        self.current_session = self.repository.load_session(self.current_idx)
        self.dataset_changed.emit(data_dir)
        self.session_changed.emit(self.current_session)

    def load_session(self, idx: int) -> None:
        if self.session_count == 0:
            self.current_idx = 0
            self.current_session = None
        else:
            self.current_idx = idx
            self.current_session = self.repository.load_session(self.current_idx)
        self.session_changed.emit(self.current_session)

    def next_session(self) -> None:
        if self.session_count == 0:
            return
        self.current_idx = (self.current_idx + 1) % self.session_count
        self.load_session(self.current_idx)

    def previous_session(self) -> None:
        if self.session_count == 0:
            return
        self.current_idx = (self.current_idx - 1) % self.session_count
        self.load_session(self.current_idx)


