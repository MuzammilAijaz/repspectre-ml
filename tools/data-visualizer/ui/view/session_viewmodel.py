from model.session_repository import Session, SessionRepository
from PySide6.QtCore import QObject, Signal


class SessionViewModel(QObject):

    session_changed = Signal(Session)

    def __init__(self, repository: SessionRepository) -> None:
        super().__init__()

        self.repository = repository
        self.current_idx: int = 0

        # caching
        self.session_count: int = self.repository.get_sessions_count()

        # initial session load
        self.current_session: Session = self.repository.load_session(self.current_idx)

    def load_session(self, idx: int) -> None:
        self.current_idx = idx
        self.current_session = self.repository.load_session(self.current_idx)
        self.session_changed.emit(self.current_session)

    def next_session(self) -> None:
        self.current_idx = (self.current_idx + 1) % self.session_count
        self.load_session(self.current_idx)

    def previous_session(self) -> None:
        self.current_idx = (self.current_idx - 1) % self.session_count
        self.load_session(self.current_idx)

