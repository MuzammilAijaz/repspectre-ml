from lift_ml.data.database.dao.session_dao import SessionDao
from lift_ml.data.database.sensor_database import DatabaseContext
from lift_ml.data.domain.session import Session


class SessionRepository:
    """Repository mapping database entities to domain Session models."""

    def __init__(self, ctx: DatabaseContext | str) -> None:
        if isinstance(ctx, str):
            self._ctx = DatabaseContext(ctx)
        else:
            self._ctx = ctx

    def get_session_by_id(self, session_id: int) -> Session | None:
        """Fetch a single session by its ID and convert to domain Session."""
        with self._ctx.session() as s:
            dao = SessionDao(s)
            entity = dao.get_by_id_with_data(session_id)
            return entity.to_session() if entity is not None else None

    def get_all_sessions(self) -> list[Session]:
        """Fetch all recorded sessions converted to domain Session objects."""
        with self._ctx.session() as s:
            dao = SessionDao(s)
            entities = dao.get_all_with_data()
            return [e.to_session() for e in entities]

    def get_sessions_by_motion_state(self, motion_state: str) -> list[Session]:
        """Fetch all sessions matching the given motion state."""
        with self._ctx.session() as s:
            dao = SessionDao(s)
            entities = dao.get_by_motion_state(motion_state)
            return [e.to_session() for e in entities]

    def get_lift_sessions_by_lift_category(self, lift_category: str) -> list[Session]:
        """Fetch all lift sessions matching the given lift category."""
        with self._ctx.session() as s:
            dao = SessionDao(s)
            entities = dao.get_by_lift_category(lift_category)
            return [e.to_session() for e in entities]

    def get_distinct_motion_states(self) -> list[str]:
        """Fetch all unique motion states recorded in the database."""
        with self._ctx.session() as s:
            dao = SessionDao(s)
            return dao.get_distinct_motion_states()

    def get_distinct_lift_categories(self) -> list[str]:
        """Fetch all unique lift categories recorded in the database."""
        with self._ctx.session() as s:
            dao = SessionDao(s)
            return dao.get_distinct_lift_categories()
