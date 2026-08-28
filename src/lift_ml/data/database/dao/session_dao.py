from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from lift_ml.data.database.dao.base_dao import BaseDao
from lift_ml.data.database.entity.lift_context_entity import LiftContext
from lift_ml.data.database.entity.session_entity import DbSession


class SessionDao(BaseDao[DbSession]):
    """Data Access Object for DbSession entities."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, DbSession)

    def get_by_id_with_data(self, session_id: int) -> DbSession | None:
        """Fetch a single DbSession by ID with full_imu_raw and lift_context preloaded."""
        stmt = (
            select(DbSession)
            .where(DbSession.session_id == session_id)
            .options(
                selectinload(DbSession.full_imu_raw),
                selectinload(DbSession.lift_context),
            )
        )
        return self._session.scalars(stmt).first()

    def get_all_with_data(self) -> list[DbSession]:
        """Fetch all sessions with full_imu_raw and lift_context preloaded."""
        stmt = (
            select(DbSession)
            .options(
                selectinload(DbSession.full_imu_raw),
                selectinload(DbSession.lift_context),
            )
            .order_by(DbSession.session_id.asc())
        )
        return list(self._session.scalars(stmt).all())

    def get_by_motion_state(self, motion_state: str) -> list[DbSession]:
        """Fetch sessions matching a motion state with data preloaded."""
        stmt = (
            select(DbSession)
            .where(DbSession.motion_state == motion_state)
            .options(
                selectinload(DbSession.full_imu_raw),
                selectinload(DbSession.lift_context),
            )
            .order_by(DbSession.session_id.asc())
        )
        return list(self._session.scalars(stmt).all())

    def get_by_lift_category(self, lift_category: str) -> list[DbSession]:
        """Fetch sessions matching a lift category via joined lift_context with data preloaded."""
        stmt = (
            select(DbSession)
            .join(DbSession.lift_context)
            .where(LiftContext.lift_category == lift_category)
            .options(
                selectinload(DbSession.full_imu_raw),
                selectinload(DbSession.lift_context),
            )
            .order_by(DbSession.session_id.asc())
        )
        return list(self._session.scalars(stmt).all())

    def get_distinct_motion_states(self) -> list[str]:
        """Fetch distinct motion states present in the session table."""
        stmt = select(DbSession.motion_state).distinct().order_by(DbSession.motion_state.asc())
        return list(self._session.scalars(stmt).all())

    def get_distinct_lift_categories(self) -> list[str]:
        """Fetch distinct lift categories present in the lift_context table."""
        stmt = (
            select(LiftContext.lift_category).distinct().order_by(LiftContext.lift_category.asc())
        )
        return list(self._session.scalars(stmt).all())
