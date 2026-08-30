from sqlalchemy import select
from sqlalchemy.orm import Session

from lift_ml.data.database.dao.base_dao import BaseDao
from lift_ml.data.database.entity.full_imu_raw_entity import FullImuRaw


class FullImuRawDao(BaseDao[FullImuRaw]):
    """Data Access Object for FullImuRaw sensor readings."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, FullImuRaw)

    def get_by_session_id(self, session_id: int) -> list[FullImuRaw]:
        """Fetch all IMU readings for a specific session ordered by ID."""
        stmt = (
            select(FullImuRaw)
            .where(FullImuRaw.session_id == session_id)
            .order_by(FullImuRaw.id.asc())
        )
        return list(self._session.scalars(stmt).all())
