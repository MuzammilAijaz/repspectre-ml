from sqlalchemy import select
from sqlalchemy.orm import Session

from lift_ml.data.database.sensor_database import Base


class BaseDao[T: Base]:
    """Generic DAO providing standard read operations using SQLAlchemy 2.0 select()."""

    def __init__(self, session: Session, model_cls: type[T]) -> None:
        self._session = session
        self._model_cls = model_cls

    def get_by_id(self, entity_id: int) -> T | None:
        """Fetch a single record by primary key."""
        return self._session.get(self._model_cls, entity_id)

    def get_all(self) -> list[T]:
        """Fetch all records for the model class."""
        stmt = select(self._model_cls)
        return list(self._session.scalars(stmt).all())
