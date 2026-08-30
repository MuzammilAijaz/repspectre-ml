from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.orm import Session as SqlSession


class Base(DeclarativeBase):
    """Declarative base class for all database models."""


class DatabaseContext:
    """Manages SQLAlchemy Engine and Session lifecycle with context-manager support."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._engine: Engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self._session_factory: sessionmaker[SqlSession] = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> Engine:
        return self._engine

    @contextmanager
    def session(self) -> Generator[SqlSession]:
        """Context manager for obtaining a database session."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
