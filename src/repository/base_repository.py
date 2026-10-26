from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import delete, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from src.model.base_model import Base
from src.schema.base_schema import Entity, utcnow


@contextmanager
def session_scope(
    session_factory: sessionmaker[Session], session: Session | None, write: bool = False
) -> Iterator[Session]:
    """Yield the caller's session (flushed, never committed) or open and own a new one.

    A session passed in belongs to a UnitOfWork, which commits or rolls back the whole block.
    With no session the repo opens its own: a transaction for writes, a plain session for reads.
    """
    if session is not None:
        yield session
        if write:
            session.flush()
        return
    if write:
        with session_factory.begin() as own:
            yield own
        return
    with session_factory() as own:
        yield own


class BaseRepository[T: Entity]:
    """CRUD over one table. Maps ORM row <-> domain model at the boundary."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        row_type: type[Base],
        model_type: type[T],
    ) -> None:
        self._session_factory = session_factory
        self._row_type = row_type
        self._model_type = model_type
        self._columns = [c.key for c in inspect(row_type).columns]

    def get(self, id: str, session: Session | None = None) -> T | None:
        with session_scope(self._session_factory, session) as s:
            row = s.get(self._row_type, id)
            return None if row is None else self._to_model(row)

    def list(self, session: Session | None = None, **filters: Any) -> list[T]:
        stmt = self._filtered(filters).order_by(self._row_type.created_at, self._row_type.id)
        with session_scope(self._session_factory, session) as s:
            return [self._to_model(row) for row in s.scalars(stmt)]

    def add(self, entity: T, session: Session | None = None) -> T:
        row = self._row_type(**entity.model_dump())
        with session_scope(self._session_factory, session, write=True) as s:
            s.add(row)
        return self._to_model(row)

    def add_many(self, entities: list[T], session: Session | None = None) -> list[T]:
        rows = [self._row_type(**entity.model_dump()) for entity in entities]
        with session_scope(self._session_factory, session, write=True) as s:
            s.add_all(rows)
        return [self._to_model(row) for row in rows]

    def update(self, entity: T, session: Session | None = None) -> T:
        with session_scope(self._session_factory, session, write=True) as s:
            row = s.get(self._row_type, entity.id)
            if row is None:
                raise LookupError(f"{self._model_type.__name__} {entity.id} not found")
            data = entity.model_dump(exclude={"id", "created_at", "updated_at"})
            for key, value in data.items():
                setattr(row, key, value)
            row.updated_at = utcnow()
            s.flush()
            return self._to_model(row)

    def delete(self, id: str, session: Session | None = None) -> None:
        with session_scope(self._session_factory, session, write=True) as s:
            row = s.get(self._row_type, id)
            if row is not None:
                s.delete(row)

    def delete_where(self, session: Session | None = None, **filters: Any) -> int:
        """Delete every row matching the filters (column == value; None matches NULL). Returns the row count."""
        stmt = self._where(delete(self._row_type), filters)
        with session_scope(self._session_factory, session, write=True) as s:
            return s.execute(stmt).rowcount

    def find_one(self, session: Session | None = None, **filters: Any) -> T | None:
        stmt = self._filtered(filters).limit(1)
        with session_scope(self._session_factory, session) as s:
            row = s.scalars(stmt).first()
            return None if row is None else self._to_model(row)

    def _filtered(self, filters: dict[str, Any]):
        return self._where(select(self._row_type), filters)

    def _where(self, stmt, filters: dict[str, Any]):
        for key, value in filters.items():
            stmt = stmt.where(getattr(self._row_type, key) == value)
        return stmt

    def _to_model(self, row: Base) -> T:
        return self._model_type.model_validate({key: getattr(row, key) for key in self._columns})
