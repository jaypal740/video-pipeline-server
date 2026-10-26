from types import TracebackType

from sqlalchemy.orm import Session, sessionmaker


class UnitOfWork:
    """One transaction spanning several repository calls.

    `with uow as session:` opens a session; the block commits on clean exit and rolls back on
    exception. Services pass `session` to every repository call inside the block so all writes
    land together or not at all. A new UnitOfWork is built per use (container Factory).
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> Session:
        self._session = self._session_factory()
        return self._session

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        session = self._session
        self._session = None
        if session is None:
            return
        try:
            if exc_type is None:
                session.commit()
            else:
                session.rollback()
        finally:
            session.close()
