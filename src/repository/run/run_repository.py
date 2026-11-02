from sqlalchemy.orm import Session, sessionmaker

from src.model.run.run_model import RunRow
from src.model.run.stage_state_model import StageStateRow
from src.repository.base_repository import BaseRepository
from src.schema.enums import StageName
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState


class RunRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self.runs = BaseRepository(session_factory, RunRow, Run)
        self.stage_states = BaseRepository(session_factory, StageStateRow, StageState)

    def stage(self, run_id: str, stage: StageName, session: Session | None = None) -> StageState | None:
        return self.stage_states.find_one(session=session, run_id=run_id, stage=stage)

    def stages(self, run_id: str, session: Session | None = None) -> list[StageState]:
        order = list(StageName)
        return sorted(self.stage_states.list(session=session, run_id=run_id), key=lambda s: order.index(s.stage))

    def waiting_review(self, run_id: str, session: Session | None = None) -> list[str]:
        """The asset ids this run's stages are waiting on a person for, wherever those images were drawn:
        a stage lists them in its output, so a reused image from an earlier run is included."""
        return [asset_id for state in self.stages(run_id, session=session)
                for asset_id in (state.output or {}).get("needs_review", [])]
