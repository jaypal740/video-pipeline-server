import pytest

from src.core.container import Container
from src.schema.asset.asset_schema import Asset
from src.schema.asset.cost_entry_schema import CostEntry
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState


@pytest.fixture
def run(container: Container) -> Run:
    return container.run_repository().runs.add(Run(name="Ocean", audio_path="a.mp3", script_path="l.txt"))


def test_run_crud(container: Container):
    repo = container.run_repository().runs
    run = repo.add(Run(name="A", audio_path="a.mp3", script_path="l.txt", model_pins={"video": "kling"}))
    fetched = repo.get(run.id)
    assert fetched.name == "A" and fetched.model_pins == {"video": "kling"}
    fetched.name = "B"
    assert repo.update(fetched).name == "B"
    repo.delete(run.id)
    assert repo.get(run.id) is None


def test_stage_states_come_back_in_pipeline_order(container: Container, run: Run):
    repo = container.run_repository()
    repo.stage_states.add_many([StageState(run_id=run.id, stage=StageName.VIDEO),
                                StageState(run_id=run.id, stage=StageName.INTAKE, status=StageStatus.DONE)])
    assert [s.stage for s in repo.stages(run.id)] == [StageName.INTAKE, StageName.VIDEO]
    assert repo.stage(run.id, StageName.INTAKE).status == StageStatus.DONE
    assert repo.stage(run.id, StageName.QC) is None


def test_cached_asset_is_found_by_key(container: Container, run: Run):
    repo = container.asset_repository()
    asset = repo.assets.add(Asset(run_id=run.id, stage=StageName.KEYFRAMES, kind=AssetKind.IMAGE,
                                  path="k.png", sha256="0" * 64, cache_key="abc"))
    assert repo.find_cached("abc").id == asset.id
    assert repo.find_cached("missing") is None


def test_run_cost_sums_entries(container: Container, run: Run):
    repo = container.asset_repository()
    assert repo.run_cost(run.id) == 0.0
    repo.cost_entries.add_many([CostEntry(run_id=run.id, stage=StageName.VIDEO, provider="fal", model="m", cost_usd=0.4),
                                CostEntry(run_id=run.id, stage=StageName.KEYFRAMES, provider="fal", model="m", cost_usd=0.1)])
    assert repo.run_cost(run.id) == pytest.approx(0.5)


def test_unit_of_work_rolls_back_every_write(container: Container):
    repo = container.run_repository().runs
    with pytest.raises(RuntimeError):
        with container.unit_of_work() as session:
            repo.add(Run(name="X", audio_path="a", script_path="b"), session=session)
            raise RuntimeError("boom")
    assert repo.list() == []
