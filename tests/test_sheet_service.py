import io

import pytest
from PIL import Image
from dependency_injector import providers

from src.core.container import Container
from src.schema.asset.asset_schema import ReviewState
from src.schema.enums import StageName, StageStatus
from src.schema.image.image_check_schema import Answer, AskedFor, Checklist, Fill
from src.schema.lesson.lesson_schema import GradeBand, VideoType
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_schema import Song
from src.service.image.fal_image_service import ImageReply
from src.service.llm.llm_service import ProviderReply
from src.service.storage.storage_service import LocalStorage

BOARD = {
    "grade_band": "3-5", "video_type": "journey", "shape": "s", "motifs": [], "flags": [], "model": "m",
    "prompt_version": "v",
    "cast": [{"id": "host", "kind": "person", "look": "A man", "size": "tall", "scale": "human"},
             {"id": "operator", "kind": "person", "look": "A Victorian man in a brown waistcoat", "size": "average",
              "scale": "human"},
             {"id": "ship", "kind": "thing", "look": "A wooden steamship with three masts", "size": "70 m", "scale": "landscape"}],
    "places": [{"id": "sea", "look": "A grey Atlantic", "lighting": "overcast", "scale": "landscape", "walkable": True,
                "host_outfit": "explorer"}],
    "shots": [{"id": "s02", "slot": "s02", "start": 4.0, "end": 8.0, "lines": [1], "kind": "sung", "repeat_of": None,
               "performer": "host", "continuity": "new_scene", "place": "sea", "subjects": ["host"],
               "size": "medium_close", "angle": "eye", "position": "centre", "start_state": "s", "action": "a",
               "end_state": "e", "camera": {"move": "static"}, "motion": "low", "screen_direction": "none",
               "transition_in": "cut"}],
}


def tiny_png(seed: int, aspect: str) -> bytes:
    """A real 1-colour PNG so code that opens images works; the colour comes from the seed."""
    w, h = {"1:1": (8, 8), "3:4": (6, 8), "16:9": (16, 9)}[aspect]
    out = io.BytesIO()
    Image.new("RGB", (w, h), (seed % 256, (seed * 7) % 256, (seed * 13) % 256)).save(out, format="PNG")
    return out.getvalue()


class FakeFal:
    """Returns a different fake image for every (prompt, seed); counts what it drew."""

    name = "fal"

    def __init__(self):
        self.calls = []

    def generate(self, model, prompt, refs, aspect, seed, thinking="medium"):
        self.calls.append((model, aspect, len(refs), seed))
        return ImageReply(data=tiny_png(seed + len(prompt), aspect), content_type="image/png", model=model,
                          seed=seed, cost_usd=0.15 if model == "nano-banana-pro" else 0.044)

    def upload(self, data, content_type):
        return f"https://fal.fake/{len(data)}"


class FakeVision:
    """Answers every checklist. `fail_first` fails the next N checks; `fail_when = (text, N)` fails the next N
    checks whose checklist mentions `text`, which stays deterministic when checks run in parallel."""

    name = "anthropic"

    def __init__(self, fail_first=0):
        self.fail_first = fail_first
        self.fail_score = 4
        self.fail_when = None
        self.failed_when = 0
        self.extra_answer = False
        self.asked_for = []
        self.drop = 0          # this many replies leave out the score and every rule answer
        self.checks = 0
        self.seen = []

    def parse(self, *a, **k):
        raise AssertionError("no text calls expected")

    def inspect(self, model, system, user, images, output_type):
        self.seen.append((user, [i.label for i in images]))
        if output_type is Fill:                                 # a follow-up: answer only the rules asked for
            asked = [line.split(". ", 1)[1] for line in user.split("leave every other field empty:\n", 1)[1].splitlines()
                     if line.startswith("- rule ")]
            if self.drop > 0:
                self.drop -= 1
                return ProviderReply(output=Fill(asked_for=[], score=None, score_reason=None, answers=[]),
                                     model=model, input_tokens=6000, output_tokens=20, cost_usd=0.001)
            asked_for = [AskedFor(item="what the moment asks for", reason="it is there", ok=True)]
            return ProviderReply(output=Fill(asked_for=asked_for, score=9, score_reason="a close match",
                                             answers=[Answer(rule=r, ok=True, reason="") for r in asked]),
                                 model=model, input_tokens=6000, output_tokens=200, cost_usd=0.001)
        self.checks += 1
        moment = [AskedFor(item="what the moment asks for", reason="it is there", ok=True)] if "Opening moment:" in user else []
        asked_for = self.asked_for or moment
        if self.drop > 0:
            self.drop -= 1
            return ProviderReply(output=Checklist(asked_for=[], score=0, score_reason="", answers=[]),
                                 model=model, input_tokens=6000, output_tokens=20, cost_usd=0.001)
        rules = [line.split(". ", 1)[1] for line in user.split("Rules:\n", 1)[1].splitlines() if line.strip()]
        answers = [Answer(rule=r, ok=True, reason="") for r in rules]
        if self.extra_answer:
            answers.append(Answer(rule="a rule the judge made up", ok=False, reason="ignored"))
        hit = self.fail_when is not None and self.fail_when[0] in user and self.failed_when < self.fail_when[1]
        if hit:
            self.failed_when += 1
        score, reason = ((self.fail_score, "the waistcoat is blue") if (self.checks <= self.fail_first or hit)
                         else (9, "a close match"))
        return ProviderReply(output=Checklist(asked_for=asked_for, score=score, score_reason=reason, answers=answers), model=model,
                             input_tokens=6000, output_tokens=200, cost_usd=0.001)


@pytest.fixture
def stage(container: Container, tmp_path):
    fal, vision = FakeFal(), FakeVision()
    container.storage.override(providers.Object(LocalStorage(str(tmp_path))))
    container.fal_image_service.override(providers.Object(fal))
    container.anthropic_service.override(providers.Object(vision))
    return container.sheet_service(), fal, vision


def board_run(container: Container) -> Run:
    n = len(container.run_repository().runs.list())
    song = container.song_repository().songs.add(Song(name=f"song-{n}", audio_path="a.mp3", lyrics="x"))
    run = container.run_repository().runs.add(Run(song_id=song.id, name="r", audience=GradeBand.G3_5))
    container.run_repository().stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD,
                                                           status=StageStatus.DONE, output=BOARD))
    return run


def approve_a_host(service, fal):
    candidates = service.host_candidates(GradeBand.G3_5, count=2)
    assert [c.params["candidate"] for c in candidates] == [1, 2] and fal.calls[0][2] == 0   # drawn from text alone
    return service.approve_host(GradeBand.G3_5, candidates[1].id)


def test_a_host_is_drawn_as_candidates_then_approved_and_completed(stage):
    service, fal, vision = stage
    made = approve_a_host(service, fal)
    assert [a.params["view"] for a in made] == ["front", "three_quarter", "side", "back", "to_camera", "smiling", "angry"]
    assert all(a.review in (ReviewState.APPROVED, ReviewState.PASSED) for a in made) and all(a.run_id is None for a in made)
    assert [c[1] for c in fal.calls[:8]] == ["3:4", "3:4", "3:4", "3:4", "3:4", "1:1", "1:1", "1:1"]
    sheets = service.host_sheets(GradeBand.G3_5)
    assert set(sheets) == {"front", "three_quarter", "side", "back", "to_camera", "smiling", "angry"}
    sheet = Image.open(io.BytesIO(service.host_sheet_image(GradeBand.G3_5)))
    assert sheet.width > sheet.height                               # seven tiles in a row, labelled by code
    losers = [a for a in service._assets.assets.list(target_id="host:3-5:everyday:front") if a.review == ReviewState.FAILED]
    assert len(losers) == 1


def test_sheets_need_an_approved_host_first(stage, container):
    service, _, _ = stage
    run = board_run(container)
    with pytest.raises(ValueError, match="no approved host"):
        service.run(run.id)


def test_sheets_are_drawn_in_the_host_style_checked_and_costed(stage, container):
    service, fal, vision = stage
    approve_a_host(service, fal)
    run = board_run(container)
    drawn_before = len(fal.calls)
    sheets = service.run(run.id)
    assert set(sheets) == {"operator", "ship", "sea"} and set(sheets["operator"]) == {"front", "three_quarter"}
    assert set(sheets["ship"]) == {"front"} and set(sheets["sea"]) == {"plate"}
    assert len(fal.calls) - drawn_before == 7 and all(c[2] == 1 for c in fal.calls[drawn_before:])   # 3 kit views + 4
    explorer = service.host_sheets(GradeBand.G3_5, "explorer")
    assert set(explorer) == {"front", "three_quarter", "to_camera", "smiling", "angry"}
    assert "field jacket" in explorer["front"].prompt and "now dressed as described" in explorer["front"].prompt
    base = service.host_sheets(GradeBand.G3_5)                  # each outfit view is drawn from the same-angle base view
    assert explorer["three_quarter"].params["refs"] == [base["three_quarter"].id]
    assert explorer["to_camera"].params["refs"] == [base["to_camera"].id]
    service.run(run.id, force=True)
    assert len(fal.calls) == drawn_before + 7                   # neither the kit nor the sheets are drawn twice
    assert all(a.review == ReviewState.PASSED for views in sheets.values() for a in views.values())
    state = container.run_repository().stage(run.id, StageName.VISUAL_IDENTITY)
    assert state.status == StageStatus.DONE and set(state.output["sheets"]) == {"operator", "ship", "sea"}
    cost = container.asset_repository().run_cost(run.id)
    assert cost == pytest.approx(4 * 0.044 + 4 * 0.001)                            # kit images carry no run cost
    assert container.storage().exists(sheets["sea"]["plate"].path)


def test_a_sheet_that_fails_every_draw_below_the_floor_fails_the_stage(stage, container):
    service, fal, vision = stage
    approve_a_host(service, fal)
    vision.fail_when = ("Victorian", 2)                        # the operator's front view fails both draws at 4/10
    run = board_run(container)
    sheets = service.run(run.id)
    operator = sheets["operator"]["front"]
    assert operator.review == ReviewState.FAILED and "waistcoat is blue" in operator.review_notes
    tries = [a for a in container.asset_repository().assets.list(run_id=run.id, target_id="operator")
             if a.params["view"] == "front"]
    assert sorted(a.params["try"] for a in tries) == [1, 2] and all(a.review == ReviewState.FAILED for a in tries)
    state = container.run_repository().stage(run.id, StageName.VISUAL_IDENTITY)
    assert state.status == StageStatus.FAILED and "operator/front: failed every draw; best 4/10" in state.error


def test_a_sheet_that_fails_every_draw_uses_its_best_try_above_the_floor(stage, container):
    service, fal, vision = stage
    approve_a_host(service, fal)
    vision.fail_when, vision.fail_score = ("Victorian", 2), 5   # fails the bar of 6, meets the floor of 5
    run = board_run(container)
    sheets = service.run(run.id)
    operator = sheets["operator"]["front"]
    assert operator.review == ReviewState.USED_BEST and operator.params["score"] == 5
    state = container.run_repository().stage(run.id, StageName.VISUAL_IDENTITY)
    assert state.status == StageStatus.DONE and state.output["used_best"] == [operator.id] and state.error is None
    vision.fail_score = 4


def test_the_same_sheet_is_reused_not_redrawn(stage, container):
    service, fal, vision = stage
    approve_a_host(service, fal)
    first = board_run(container)
    service.run(first.id)
    drawn = len(fal.calls)
    second = board_run(container)                              # another run, identical looks
    sheets = service.run(second.id)
    assert len(fal.calls) == drawn and all(a.cost_usd == 0.0 for views in sheets.values() for a in views.values())
    assert sheets["ship"]["front"].params["reused_from"]
    service.run(second.id)
    assert len(fal.calls) == drawn                                                # the stage itself is cached too
