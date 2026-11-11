import io

import numpy as np
import pytest
from PIL import Image
from dependency_injector import providers

from src.core.container import Container
from src.schema.asset.asset_schema import ReviewState
from src.schema.enums import StageName, StageStatus
from src.schema.image.image_check_schema import Answer, Checklist, Correction
from src.schema.lesson.lesson_schema import Beat, GradeBand, LessonPlan
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_schema import Song
from src.schema.storyboard.script_schema import AnimationScript
from src.service.drawing.cutout import KEY, cut, cut_out
from src.service.drawing.drawing_prompt import drawing_brief, drawing_prompt, lead, must_not_show, reference_of
from src.service.image.fal_image_service import ImageReply
from src.service.llm.llm_service import ProviderReply
from src.service.storage.storage_service import LocalStorage


def show(drawing, at, pos="center"):
    return {"verb": "show", "drawing": drawing, "pos": pos, "at": at}


BOARD = {
    "grade_band": "3-5", "style": "A bold 2D cartoon with thick outlines", "palette": ["#12A594", "#F2F2F2"],
    "performer": "Dot, a round teal bacterium.", "model": "m", "prompt_version": "v",
    "drawings": [{"id": "bench", "kind": "background", "look": "A white lab bench"},
                 {"id": "dot_small", "kind": "performer", "look": "Tiny and curled up", "motion": "wobbles"},
                 {"id": "dot", "kind": "performer", "look": "Arms wide", "motion": "sways", "sings": True},
                 {"id": "dot_point", "kind": "performer", "look": "Pointing up", "of": "dot", "motion": "bobs"},
                 {"id": "dish", "kind": "object", "look": "A clear glass dish"},
                 {"id": "dish_full", "kind": "object", "look": "The dish full of colonies", "of": "dish"},
                 {"id": "arrow", "kind": "icon", "look": "A curved arrow"}],
    "beats": [{"beat": "b01", "start": 0, "end": 4, "type": "explain", "background": "bench",
               "actions": [show("dish", 0), show("dot_point", 1, "left")]},
              {"beat": "b02", "start": 4, "end": 8, "type": "explain", "background": "bench",
               "actions": [{"verb": "swap", "drawing": "dish", "to": "dish_full", "at": 5}, show("arrow", 6, "top"),
                           {"verb": "swap", "drawing": "dot_point", "to": "dot_small", "at": 7}]}],
}
SCRIPT = AnimationScript(**BOARD)
PLAN = LessonPlan(topic="Bacteria", subject="Science", summary="S.", key_terms=[], must_not_teach=[],
                  grade_band=GradeBand.G3_5, model="m", prompt_version="v",
                  beats=[Beat(id="b01", start=0, end=4, type="explain", avoid="One giant germ."),
                         Beat(id="b02", start=4, end=8, type="explain", avoid="A dish with no colonies.")])
BY_ID = {d.id: d for d in SCRIPT.drawings}


def picture(colour, aspect="1:1", box=(8, 8, 24, 24), size=32) -> bytes:
    """A drawing as the image model returns it: a backdrop in one colour, or a square on the key colour."""
    image = Image.new("RGB", (size * 16 // 9, size) if aspect == "16:9" else (size, size), colour if aspect == "16:9" else KEY)
    if aspect != "16:9":
        image.paste(colour, box)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class FakeFal:
    """Draws a different square for every seed; with `whole`, fills the frame so nothing can be cut out."""

    name = "fal"

    def __init__(self):
        self.calls, self.whole = [], False

    def generate(self, model, prompt, refs, aspect, seed, thinking="medium"):
        self.calls.append((prompt, len(refs)))
        colour = (seed % 200, (seed * 7) % 200, 40)
        data = picture(colour, aspect, box=(0, 0, 32, 32) if self.whole else (8, 8, 24, 24))
        return ImageReply(data=data, content_type="image/png", model=model, seed=seed, cost_usd=0.044)

    def upload(self, data, content_type):
        return f"https://fal.fake/{len(data)}"


class FakeVision:
    """The judge: passes every picture, except the first `fail_first` checks. It also writes the correction
    a retry is drawn with."""

    name = "anthropic"

    def __init__(self):
        self.seen, self.fail_first, self.asked = [], 0, []

    def parse(self, model, system, user, output_type, effort):
        self.asked.append(user)
        return ProviderReply(output=Correction(text="Its arms are held out wide."), model=model, input_tokens=400,
                             output_tokens=20, cost_usd=0.0005)

    def inspect(self, model, system, user, images, output_type):
        self.seen.append((user, [i.label for i in images]))
        rules = [line.split(". ", 1)[1] for line in user.split("Rules:\n", 1)[1].splitlines() if line.strip()]
        score = 3 if len(self.seen) <= self.fail_first else 9
        return ProviderReply(output=Checklist(asked_for=[], score=score, score_reason="as seen",
                                              answers=[Answer(rule=r, ok=True, reason="") for r in rules]),
                             model=model, input_tokens=6000, output_tokens=200, cost_usd=0.001)


@pytest.fixture
def stage(container: Container, tmp_path):
    fal, vision = FakeFal(), FakeVision()
    container.storage.override(providers.Object(LocalStorage(str(tmp_path))))
    container.fal_image_service.override(providers.Object(fal))
    container.anthropic_service.override(providers.Object(vision))
    song = container.song_repository().songs.add(Song(name="colony", audio_path="a.mp3", lyrics="x"))
    run = container.run_repository().runs.add(Run(song_id=song.id, name="r", audience=GradeBand.G3_5))
    container.run_repository().stage_states.add(StageState(run_id=run.id, stage=StageName.LESSON, status=StageStatus.DONE,
                                                           output=PLAN.model_dump(mode="json")))
    container.run_repository().stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD,
                                                           status=StageStatus.DONE, output=BOARD))
    return container.drawing_service(), fal, vision, run


def alpha(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)))[..., 3]


def test_the_key_colour_around_a_subject_and_in_its_gaps_is_cut_away():
    image = Image.open(io.BytesIO(picture((20, 160, 40), box=(16, 16, 112, 112), size=128)))
    image.paste(KEY, (30, 30, 50, 50))                         # a gap in the subject, such as between arm and body
    image.paste(KEY, (80, 80, 84, 84))                         # a speck of the key colour
    image.paste((230, 100, 130), (60, 30, 80, 50))             # a pink mouth: like the key, but not it
    out = io.BytesIO()
    image.save(out, format="PNG")
    cut, found = cut_out(out.getvalue())
    a = alpha(cut)
    assert found == [] and a[0, 0] == 0 and a[120, 60] == 0 and a[100, 100] == 255
    assert a[40, 40] == 0 and a[82, 82] == 255 and a[40, 70] == 255


def test_a_subject_that_touches_the_edge_or_is_tiny_is_no_cut_out():
    assert "the frame's edge" in cut_out(picture((20, 160, 40), box=(0, 0, 32, 20)))[1][0]
    assert "fills 0% of the frame" in cut_out(picture((20, 160, 40), box=(15, 15, 17, 17)))[1][0]
    grazing = picture((20, 160, 40), box=(0, 10, 12, 16))      # on 6 of the edge's 128 pixels: a loop's frame may, a drawing not
    assert "only 95% of the frame's edge" in cut_out(grazing)[1][0]
    assert cut(np.asarray(Image.open(io.BytesIO(grazing)).convert("RGB")))[1] == []


def test_the_performer_is_drawn_from_its_lead_and_a_pose_from_what_it_is_a_pose_of():
    assert lead(SCRIPT).id == "dot"                            # it sings and has a pose of its own
    refs = {d.id: getattr(reference_of(SCRIPT, d), "id", None) for d in SCRIPT.drawings}
    assert refs == {"bench": None, "dot_small": "dot", "dot": None, "dot_point": "dot", "dish": None,
                    "dish_full": "dish", "arrow": None}


def test_a_cut_out_is_asked_for_on_the_key_colour_and_a_backdrop_fills_the_frame():
    cut = drawing_prompt(SCRIPT, BY_ID["dot"], None)
    assert cut.startswith("A bold 2D cartoon with thick outlines. The character is Dot, a round teal bacterium. "
                          "In this drawing: Arms wide. Its face is turned to the viewer")
    assert "flat solid pure magenta (#FF00FF)" in cut and "#12A594, #F2F2F2" in cut and "Image 1" not in cut
    assert "Image 1 shows the same subject" in drawing_prompt(SCRIPT, BY_ID["dish_full"], BY_ID["dish"])
    back = drawing_prompt(SCRIPT, BY_ID["bench"], None)
    assert "full-frame backdrop" in back and "magenta" not in back
    assert "no cartoon face" in drawing_prompt(SCRIPT, BY_ID["dish"], None)


def test_text_an_added_thing_an_added_face_and_a_hidden_singing_face_fail_a_drawing_on_their_own():
    text = "No letters, digits, labels, logos or watermarks anywhere in the picture"
    only = "The picture shows only what the description names"
    assert drawing_brief(SCRIPT, BY_ID["bench"]).fatal == [text, "No people, animals or characters anywhere in the picture"]
    dot = drawing_brief(SCRIPT, BY_ID["dot"]).fatal
    assert dot[0] == text and dot[1].startswith(only) and dot[2] == "The character's face and mouth are clearly visible"
    dish = drawing_brief(SCRIPT, BY_ID["dish_full"])
    assert not any(r.startswith("The subject is the same one") for r in drawing_brief(SCRIPT, BY_ID["dot_small"]).rules)
    assert dish.fatal[1].startswith(only) and "add nothing under, behind or beside it" in drawing_prompt(SCRIPT, BY_ID["dish"], None)
    assert dish.fatal[2].startswith("No cartoon face is added") and dish.rules[-1].startswith("The subject is the same one")
    assert "magenta (#FF00FF) background, which is what was asked for" in dish.description


def test_the_judge_is_told_one_drawings_look_and_what_its_beats_forbid_not_the_whole_character():
    avoid = {b.id: b.avoid for b in PLAN.beats}
    assert must_not_show(SCRIPT, BY_ID["dot_point"], avoid) == ["One giant germ", "A dish with no colonies"]
    assert must_not_show(SCRIPT, BY_ID["dish_full"], avoid) == ["A dish with no colonies"]      # only swapped in, in b02
    assert must_not_show(SCRIPT, BY_ID["arrow"], avoid) == must_not_show(SCRIPT, BY_ID["bench"], avoid) == []
    told = drawing_brief(SCRIPT, BY_ID["dot_small"], ["One giant germ", "A dish with no colonies"]).description
    assert told.startswith("A bold 2D cartoon with thick outlines. Tiny and curled up. It must not show any of these "
                           "wrong ideas: One giant germ; A dish with no colonies. The subject is alone")
    assert "bacterium" not in told and "wrong ideas" not in drawing_brief(SCRIPT, BY_ID["dot_small"]).description


def test_every_drawing_is_drawn_once_cut_out_checked_stored_and_costed(container, stage):
    service, fal, vision, run = stage
    made = service.run(run.id)
    assert list(made) == ["bench", "dot", "dish", "arrow", "dot_small", "dot_point", "dish_full"]   # leads first
    assert all(a.review == ReviewState.PASSED for a in made.values())
    assert [refs for _, refs in fal.calls] == [0, 0, 0, 0, 1, 1, 1]
    seen = [labels for _, labels in vision.seen]
    assert ["the picture to check", "reference: the drawing dish"] in seen                # dish_full is a pose of dish
    assert sum(len(labels) == 2 for labels in seen) == 2                                   # dot_small is no pose of dot
    assert sum("wrong ideas: One giant germ; A dish with no colonies." in user for user, _ in vision.seen) == 2
    storage = container.storage()
    assert made["bench"].params["cut"] is None
    assert alpha(storage.get(made["dot"].params["cut"]))[0, 0] == 0 and made["dot"].path.endswith(".png")
    state = container.run_repository().stage(run.id, StageName.VISUAL_IDENTITY)
    assert state.status == StageStatus.DONE and state.output["drawings"]["dot"] == made["dot"].id
    assert container.asset_repository().run_cost(run.id) == pytest.approx(7 * 0.045)


def test_only_the_drawings_the_first_seconds_need_are_drawn(stage):
    service, fal, _, run = stage
    made = service.run(run.id, until=4)
    assert list(made) == ["bench", "dot", "dish", "dot_point"]      # dot is drawn because dot_point is drawn from it


def test_the_same_script_reuses_its_drawings_and_a_forced_run_reuses_the_stored_pictures(stage):
    service, fal, _, run = stage
    first = service.run(run.id)
    assert service.run(run.id)["dot"].id == first["dot"].id and len(fal.calls) == 7
    again = service.run(run.id, force=True)
    assert len(fal.calls) == 7 and again["dot"].path == first["dot"].path and again["dot"].cost_usd == 0.0


def test_a_stored_drawing_that_no_longer_cuts_out_is_drawn_again(container, stage):
    service, fal, _, run = stage
    first = service.run(run.id)
    container.storage().put(first["dish"].path, picture((20, 160, 40), box=(0, 10, 12, 16)), "image/png")
    again = service.run(run.id, force=True)
    assert again["dish"].path != first["dish"].path and again["dish"].review == ReviewState.PASSED
    assert again["dot"].path == first["dot"].path and len(fal.calls) > 7


def test_a_drawing_the_judge_fails_is_drawn_again_told_what_was_wrong(container, stage):
    service, fal, vision, run = stage
    vision.fail_first = 1
    made = service.run(run.id, until=0.5)
    assert len(made) == 4 and len(fal.calls) == 5              # one of the four was drawn twice
    assert sorted(a.params["try"] for a in made.values()) == [1, 1, 1, 2]
    assert all(a.review == ReviewState.PASSED for a in made.values())
    again = next(a for a in made.values() if a.params["try"] == 2)
    assert again.params["correction"] == "Its arms are held out wide." and again.prompt.endswith(" Its arms are held out wide.")
    assert sum(prompt.endswith(" Its arms are held out wide.") for prompt, _ in fal.calls) == 1
    assert "What the judge found wrong:\n- matches the description 3/10" in vision.asked[0]
    assert container.asset_repository().run_cost(run.id) == pytest.approx(5 * 0.045 + 0.0005)


def test_when_the_corrected_draw_fails_too_the_better_try_is_used_and_the_stage_goes_on(container, stage):
    service, fal, vision, run = stage
    vision.fail_first = 99
    made = service.run(run.id)
    assert len(made) == 7 and len(fal.calls) == 14             # nothing is held back: every drawing has a picture
    assert all(a.review == ReviewState.USED_BEST and a.params["try"] == 2 for a in made.values())   # a tie: the corrected one
    state = container.run_repository().stage(run.id, StageName.VISUAL_IDENTITY)
    assert state.status == StageStatus.DONE and sorted(state.output["used_best"]) == sorted(made)
    assert len(service.run(run.id, force=True)) == 7 and len(fal.calls) == 14                      # and it is reused


def test_a_passed_picture_is_reused_though_an_older_try_with_the_same_key_failed(stage):
    service, fal, vision, run = stage
    fal.whole = True
    assert service.run(run.id, until=0.5)["dot"].review == ReviewState.FAILED
    fal.whole = False                                           # the same draws can now be cut out, and pass
    service.run(run.id, until=0.5)
    calls = len(fal.calls)
    again = service.run(run.id, until=0.5, force=True)
    assert len(fal.calls) == calls and len(again) == 4 and all(a.review == ReviewState.PASSED for a in again.values())


def test_a_picture_that_cannot_be_cut_out_fails_the_stage_and_nothing_is_drawn_from_it(container, stage):
    service, fal, vision, run = stage
    fal.whole = True
    made = service.run(run.id)
    assert made["dot"].review == ReviewState.FAILED and "dot_point" not in made
    assert len(vision.seen) == 1                                # only the backdrop reached the judge
    state = container.run_repository().stage(run.id, StageName.VISUAL_IDENTITY)
    assert state.status == StageStatus.FAILED and "dot: failed every draw" in state.error
    assert state.output["not_drawn"] == ["dot_small", "dot_point", "dish_full"]


def test_a_run_needs_an_animation_script(container, stage):
    service, _, _, run = stage
    other = container.run_repository().runs.add(Run(song_id=run.song_id, name="other", audience=GradeBand.G3_5))
    with pytest.raises(ValueError, match="no animation script"):
        service.run(other.id)
    with pytest.raises(LookupError):
        service.run("missing")
