from src.schema.storyboard.storyboard_schema import Storyboard
from src.service.lesson.lesson_rules import band_rules
from src.service.sheet.sheet_prompt import Style
from src.service.video.clip_prompt import PROMPT_WORDS, STYLE_LINE, camera_words, clip_brief, clip_prompt, host_look_for
from tests.test_keyframe_service import SHOTS, shot
from tests.test_sheet_service import BOARD


def board(*shots):
    return Storyboard(**{**BOARD, "shots": list(shots)})


STYLE = Style.of(band_rules("3-5"))


def test_a_clip_prompt_is_short_positive_and_names_the_camera_in_spatial_words():
    moving = shot("s05", "illustrate", "sea", ["ship"], "wide")
    moving.update(action="The ship turns and sails toward the horizon", end_state="The ship is a small shape far away",
                  camera={"move": "pan", "direction": "right", "speed": "slow"}, motion="medium",
                  screen_direction="left_to_right")
    b = board(*SHOTS, moving)
    cast, place = {m.id: m for m in b.cast}, b.places[0]
    for s in b.shots:
        text = clip_prompt(STYLE, s, place, [cast[x] for x in s.subjects], host_look_for([cast[x] for x in s.subjects]))
        body = text.removeprefix(f"{STYLE.name} animation. ")
        words = text.removesuffix(f" {STYLE_LINE}").split()         # the style line is fixed, not the shot's words
        assert len(words) <= PROMPT_WORDS and '"' not in text and not any(c.isdigit() for c in body)
        assert text.endswith(STYLE_LINE) == (s.kind != "sung")
    text = clip_prompt(STYLE, b.shots[-1], place, [cast["ship"]], None)
    assert text.startswith("A bold 2D cartoon animation. The ship turns and sails toward the horizon. By the last frame, the ship")
    assert "The camera pans to the right, slowly, steady, clear movement." in text
    assert "moving left to right across the frame" in text
    assert camera_words(b.shots[0]) == "the camera holds its framing"            # static


def test_a_sung_shot_asks_for_a_calm_closed_mouth_host_not_the_action():
    b = board(*SHOTS)
    cast, place = {m.id: m for m in b.cast}, b.places[0]
    sung = next(s for s in b.shots if s.kind == "sung")
    text = clip_prompt(STYLE, sung, place, [cast["host"]], host_look_for([cast["host"]]))
    assert "A man faces the camera with the lips together" in text and "sails" not in text
    assert "sing" not in text.lower() and "mouth" not in text.lower()


def test_the_clip_brief_judges_three_frames_against_the_three_moments():
    b = board(*SHOTS)
    cast, place = {m.id: m for m in b.cast}, b.places[0]
    brief = clip_brief(b.shots[0], place, [cast["ship"]])
    assert "first, middle and last frames of one short clip" in brief.description
    assert "Small camera drift is never a fault" in brief.description
    assert brief.moment == ("Opening: The ship sits on the grey sea with its sails full Action: The ship sails right "
                            "Ending: The ship is near the centre")
    assert len(brief.fatal) == 2 and all(f in brief.rules for f in brief.fatal)
