import pytest

from src.schema.lesson.lesson_schema import GradeBand, VideoType
from src.schema.storyboard.storyboard_schema import (Angle, Camera, CameraMove, CastKind, CastMember, Continuity,
                                                     Direction, Motif, Motion, Place, Position, Scale,
                                                     ScreenDirection, ShotDraft, ShotKind, ShotSize, Slot, Speed,
                                                     StoryboardDraft, Transition)
from src.service.lesson.lesson_rules import band_rules, format_rules
from src.service.storyboard.storyboard_rules import (assemble, fix, header_problems, host_for, host_place_for,
                                                     shot_problems)

BAND, FMT, HOST = band_rules(GradeBand.G3_5), format_rules(VideoType.JOURNEY), host_for(GradeBand.G3_5)
STUDIO = host_place_for(GradeBand.G3_5)


def slots(*kinds, lines=None):
    return [Slot(id=f"s{i + 1:02d}", start=i * 4, end=i * 4 + 4, lines=(lines or {}).get(i, [i]), kind=k,
                 number_text="1858" if k == ShotKind.NUMBER else None) for i, k in enumerate(kinds)]


def shot(slot, place="ship_deck", size=ShotSize.MEDIUM, **changes):
    base = dict(slot=slot, place=place, subjects=["ship"], size=size, angle=Angle.EYE, position=Position.CENTRE,
                start_state="The ship rests on calm grey water", action="The ship lowers a thick cable",
                end_state="The cable sinks below the surface", camera=Camera(move=CameraMove.STATIC),
                motion=Motion.MEDIUM, screen_direction=ScreenDirection.NONE, transition_in=Transition.CUT)
    return ShotDraft(**{**base, **changes})


def board(*shots, places=None, cast=None):
    return StoryboardDraft(
        shape="From the ship to the seabed", motifs=[Motif(thing="cable", rule="always a dark rope")],
        cast=cast or [CastMember(id="ship", kind=CastKind.THING, look="A wooden steamship with two masts",
                                 size="about 90 metres long", scale=Scale.LANDSCAPE)],
        places=places or [Place(id="ship_deck", look="The deck of a wooden steamship at sea", lighting="grey daylight",
                                scale=Scale.HUMAN),
                          Place(id="seabed", look="A dark sandy ocean floor", lighting="dim blue light",
                                scale=Scale.OBJECT)],
        shots=list(shots))


def test_a_good_board_has_no_problems():
    draft = board(shot("s01", size=ShotSize.WIDE), shot("s02"), shot("s03", place="seabed", size=ShotSize.MEDIUM_WIDE))
    s = slots(*[ShotKind.ILLUSTRATE] * 3)
    assert header_problems(draft, BAND, s) == [] and shot_problems(draft, s, BAND, FMT, HOST, STUDIO) == {}


def test_header_rules():
    cast = [CastMember(id="host", kind=CastKind.PERSON, look="A man", size="tall", scale=Scale.HUMAN),
            CastMember(id="germ", kind=CastKind.THING, look="A smiling round germ with big eyes", size="tiny",
                       scale=Scale.MICROSCOPIC)]
    places = [Place(id=f"p{i}", look="A room", lighting="soft", scale=Scale.HUMAN) for i in range(6)]
    s = slots(*[ShotKind.ILLUSTRATE] * 6)
    six = board(*[shot(f"s{i + 1:02d}", place=f"p{i}") for i in range(6)], cast=cast, places=places)
    found = header_problems(six, BAND, s)
    assert any("'host' is given" in p for p in found) and any("6 places used" in p for p in found)
    assert not any("places used" in p for p in header_problems(board(shot("s01", place="p0"), places=places), BAND, s))
    ghost = board(*[shot(f"s{i + 1:02d}", place=f"p{i}") for i in range(5)], shot("s99", place="p5"), places=places)
    assert not any("places used" in p for p in header_problems(ghost, BAND, s))        # s99 is not a slot
    assert any("germ look" in p and "only people" in p for p in found)


def test_code_fixes_camera_sung_and_number_framing():
    s = slots(ShotKind.SUNG, ShotKind.NUMBER, ShotKind.ILLUSTRATE)
    sung = shot("s01", size=ShotSize.WIDE, subjects=["sailor", "ship"],
                camera=Camera(move=CameraMove.PAN, direction=Direction.LEFT))
    number = shot("s02")
    fast = shot("s03", motion=Motion.HIGH, camera=Camera(move=CameraMove.PUSH_IN, direction=Direction.LEFT,
                                                         speed=Speed.FAST))
    for draft_shot, slot in zip((sung, number, fast), s):
        fix(draft_shot, slot, BAND, people={"sailor"})
    assert sung.subjects == ["host", "ship"] and sung.place == "host_place" and sung.size == ShotSize.MEDIUM_CLOSE and sung.camera.move == CameraMove.STATIC
    assert number.position == Position.RIGHT_THIRD
    assert fast.camera.direction is None and fast.camera.speed == Speed.SLOW
    k2 = shot("s03", camera=Camera(move=CameraMove.ORBIT, direction=Direction.LEFT))
    fix(k2, s[2], band_rules(GradeBand.K_2), people=set())
    assert k2.camera.move == CameraMove.STATIC


def test_film_grammar_is_checked():
    s = slots(*[ShotKind.ILLUSTRATE] * 4, lines={1: [1], 2: [1]})
    draft = board(shot("s01", size=ShotSize.CLOSE),                                          # no establishing shot
                  shot("s02", size=ShotSize.WIDE, screen_direction=ScreenDirection.LEFT_TO_RIGHT),
                  shot("s03", place="seabed", size=ShotSize.WIDE),                           # line 1 in two places
                  shot("s04", place="seabed", size=ShotSize.EXTREME_CLOSE))                  # size jumps 5
    found = shot_problems(draft, s, BAND, FMT, HOST, STUDIO)
    assert "wide or medium_wide" in found["s01"][0]
    assert any("two places" in p for p in found["s03"]) and any("jumps 5 steps" in p for p in found["s04"])
    flip = board(shot("s01", size=ShotSize.WIDE, screen_direction=ScreenDirection.LEFT_TO_RIGHT),
                 shot("s02", screen_direction=ScreenDirection.RIGHT_TO_LEFT))
    assert "flips" in shot_problems(flip, slots(ShotKind.ILLUSTRATE, ShotKind.ILLUSTRATE), BAND, FMT, HOST, STUDIO)["s02"][0]


def test_a_big_scale_jump_in_a_concept_video_must_be_bridged():
    places = [Place(id="ship_deck", look="A deck", lighting="day", scale=Scale.HUMAN),
              Place(id="seabed", look="A cell membrane", lighting="dim", scale=Scale.MICROSCOPIC)]
    s, fmt = slots(ShotKind.ILLUSTRATE, ShotKind.ILLUSTRATE), format_rules(VideoType.CONCEPT)
    cut = board(shot("s01", size=ShotSize.WIDE), shot("s02", place="seabed", size=ShotSize.WIDE, subjects=[]),
                places=places)
    assert "scale jumps 2 steps" in shot_problems(cut, s, BAND, fmt, HOST, STUDIO)["s02"][0]
    dissolve = board(shot("s01", size=ShotSize.WIDE), shot("s02", place="seabed", size=ShotSize.WIDE, subjects=[],
                                                          transition_in=Transition.DISSOLVE), places=places)
    assert shot_problems(dissolve, s, BAND, fmt, HOST, STUDIO) == {}
    zoom = board(shot("s01", size=ShotSize.WIDE, camera=Camera(move=CameraMove.PUSH_IN)),
                 shot("s02", place="seabed", size=ShotSize.WIDE, subjects=[]), places=places)
    assert shot_problems(zoom, s, BAND, fmt, HOST, STUDIO) == {}
    assert shot_problems(cut, s, BAND, FMT, HOST, STUDIO) == {}                 # journeys have no scale rule


def test_shot_words_unknown_ids_and_missing_shots_are_caught():
    draft = board(shot("s01", size=ShotSize.WIDE, subjects=["ship", "whale"],
                       action="The ship passes a sign with 1858 painted on it",
                       camera=Camera(move=CameraMove.PAN)))
    found = shot_problems(draft, slots(ShotKind.ILLUSTRATE, ShotKind.ILLUSTRATE), BAND, FMT, HOST, STUDIO)
    text = " ".join(found["s01"])
    assert "['whale']" in text and "digits" in text and "needs a direction" in text
    assert found["s02"] == ["this slot has no shot"]
    faces = board(shot("s01", size=ShotSize.WIDE, end_state="The ship has a happy face and a smile"))
    assert "only people" in shot_problems(faces, slots(ShotKind.ILLUSTRATE), BAND, FMT, HOST, STUDIO)["s01"][0]


def test_assembly_copies_repeats_and_works_out_continuity():
    s = slots(*[ShotKind.ILLUSTRATE] * 3)
    s[2].repeat_of = "s01"
    draft = board(shot("s01", size=ShotSize.WIDE), shot("s02", place="seabed", size=ShotSize.WIDE))
    shots, flags = assemble(draft, s, HOST)
    assert [x.continuity for x in shots] == [Continuity.NEW_SCENE, Continuity.SAME_SUBJECT, Continuity.SAME_SUBJECT]
    assert shots[2].action == shots[0].action and shots[2].repeat_of == "s01" and flags == []


def test_the_host_appears_only_in_sung_shots_unless_the_video_is_narrated():
    draft = board(shot("s01", size=ShotSize.WIDE, subjects=["host", "ship"]))
    s = slots(ShotKind.ILLUSTRATE)
    assert "only in sung shots" in shot_problems(draft, s, BAND, FMT, HOST, STUDIO)["s01"][0]
    assert shot_problems(draft, s, BAND, format_rules(VideoType.PERSONALISED_HERO), HOST, STUDIO) == {}


def test_a_repeat_is_never_blamed_and_the_shot_after_it_is_checked_against_it():
    s = slots(*[ShotKind.ILLUSTRATE] * 5)
    s[3].repeat_of = "s02"                                            # chorus: a medium shot copied mid-place
    draft = board(shot("s01", size=ShotSize.WIDE), shot("s02", size=ShotSize.MEDIUM),
                  shot("s03", place="seabed", size=ShotSize.WIDE), shot("s05", size=ShotSize.MEDIUM))
    assert shot_problems(draft, s, BAND, FMT, HOST, STUDIO) == {}             # s04 copies s02 and is not re-checked
    draft.shots[-1].size = ShotSize.EXTREME_CLOSE                     # s05 jumps 3 steps from the repeat's medium
    assert "jumps 3 steps" in shot_problems(draft, s, BAND, FMT, HOST, STUDIO)["s05"][0]


def test_a_place_opened_by_a_sung_shot_still_needs_a_wide_picture_shot():
    s = slots(ShotKind.SUNG, ShotKind.ILLUSTRATE, ShotKind.ILLUSTRATE)
    draft = board(shot("s01", subjects=["host"], size=ShotSize.MEDIUM_CLOSE), shot("s02", size=ShotSize.CLOSE),
                  shot("s03", size=ShotSize.CLOSE))
    found = shot_problems(draft, s, BAND, format_rules(VideoType.PERSONALISED_HERO), HOST, STUDIO)
    assert "first picture shot" in found["s02"][0] and "s03" not in found


def test_everyday_words_are_not_mistaken_for_faces_or_text():
    fine = board(shot("s01", size=ShotSize.WIDE, start_state="An eye-level view of the right face of the prism",
                      action="Nothing but a notebook falls", end_state="The ship cannot be seen past the cape"))
    assert shot_problems(fine, slots(ShotKind.ILLUSTRATE), BAND, FMT, HOST, STUDIO) == {}
    follow = board(shot("s01", size=ShotSize.WIDE, camera=Camera(move=CameraMove.FOLLOW, direction=Direction.UP)))
    assert "needs a direction" in shot_problems(follow, slots(ShotKind.ILLUSTRATE), BAND, FMT, HOST, STUDIO)["s01"][0]


def test_assembling_a_board_with_a_missing_shot_is_a_clear_error():
    s = slots(ShotKind.ILLUSTRATE, ShotKind.ILLUSTRATE)
    with pytest.raises(ValueError, match="s02 has no shot"):
        assemble(board(shot("s01", size=ShotSize.WIDE)), s, HOST)


def test_a_subject_must_fit_the_scale_of_its_place():
    cast = [CastMember(id="cable", kind=CastKind.THING, look="A thin black cable", size="hose thick", scale=Scale.OBJECT)]
    places = [Place(id="globe", look="Earth seen from space", lighting="sunlit", scale=Scale.PLANET),
              Place(id="ship_deck", look="The deck of a steamship", lighting="daylight", scale=Scale.HUMAN)]
    draft = board(shot("s01", place="globe", subjects=["cable"], size=ShotSize.WIDE),
                  shot("s02", place="ship_deck", subjects=["cable"], size=ShotSize.WIDE),
                  shot("s03", place="globe", subjects=[], size=ShotSize.WIDE), cast=cast, places=places)
    found = shot_problems(draft, slots(*[ShotKind.ILLUSTRATE] * 3), BAND, FMT, HOST, STUDIO)
    assert "far too small or large" in found["s01"][0] and "s02" not in found and "s03" not in found
