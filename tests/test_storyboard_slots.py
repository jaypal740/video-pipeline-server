from src.schema.lesson.lesson_schema import LessonLine, LineTag
from src.schema.song.song_analysis_schema import LyricLine
from src.schema.storyboard.storyboard_schema import ShotKind
from src.service.storyboard.storyboard_slots import make_slots


def song(*spec):
    """spec: (start, end, tag, section, text) per line."""
    lines = [LyricLine(index=i, text=text, start=s, end=e, confidence=0.9, section=section)
             for i, (s, e, _, section, text) in enumerate(spec)]
    lesson = [LessonLine(index=i, teaches="t", show="s", tag=tag, number_text="7" if tag == LineTag.NUMBER else None)
              for i, (_, _, tag, _, _) in enumerate(spec)]
    return lines, lesson


def spans(slots):
    return [(s.start, s.end, s.lines, s.kind) for s in slots]


def test_a_long_picture_line_is_cut_to_the_band_length_on_the_beat():
    lines, lesson = song((0.0, 10.0, LineTag.ILLUSTRATE, None, "a"))
    slots = make_slots(lines, lesson, beats=[5.2], duration=10.0, shot_seconds=[3, 5])
    assert spans(slots) == [(0.0, 5.2, [0], ShotKind.ILLUSTRATE), (5.2, 10.0, [0], ShotKind.ILLUSTRATE)]


def test_sung_lines_stay_whole_up_to_ten_seconds():
    lines, lesson = song((0.0, 8.0, LineTag.FACE_TO_CAMERA, None, "a"), (8.0, 22.0, LineTag.FACE_TO_CAMERA, None, "b"))
    slots = make_slots(lines, lesson, beats=[], duration=22.0, shot_seconds=[3, 5])
    assert [(s.start, s.end) for s in slots] == [(0.0, 8.0), (8.0, 15.0), (15.0, 22.0)]
    assert all(s.kind == ShotKind.SUNG for s in slots)


def test_a_short_picture_line_shares_a_shot_with_the_next():
    lines, lesson = song((0.0, 1.5, LineTag.ILLUSTRATE, None, "a"), (1.5, 4.0, LineTag.ILLUSTRATE, None, "b"),
                         (4.0, 5.0, LineTag.NUMBER, None, "7 c"))
    slots = make_slots(lines, lesson, beats=[], duration=5.0, shot_seconds=[3, 5])
    assert spans(slots) == [(0.0, 4.0, [0, 1], ShotKind.ILLUSTRATE), (4.0, 5.0, [2], ShotKind.NUMBER)]
    assert slots[1].number_text == "7"


def test_long_intros_and_outros_get_music_shots_short_ones_join_a_line():
    lines, lesson = song((4.0, 7.0, LineTag.ILLUSTRATE, None, "a"))
    slots = make_slots(lines, lesson, beats=[3.9], duration=11.0, shot_seconds=[3, 5])
    assert spans(slots) == [(0.0, 3.9, [], ShotKind.MUSIC), (3.9, 7.0, [0], ShotKind.ILLUSTRATE),
                            (7.0, 11.0, [], ShotKind.MUSIC)]
    lines, lesson = song((1.0, 7.0, LineTag.ILLUSTRATE, None, "a"))
    assert spans(make_slots(lines, lesson, beats=[], duration=8.0, shot_seconds=[3, 5])) == [
        (0.0, 4.0, [0], ShotKind.ILLUSTRATE), (4.0, 8.0, [0], ShotKind.ILLUSTRATE)]


def test_a_chorus_sung_again_gets_shots_of_its_own():
    lines, lesson = song((0.0, 4.0, LineTag.ILLUSTRATE, "chorus", "Germs adapt"),
                         (4.0, 8.0, LineTag.ILLUSTRATE, "verse", "A drug arrives"),
                         (8.0, 12.0, LineTag.ILLUSTRATE, "chorus", "Germs adapt"))
    slots = make_slots(lines, lesson, beats=[], duration=12.0, shot_seconds=[3, 5])
    assert [s.repeat_of for s in slots] == [None, None, None] and [s.lines for s in slots] == [[0], [1], [2]]


def test_a_chorus_line_never_merges_with_a_verse_line():
    lines, lesson = song((0.0, 1.5, LineTag.ILLUSTRATE, "chorus", "Germs adapt"),
                         (1.5, 4.0, LineTag.ILLUSTRATE, "verse", "A drug arrives"),
                         (4.0, 5.5, LineTag.ILLUSTRATE, "chorus", "Germs adapt"))
    assert [s.lines for s in make_slots(lines, lesson, beats=[], duration=5.5, shot_seconds=[3, 5])] == [[0], [1], [2]]


def test_a_break_between_lines_gets_its_own_music_shot():
    lines, lesson = song((0.0, 4.0, LineTag.FACE_TO_CAMERA, None, "a"), (9.0, 12.0, LineTag.ILLUSTRATE, None, "b"),
                         (12.5, 15.0, LineTag.ILLUSTRATE, None, "c"))
    slots = make_slots(lines, lesson, beats=[], duration=15.0, shot_seconds=[3, 5])
    assert spans(slots) == [(0.0, 4.0, [0], ShotKind.SUNG), (4.0, 9.0, [], ShotKind.MUSIC),
                            (9.0, 12.5, [1], ShotKind.ILLUSTRATE), (12.5, 15.0, [2], ShotKind.ILLUSTRATE)]


def test_a_line_just_over_the_maximum_stays_one_shot():
    lines, lesson = song((0.0, 5.1, LineTag.ILLUSTRATE, None, "a"))
    assert spans(make_slots(lines, lesson, beats=[], duration=5.1, shot_seconds=[3, 5])) == [
        (0.0, 5.1, [0], ShotKind.ILLUSTRATE)]
    lines, lesson = song((0.0, 10.0, LineTag.ILLUSTRATE, None, "a"))
    assert len(make_slots(lines, lesson, beats=[], duration=10.0, shot_seconds=[3, 5])) == 2


def test_a_line_timed_at_almost_zero_length_joins_its_neighbour():
    lines, lesson = song((0.0, 4.0, LineTag.ILLUSTRATE, None, "a"), (4.0, 4.1, LineTag.FACE_TO_CAMERA, None, "oh"),
                         (4.1, 8.0, LineTag.ILLUSTRATE, None, "b"))
    assert spans(make_slots(lines, lesson, beats=[], duration=8.0, shot_seconds=[3, 5])) == [
        (0.0, 4.1, [0, 1], ShotKind.ILLUSTRATE), (4.1, 8.0, [2], ShotKind.ILLUSTRATE)]
    lines, lesson = song((0.0, 0.1, LineTag.NUMBER, None, "7"), (0.1, 4.0, LineTag.ILLUSTRATE, None, "b"))
    assert spans(make_slots(lines, lesson, beats=[], duration=4.0, shot_seconds=[3, 5])) == [
        (0.0, 4.0, [0, 1], ShotKind.ILLUSTRATE)]


def test_the_intro_is_measured_after_snapping_to_the_beat():
    lines, lesson = song((3.0, 5.5, LineTag.ILLUSTRATE, None, "a"))
    slots = make_slots(lines, lesson, beats=[2.8], duration=5.5, shot_seconds=[3, 5])
    assert spans(slots) == [(0.0, 5.5, [0], ShotKind.ILLUSTRATE)]       # a 2.8 s intro is too short for a shot
    lines, lesson = song((3.3, 5.8, LineTag.ILLUSTRATE, None, "a"))
    slots = make_slots(lines, lesson, beats=[3.2], duration=5.8, shot_seconds=[3, 5])
    assert spans(slots)[0] == (0.0, 3.2, [], ShotKind.MUSIC)             # snapped back to 3.2 s, long enough
