import pytest

from src.schema.lesson.lesson_schema import GradeBand, KeyTerm, LessonDraft, LessonLine, LineTag, VideoType
from src.schema.song.song_analysis_schema import LyricLine
from src.service.lesson.lesson_prompt import user_prompt
from src.service.lesson.lesson_rules import (allowed_types, band_for, band_rules, face_limit, format_rules, problems,
                                             repeats)

LINES = [LyricLine(index=i, text=t, start=i, end=i + 1, confidence=0.9, unclear=(i == 2))
         for i, t in enumerate(["This is a colony.", "In 1858 it grew.", "Mumble words.", "Sing it with me!"])]


def draft(**changes) -> LessonDraft:
    base = dict(topic="Bacteria", subject="Science", summary="S.", video_type=VideoType.CONCEPT, type_reason="r",
                key_terms=[KeyTerm(term="colony", definition="a group")], must_not_teach=[],
                lines=[LessonLine(index=0, teaches="t", show="s", tag=LineTag.ILLUSTRATE),
                       LessonLine(index=1, teaches="t", show="s", tag=LineTag.NUMBER, number_text="1858"),
                       LessonLine(index=2, teaches="t", show="s", tag=LineTag.UNCLEAR),
                       LessonLine(index=3, teaches="t", show="s", tag=LineTag.FACE_TO_CAMERA)])
    return LessonDraft(**{**base, **changes})


def test_every_band_and_type_has_a_rules_file():
    for band in GradeBand:
        rules = band_rules(band)
        assert rules.band == band and all(format_rules(t).type == t for t in rules.allowed_types + rules.override_types)


def test_grades_map_to_bands():
    assert [band_for(g) for g in (0, 2, 3, 5, 6, 8)] == ["k-2", "k-2", "3-5", "3-5", "6-8", "6-8"]
    with pytest.raises(ValueError):
        band_for(9)


def test_types_come_from_the_band_or_a_permitted_request():
    assert VideoType.CONCEPT not in allowed_types(GradeBand.K_2, None)
    assert allowed_types(GradeBand.G6_8, VideoType.PERSONALISED_HERO) == [VideoType.PERSONALISED_HERO]
    with pytest.raises(ValueError, match="not allowed"):
        allowed_types(GradeBand.K_2, VideoType.JOURNEY)


def test_a_good_draft_has_no_problems():
    assert problems(draft(), LINES, GradeBand.G3_5, allowed_types(GradeBand.G3_5, None)) == []


def test_drafts_breaking_the_rules_are_caught():
    types = allowed_types(GradeBand.K_2, None)
    found = problems(draft(video_type=VideoType.CONCEPT, key_terms=[], lines=draft().lines[:3]), LINES,
                     GradeBand.K_2, types)
    assert any("indexes" in p for p in found) and any("video_type" in p for p in found)
    assert any("key_terms" in p for p in found)


def test_line_tags_are_checked_against_intake_and_the_text():
    lines = draft().lines
    lines[0].tag = LineTag.UNCLEAR
    lines[1].number_text = "1866"
    found = problems(draft(lines=lines), LINES, GradeBand.G3_5, allowed_types(GradeBand.G3_5, None))
    assert any("line 0" in p and "unclear" in p for p in found) and any("line 1" in p for p in found)


def test_face_to_camera_lines_are_capped_by_band_and_type():
    assert face_limit(GradeBand.G6_8, VideoType.CONCEPT, 20) == 4 and face_limit(GradeBand.K_2, VideoType.PROCESS, 20) == 4
    lines = draft().lines
    lines[0].tag = LineTag.FACE_TO_CAMERA
    found = problems(draft(lines=lines), LINES, GradeBand.G6_8, allowed_types(GradeBand.G6_8, None))
    assert any("face_to_camera" in p for p in found)


def test_the_prompt_says_when_to_pick_each_allowed_type():
    prompt = user_prompt(LINES, GradeBand.K_2, allowed_types(GradeBand.K_2, None), None)
    assert all(format_rules(t).pick_when in prompt for t in allowed_types(GradeBand.K_2, None))
    assert format_rules(VideoType.JOURNEY).pick_when not in prompt


CHORUS = [LyricLine(index=i, text=t, start=i, end=i + 1, confidence=0.9, section=s)
          for i, (s, t) in enumerate([("chorus", "Sing it!"), ("verse", "Cells split."), ("chorus", "sing it!"),
                                      (None, "Cells split."), (None, "Cells split.")])]


def test_a_chorus_sung_again_must_look_the_same():
    assert repeats(CHORUS) == {2: 0}                               # untagged repeats don't count
    lines = [LessonLine(index=i, teaches="t", show="s", tag=LineTag.ILLUSTRATE) for i in range(5)]
    lesson = draft(lines=lines)
    assert problems(lesson, CHORUS, GradeBand.G3_5, allowed_types(GradeBand.G3_5, None)) == []
    lines[2].show = "a different scene"
    found = problems(draft(lines=lines), CHORUS, GradeBand.G3_5, allowed_types(GradeBand.G3_5, None))
    assert found == ["line 2 repeats line 0; give it the same teaches, show and tag"]
    assert "2. [2.0-3.0s] [chorus] [repeat of 0] sing it!" in user_prompt(CHORUS, GradeBand.G3_5, [VideoType.CONCEPT], None)
