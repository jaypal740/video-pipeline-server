import pytest

from src.schema.lesson.lesson_schema import (GradeBand, KeyTerm, LessonDraft, LessonLine, LessonSequence, LineTag,
                                             VideoType)
from src.schema.song.song_analysis_schema import LyricLine
from src.service.lesson.lesson_prompt import jamwise_direction, user_prompt
from src.service.lesson.lesson_rules import (allowed_types, band_for, band_rules, face_limit, format_rules, problems,
                                             repeats, sequence_problems)

LINES = [LyricLine(index=i, text=t, start=i, end=i + 1, confidence=0.9, unclear=(i == 2))
         for i, t in enumerate(["This is a colony.", "In 1858 it grew.", "Mumble words.", "Sing it with me!"])]


def draft(**changes) -> LessonDraft:
    base = dict(topic="Bacteria", subject="Science", summary="S.", video_type=VideoType.CONCEPT, type_reason="r",
                key_terms=[KeyTerm(term="colony", definition="a group")], must_not_teach=[],
                lines=[LessonLine(index=0, teaches="t", show="s", tag=LineTag.ILLUSTRATE),
                       LessonLine(index=1, teaches="t", show="s", tag=LineTag.NUMBER, number_text="1858"),
                       LessonLine(index=2, teaches="t", show="s", tag=LineTag.UNCLEAR),
                       LessonLine(index=3, teaches="t", show="s", tag=LineTag.FACE_TO_CAMERA)],
                sequences=[LessonSequence(first_line=0, last_line=3, topic="t", place="p", change="c",
                                          fact="A colony grows.", steps=["A colony spreads"])])
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


def test_a_chorus_sung_again_keeps_its_tag_but_shows_something_new():
    assert repeats(CHORUS) == {2: 0}                               # untagged repeats don't count
    lines = [LessonLine(index=i, teaches="t", show="s", tag=LineTag.ILLUSTRATE) for i in range(5)]
    whole = [LessonSequence(first_line=0, last_line=4, topic="t", place="p", change="c", fact="f", steps=["a"])]
    lines[2].show = "a different scene"
    lesson = draft(lines=lines, sequences=whole)
    assert problems(lesson, CHORUS, GradeBand.G3_5, allowed_types(GradeBand.G3_5, None)) == []
    lines[2].tag = LineTag.FACE_TO_CAMERA
    found = problems(draft(lines=lines, sequences=whole), CHORUS, GradeBand.G3_5, allowed_types(GradeBand.G3_5, None))
    assert found == ["line 2 repeats line 0; give it the same tag"]
    assert "2. [2.0-3.0s] [chorus] [repeat of 0] sing it!" in user_prompt(CHORUS, GradeBand.G3_5, [VideoType.CONCEPT], None)


def test_a_sequence_says_what_it_shows_in_steps_that_fit_its_length():
    long = [LyricLine(index=i, text="Cells split.", start=i * 4.0, end=i * 4.0 + 3.8, confidence=0.9) for i in range(3)]
    lines = [LessonLine(index=i, teaches="t", show="s", tag=LineTag.ILLUSTRATE) for i in range(3)]
    sequence = lambda **c: [LessonSequence(**{**dict(first_line=0, last_line=2, topic="t", place="p", change="c",
                                                     fact="A cell splits in two.", steps=["a", "b"]), **c})]
    check = lambda seqs: sequence_problems(draft(lines=lines, sequences=seqs), long, [])
    assert check(sequence()) == []
    assert check(sequence(fact=" ")) == ["sequence 0-2 needs a fact: what a viewer must understand from it"]
    for steps in (["a"], list("abcde")):
        assert check(sequence(steps=steps)) == [f"sequence 0-2 has {len(steps)} steps; give 2 to 4 visible steps, "
                                                "each at least 2.5 s of its 11.8 s"]
    short = sequence(last_line=0, steps=["a", "b"])                  # 3.8 s: room for one step only
    assert sequence_problems(draft(lines=lines[:1], sequences=short), long[:1], []) == [
        "sequence 0-0 has 2 steps; give 1 to 1 visible steps, each at least 2.5 s of its 3.8 s"]


def test_the_jamwise_plan_gives_the_lesson_its_facts_terms_and_demonstrations():
    export = {"contentPackage": {
        "learningObjectives": ["Explain fusion"],
        "keyFacts": [{"fact": "F.", "kidVersion": "The Sun is a ball of gas.", "explanation": "Gravity holds it.",
                      "misconception": "It burns like a campfire."}],
        "vocabularyCards": [{"term": "corona", "kidDefinition": "The Sun's outer layer.", "nonExamples": ["sunspot"]}],
        "wowFact": {"fact": "So much energy."}},
        "creativeBrief": {"teachingMoments": [{"concept": "Fusion vs burning", "visualDemonstration": "Candle beside it"}],
                          "featuredTerms": [{"word": "corona"}]}}
    text = jamwise_direction(export)
    for part in ("- Learning goal: Explain fusion", "- Fact: The Sun is a ball of gas. Gravity holds it.",
                 "- Wrong idea to avoid: It burns like a campfire.",
                 "- Term corona: The Sun's outer layer. Not to be confused with: sunspot",
                 "- How to show Fusion vs burning: Candle beside it", "- Wow fact: So much energy.",
                 "- Featured terms: corona"):
        assert part in text


def test_a_shown_term_must_be_a_defined_key_term():
    lines = [LessonLine(index=i, teaches="t", show="s", tag=LineTag.ILLUSTRATE) for i in range(4)]
    seq = lambda shown: [LessonSequence(first_line=0, last_line=3, topic="t", place="p", change="c", fact="f",
                                        steps=["a"], shown_terms=shown)]
    plain = [LyricLine(index=i, text="Cells split.", start=i, end=i + 1, confidence=0.9) for i in range(4)]
    assert sequence_problems(draft(lines=lines, sequences=seq(["Colony"])), plain, []) == []
    assert sequence_problems(draft(lines=lines, sequences=seq(["corona"])), plain, []) == [
        "sequence 0-3 shows terms ['corona'] that are not in key_terms; define them there or leave them out"]


def test_steps_fit_the_band_shot_length_and_places_fit_the_band_limit():
    long = [LyricLine(index=i, text="Cells split.", start=i * 4.0, end=i * 4.0 + 3.8, confidence=0.9) for i in range(3)]
    lines = [LessonLine(index=i, teaches="t", show="s", tag=LineTag.ILLUSTRATE) for i in range(3)]
    seq = lambda **c: LessonSequence(**{**dict(first_line=0, last_line=2, topic="t", place="p", change="c", fact="f",
                                              steps=["a", "b"]), **c})
    assert sequence_problems(draft(lines=lines, sequences=[seq()]), long, [], GradeBand.G6_8) == []
    assert sequence_problems(draft(lines=lines, sequences=[seq(steps=list("abcd"))]), long, [], GradeBand.K_2) == [
        "sequence 0-2 has 4 steps; give 1 to 1 visible steps, each at least 6.0 s of its 11.8 s"]   # K-2 shots are 6 s
    three = [seq(last_line=0, place="a", steps=["x"]), seq(first_line=1, last_line=1, place="b", steps=["x"]),
             seq(first_line=2, place="c", steps=["x"])]
    found = sequence_problems(draft(lines=lines, sequences=three), long, [], GradeBand.K_2)
    assert any(p.startswith("the sequences use 3 places ['a', 'b', 'c']; at most 2 for grades k-2") for p in found)
    assert "Sequences use at most 2 places between them; each step lasts at least 6.0 s." in user_prompt(
        long, GradeBand.K_2, [VideoType.CONCEPT], None)


def test_a_fact_may_rest_on_a_figure_only_when_a_line_is_a_number_line():
    long = [LyricLine(index=i, text="In 1858 it grew.", start=i * 4.0, end=i * 4.0 + 3.8, confidence=0.9) for i in range(3)]
    plain = [LessonLine(index=i, teaches="t", show="s", tag=LineTag.ILLUSTRATE) for i in range(3)]
    numbered = [plain[0], LessonLine(index=1, teaches="t", show="s", tag=LineTag.NUMBER, number_text="1858"), plain[2]]
    seq = [LessonSequence(first_line=0, last_line=2, topic="t", place="p", change="c", steps=["a", "b"],
                          fact="The colony was first seen in 1858.")]
    assert sequence_problems(draft(lines=numbered, sequences=seq), long, []) == []
    assert sequence_problems(draft(lines=plain, sequences=seq), long, [])[0].startswith(
        "sequence 0-2's fact leans on a figure, but none of its lines is a number line")
