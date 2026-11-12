import pytest

from src.schema.lesson.lesson_schema import BeatDraft, BeatType, GradeBand, KeyTerm, LessonDraft
from src.schema.song.song_analysis_schema import LyricLine, TimedWord
from src.service.lesson.lesson_prompt import jamwise_direction, user_prompt
from src.service.lesson.lesson_rules import (band_for, band_rules, format_rules, hero_limit, problems, repeats,
                                             timed_beats)

LINES = [LyricLine(index=i, text=t, start=i, end=i + 1, confidence=0.9, unclear=(i == 2))
         for i, t in enumerate(["This is a colony.", "In 1858 it grew.", "Mumble words.", "Sing it with me!"])]


def beat(first, last, type=BeatType.EXPLAIN, **fields) -> BeatDraft:
    return BeatDraft(**{**dict(first_line=first, last_line=last, type=type, concept="A colony is a group of bacteria.",
                               hint="A dish with a colony spreading across it"), **fields})


def draft(*beats, **changes) -> LessonDraft:
    base = dict(topic="Bacteria", subject="Science", summary="S.",
                key_terms=[KeyTerm(term="colony", definition="a group")], must_not_teach=[],
                beats=list(beats) or [beat(0, 1, keyword="colony", number_text="1858"), beat(2, 2, BeatType.EXAMPLE),
                                      beat(3, 3, BeatType.HERO)])
    return LessonDraft(**{**base, **changes})


def check(lesson, lines=LINES, band=GradeBand.G3_5, duration=4.0):
    return problems(lesson, lines, band, duration)


def test_every_band_and_type_has_a_rules_file():
    for band in GradeBand:
        rules = band_rules(band)
        assert rules.band == band and all(format_rules(t).type == t for t in rules.allowed_types + rules.override_types)


def test_grades_map_to_bands():
    assert [band_for(g) for g in (0, 2, 3, 5, 6, 8)] == ["k-2", "k-2", "3-5", "3-5", "6-8", "6-8"]
    with pytest.raises(ValueError):
        band_for(9)


def test_a_good_draft_has_no_problems():
    assert check(draft()) == []


def test_beats_cover_every_line_once_in_order_and_key_terms_are_listed():
    found = check(draft(beat(0, 1), beat(3, 3), key_terms=[]))
    assert found == ["key_terms must list the terms the song teaches",
                     "beats must cover lines 0-3 once each, in order, without gaps"]


def test_a_beat_teaches_one_idea_in_a_few_lines_and_seconds():
    five = [LyricLine(index=i, text="Cells split.", start=i * 4.0, end=i * 4.0 + 3.8, confidence=0.9) for i in range(5)]
    assert check(draft(beat(0, 4)), five, duration=20.0) == [
        "beat 0-4 has 5 lines; a beat teaches one idea in at most 4 lines, so split it"]
    assert check(draft(beat(0, 3), beat(4, 4)), five, duration=20.0) == [
        "beat 0-3 lasts 16.0 s; split it so each part lasts at most 12 s"]
    one = [LyricLine(index=0, text="Cells split.", start=0.0, end=14.0, confidence=0.9)]
    assert check(draft(beat(0, 0)), one, duration=14.0) == []                   # a single line may run long
    for concept in ("", "A cell splits. Then it grows.", " ".join(["word"] * 21)):
        assert check(draft(beat(0, 1, concept=concept), beat(2, 3))) == [
            "beat 0-1's concept must be one sentence of at most 20 words; split the beat if it teaches two ideas"]
    assert check(draft(beat(0, 1, hint=" "), beat(2, 3)))[0].startswith("beat 0-1 needs a hint")
    assert check(draft(beat(0, 1, BeatType.MUSIC), beat(2, 3)))[0].startswith("beat 0-1 is a music beat")


def test_a_keyword_and_a_figure_are_copied_from_the_clear_lines_of_the_beat():
    rest = beat(2, 3)
    assert check(draft(beat(0, 1, keyword="a colony"), rest)) == []
    assert check(draft(beat(0, 1, keyword="bacteria"), rest))[0].startswith(
        "beat 0-1's keyword 'bacteria' is not sung in its lines")
    assert check(draft(beat(0, 1, keyword="this is a colony"), rest)) == [
        "beat 0-1's keyword 'this is a colony' must be 1 to 3 words"]
    sung = [LyricLine(index=0, text="Met-a-MOR-pho-SIS! That is the word", start=0, end=2, confidence=0.9)]
    assert problems(draft(beat(0, 0, keyword="Met-a-MOR-pho-SIS")), sung, GradeBand.G3_5, 2.0) == []   # one word
    assert check(draft(beat(0, 1), beat(2, 3, keyword="mumble")))[0].startswith(
        "beat 2-3's keyword 'mumble' is not sung in its lines")                 # line 2 is unclear
    assert check(draft(beat(0, 1, number_text="1866"), rest))[0].startswith("beat 0-1's number_text '1866' must be")


def test_hero_beats_are_short_clear_and_capped_by_the_band():
    assert [hero_limit(b, 20) for b in GradeBand] == [12, 8, 4]
    found = check(draft(beat(0, 1), beat(2, 3, BeatType.HERO)))
    assert found == ["beat 2-3 is a hero beat with an [unclear] line; the performer sings only clear lines",
                     "2 lines are in hero beats; at most 1 are allowed for grades 3-5"]
    long = [LyricLine(index=i, text="Sing it!", start=i * 6.0, end=i * 6.0 + 5.5, confidence=0.9) for i in range(4)]
    assert check(draft(beat(0, 1, BeatType.HERO), beat(2, 3)), long, GradeBand.K_2, 24.0) == [
        "beat 0-1 is a hero beat lasting 12.0 s; lip-sync takes at most 10 s, so split it or give it another type"]


CHORUS = [LyricLine(index=i, text=t, start=i, end=i + 1, confidence=0.9, section=s)
          for i, (s, t) in enumerate([("chorus", "Sing it!"), ("verse", "Cells split."), ("chorus", "sing it!"),
                                      (None, "Cells split."), (None, "Cells split.")])]


def test_a_chorus_sung_again_is_a_recap_or_hero_beat():
    assert repeats(CHORUS) == {2: 0}                               # untagged repeats don't count
    beats = lambda kind: draft(beat(0, 1), beat(2, 2, kind), beat(3, 4))
    assert check(beats(BeatType.EXPLAIN), CHORUS, duration=5.0) == [
        "line 2 repeats line 0; make beat 2-2 a recap or hero beat"]
    assert check(beats(BeatType.RECAP), CHORUS, duration=5.0) == check(beats(BeatType.HERO), CHORUS, duration=5.0) == []
    prompt = user_prompt(CHORUS, GradeBand.G3_5, None)
    assert "2. [2.0-3.0s] [chorus] [repeat of 0] sing it!" in prompt
    assert "At most 2 lines may be in hero beats." in prompt


def test_beats_are_set_in_time_with_music_filling_the_stretches_without_lyrics():
    lines = [LyricLine(index=0, text="It starts as an egg", start=5.0, end=8.0, confidence=0.9),
             LyricLine(index=1, text="in the life-cycle song", start=9.0, end=12.0, confidence=0.9),
             LyricLine(index=2, text="Mumble egg", start=20.0, end=23.0, confidence=0.9, unclear=True)]
    texts = [(0, "It", 5.0), (0, "starts", 5.5), (0, "as", 6.0), (0, "an", 6.5), (0, "egg", 7.0),
             (1, "in", 9.0), (1, "the", 9.5), (1, "life-cycle", 10.0), (1, "song", 11.0), (2, "Mumble", 20.0), (2, "egg", 21.5)]
    words = [TimedWord(index=i, line_index=l, text=t, start=s, end=s + 0.4) for i, (l, t, s) in enumerate(texts)]
    beats = timed_beats([beat(0, 0, keyword="egg"), beat(1, 1, keyword="Life cycle"), beat(2, 2, keyword="")],
                        lines, words, duration=30.0)
    assert [(b.id, b.start, b.end, b.type, b.lines) for b in beats] == [
        ("b01", 0.0, 5.0, "music", []), ("b02", 5.0, 9.0, "explain", [0]),    # a 1 s gap joins the beat before it
        ("b03", 9.0, 12.0, "explain", [1]), ("b04", 12.0, 20.0, "music", []),  # an 8 s break is a beat of its own
        ("b05", 20.0, 23.0, "explain", [2]), ("b06", 23.0, 30.0, "music", [])]
    assert [(b.keyword, b.keyword_at) for b in beats if b.lines] == [("egg", 7.0), ("Life cycle", 10.0), (None, None)]
    stray = [TimedWord(index=0, line_index=1, text="song", start=8.0, end=8.4)]       # timed before its own line
    assert timed_beats([beat(0, 0), beat(1, 1, keyword="song")], lines[:2], stray, 12.0)[2].keyword_at == 9.0
    short = timed_beats([beat(0, 0)], [lines[0].model_copy(update={"start": 2.0})], [], duration=9.5)
    assert [(b.start, b.end, b.type) for b in short] == [(0.0, 9.5, "explain")]  # short ends join the beat


def test_the_jamwise_plan_gives_the_lesson_its_facts_terms_and_demonstrations():
    export = {"contentPackage": {
        "learningObjectives": ["Explain fusion"],
        "keyFacts": [{"fact": "F.", "kidVersion": "The Sun is a ball of gas.", "explanation": "Gravity holds it.",
                      "misconception": "It burns like a campfire."}],
        "vocabularyCards": [{"term": "corona", "kidDefinition": "The Sun's outer layer.", "nonExamples": ["sunspot"]}],
        "wowFact": {"fact": "So much energy."}},
        "creativeBrief": {"teachingMoments": [{"concept": "Fusion vs burning", "visualDemonstration": "Candle beside it"}],
                          "sectionPlans": [{"section": "chorus", "educationalFocus": "Layers named",
                                            "keyMoments": ["Identical to opening chorus"]}],
                          "numberAnchors": [{"statistic": "Four layers", "visual": "Each lights up in turn"}],
                          "visualWorld": {"chorusVisual": "The layers one by one"}, "loopVisual": "Back to the first shot",
                          "featuredTerms": [{"word": "corona"}]}}
    text = jamwise_direction(export)
    for part in ("- Learning goal: Explain fusion", "- Fact: The Sun is a ball of gas. Gravity holds it.",
                 "- Wrong idea to avoid: It burns like a campfire.",
                 "- Term corona: The Sun's outer layer. Not to be confused with: sunspot",
                 "- How to show Fusion vs burning: Candle beside it", "- Wow fact: So much energy.",
                 "- Section chorus: Layers named Key moments: Identical to opening chorus",
                 "- Number: Four layers Shown as: Each lights up in turn",
                 "- How to show the chorus: The layers one by one", "- How the video ends: Back to the first shot",
                 "- Featured terms: corona"):
        assert part in text
