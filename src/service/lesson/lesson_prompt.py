from src.schema.lesson.lesson_schema import GradeBand
from src.schema.song.song_analysis_schema import LyricLine
from src.service.lesson.lesson_rules import (HERO_MAX_SECONDS, MAX_BEAT_LINES, MAX_BEAT_SECONDS, band_rules, hero_limit,
                                             repeats)

# Bumped whenever the prompt changes, so every run is planned again with the new prompt.
PROMPT_VERSION = "lesson.v9"

SYSTEM = f"""You plan short educational music videos for school classes. The song already exists; you never \
change its words or order. The video is explanatory motion graphics, not a story: flat drawings appear, move \
and swap in time with the song, and one key word pops on screen as it is sung. You decide what each part of \
the song teaches.

Split the lyrics into teaching beats: runs of 1 to {MAX_BEAT_LINES} consecutive lines that together teach one \
idea, covering every line once, in order. For each beat give:
- first_line and last_line.
- type: explain (a new idea), example (an instance of an idea already explained), recap (an idea shown \
again, such as a chorus coming back) or hero (the performer is seen singing these lines to the viewer \
while the beat's picture is shown; use it for hooks and choruses, within the limit you are given).
- concept: the one idea the beat teaches, in one plain sentence. If it needs "and", split the beat.
- keyword: the one word or short phrase a child should remember from this beat, copied exactly as it is sung \
in the beat's lines; our code shows it on screen at the moment it is sung. Null when the lines teach no term.
- number_text: when the beat's point is a figure (a year, an amount), the figure exactly as it is written in \
its lines; our code draws it. Otherwise null.
- avoid: the wrong idea this beat's picture must not suggest, or an empty string.
- hint: what to draw and what happens to it, step by step, in one to three sentences (the seed is swapped \
for a sprout). Where the song's plan says how to show the concept, keep every step it names: how many times \
a thing happens, what gets bigger, what is left behind, what leads back to the start. Real things as they \
really look; no written words, our code adds all text. Name only things the lesson is about: never add a \
thing to stand for time passing, the weather or a mood (a sun crossing the sky, a clock, a moon), and never \
two forms of one creature side by side, which would look like two animals.

A beat lasts at most {MAX_BEAT_SECONDS:.0f} s unless it is a single line, and a hero beat at most \
{HERO_MAX_SECONDS:.0f} s. Lines marked [unclear] may not be what is sung: never take a keyword or figure from \
them and never make them hero. A line marked [repeat of N] is the chorus again: make it a recap or hero beat \
with the same concept, keyword and hint as the beat of line N, so the same picture comes back.

Also give the topic, the school subject, a two-sentence summary, the key terms with child-friendly \
definitions, and must_not_teach: wrong ideas the video must not teach. Be factually accurate; if a line is \
wrong or misleading, say so in must_not_teach and in that beat's avoid rather than illustrating the error."""


def jamwise_direction(export: dict) -> str:
    """What JamWise planned the song to teach, as direction for the lesson: its goals, facts, vocabulary and
    how it meant to demonstrate each concept. Its characters' looks are left out: our pipeline designs them."""
    brief, package = export.get("creativeBrief") or {}, export.get("contentPackage") or {}
    parts = ["The song was written from this plan. Follow its topics, terms and numbers, but check every fact: "
             "where the plan is wrong, teach what is true."]
    parts += [f"- Learning goal: {g}" for g in package.get("learningObjectives", [])]
    parts += [f"- Scene {s.get('id')}: {s.get('title')}: teaches {s.get('teaches')}" for s in brief.get("scenes", [])]
    parts += [f"- Section {s.get('section')}: {s.get('educationalFocus')}"
              + (f" Key moments: {'; '.join(s['keyMoments'])}" if s.get("keyMoments") else "")
              for s in brief.get("sectionPlans", [])]
    parts += [f"- Number: {a.get('statistic')}" + (f" Shown as: {a['visual']}" if a.get("visual") else "")
              for a in brief.get("numberAnchors", [])]
    parts += [f"- Fact: {f.get('kidVersion') or f.get('fact')} {f.get('explanation') or ''}".rstrip()
              for f in package.get("keyFacts", [])]
    parts += [f"- Wrong idea to avoid: {f.get('misconception')}" for f in package.get("keyFacts", [])
              if f.get("misconception")]
    parts += [f"- Term {c.get('term')}: {c.get('kidDefinition')}"
              + (f" Not to be confused with: {'; '.join(c['nonExamples'])}" if c.get("nonExamples") else "")
              for c in package.get("vocabularyCards", []) if c.get("term")]
    parts += [f"- How to show {m.get('concept')}: {m.get('visualDemonstration')}"
              for m in brief.get("teachingMoments", []) if m.get("concept")]
    chorus = (brief.get("visualWorld") or {}).get("chorusVisual")
    parts += [f"- How to show the chorus: {chorus}"] if chorus else []
    parts += [f"- How the video ends: {brief['loopVisual']}"] if brief.get("loopVisual") else []
    wow = package.get("wowFact") or {}
    if wow.get("fact"):
        parts.append(f"- Wow fact: {wow['fact']}")
    terms = featured_terms(export)
    if terms:
        parts.append(f"- Featured terms: {', '.join(terms)}")
    return "\n".join(parts)


def featured_terms(export: dict) -> list[str]:
    return [t["word"] for t in (export.get("creativeBrief") or {}).get("featuredTerms", []) if t.get("word")]


def user_prompt(lines: list[LyricLine], band: GradeBand, brief: str | None, direction: str | None = None) -> str:
    rules = band_rules(band)
    again = repeats(lines)
    lyric = "\n".join(f"{l.index}. [{l.start:.1f}-{l.end:.1f}s]{f' [{l.section}]' if l.section else ''}"
                      f"{f' [repeat of {again[l.index]}]' if l.index in again else ''}"
                      f"{' [unclear]' if l.unclear else ''} {l.text}"
                      for l in lines)
    return (f"Audience: grades {band} (grade 0 is kindergarten).\n"
            f"On-screen text allowed: {rules.on_screen_text}\n"
            f"At most {hero_limit(band, len(lines))} lines may be in hero beats.\n"
            f"Avoid: {'; '.join(rules.avoid)}.\n\n"
            + (f"Teacher's brief: {brief}\n\n" if brief else "")
            + (f"{direction}\n\n" if direction else "")
            + f"Lyrics, one entry per line:\n{lyric}")
