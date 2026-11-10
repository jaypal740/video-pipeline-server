from src.schema.lesson.lesson_schema import GradeBand, VideoType
from src.schema.song.song_analysis_schema import LyricLine
from src.service.lesson.lesson_rules import (MAX_SEQUENCE_SECONDS, MAX_SEQUENCES, MAX_STEPS, MIN_SEQUENCE_SECONDS,
                                             MAX_SHOWN_TERMS, MIN_STEPS, band_rules, face_limit, format_rules,
                                             place_limit, repeats, step_seconds)

# Bumped whenever the prompt changes, so every run is planned again with the new prompt.
PROMPT_VERSION = "lesson.v5"

SYSTEM = f"""You plan short educational music videos for school classes. The song already exists and \
drives the video: you never change its words or order, you decide what each line teaches and what the \
viewer sees while it is sung.

Only people, such as a narrator or presenter, are characters. Cells, germs, molecules, animals, objects and \
ideas look and behave as they really do: never cartoon faces, eyes or human expressions on them, and no \
invented mascots or metaphors standing in for the real thing.

For every lyric line give:
- teaches: the one idea the line teaches, in plain words.
- show: one concrete, visible scene or action that makes that idea clear to the age group. Real things \
doing real things, filmed where they really are, never a picture of them on a screen, card, poster or \
window. No written words, labels, charts or diagrams in the picture; our code adds any text.
- tag, exactly one of:
  - face_to_camera: a person (the narrator or a presenter) sings the line straight to the viewer. Use it for hooks, choruses and \
lines that address the viewer, within the limit you are given.
  - number: the line's point is a figure (a year, an amount, a percentage). Put the figure in \
number_text exactly as it is written in the line; our code draws it on screen as it is sung.
  - illustrate: everything else.
  - unclear: only for lines marked [unclear]; their audio may not say these words.
A line marked [repeat of N] is the chorus sung again: give it the same tag as line N but a new show that \
takes the idea further, so the chorus never looks the same twice.

Then group the lines into sequences: runs of consecutive lines on one topic, together covering every line \
once, in order. Use at most {MAX_SEQUENCES} sequences, each lasting {MIN_SEQUENCE_SECONDS:.0f} to {MAX_SEQUENCE_SECONDS:.0f} s from \
its first line's start to its last line's end: join a shorter one to a neighbour on a related topic, \
split a longer one where its topic changes. The video shows each sequence as a run of moving shots in \
one place, so the viewer watches one idea unfold instead of a slideshow. For each give first_line and \
last_line, the topic, the one place it is shown in, change: what visibly moves or changes from its first \
shot to its last, real things doing what they really do, and terms: the key terms its lines sing, copied \
exactly as they are sung. A term the topic is about but its lines don't sing does not go in terms. A \
chorus sung again goes in a sequence with the same place as the first chorus.

The video must teach with the sound off: a viewer who cannot hear the song must still understand each \
sequence from its pictures and the term labels our code draws. So for each sequence also give fact: the \
one thing a viewer must understand from it, in one plain sentence, and steps: {MIN_STEPS} to {MAX_STEPS} \
visible stages that show that fact, in the order its lines sing them (the pictures follow the song, even \
when the song names an inner layer after an outer one). Each step is something a camera could film, real things doing what they really do, never an abstract \
idea, and needs at least the step length you are given. Where the plan you are given shows how to \
demonstrate a concept, use that idea with real things and no written words. To explain parts (the layers \
of the Sun, the parts of a cell), show one part per step, from one view the viewer keeps, so each part \
can be labelled as it is sung. When a part the explanation needs is not sung (the corona, after the song \
names the inner layers), put it in shown_terms, at most {MAX_SHOWN_TERMS}, each also in key_terms: our \
code labels it on screen anyway. Steps film the real thing too, never its picture on a screen or card. A \
fact may rest on a figure only when one of the sequence's lines is tagged number, since only then is the \
figure drawn on screen; otherwise write the fact without it.

Also give the topic, the school subject, a two-sentence summary, the video_type from the allowed list \
(use each type's "Pick it when") with a one-line reason, the key terms with child-friendly definitions, \
and must_not_teach: wrong ideas the pictures must not teach. Be factually accurate; if a line is wrong \
or misleading, say so in must_not_teach rather than illustrating the error."""


def jamwise_direction(export: dict) -> str:
    """What JamWise planned the song to teach, as direction for the lesson: its goals, facts, vocabulary and
    how it meant to demonstrate each concept. Its characters' looks are left out: our pipeline designs them."""
    brief, package = export.get("creativeBrief") or {}, export.get("contentPackage") or {}
    parts = ["The song was written from this plan. Follow its topics, terms and numbers, but check every fact: "
             "where the plan is wrong, teach what is true."]
    parts += [f"- Learning goal: {g}" for g in package.get("learningObjectives", [])]
    parts += [f"- Scene {s.get('id')}: {s.get('title')}: teaches {s.get('teaches')}" for s in brief.get("scenes", [])]
    parts += [f"- Section {s.get('section')}: {s.get('educationalFocus')}" for s in brief.get("sectionPlans", [])]
    parts += [f"- Number: {a.get('statistic')}" for a in brief.get("numberAnchors", [])]
    parts += [f"- Fact: {f.get('kidVersion') or f.get('fact')} {f.get('explanation') or ''}".rstrip()
              for f in package.get("keyFacts", [])]
    parts += [f"- Wrong idea to avoid: {f.get('misconception')}" for f in package.get("keyFacts", [])
              if f.get("misconception")]
    parts += [f"- Term {c.get('term')}: {c.get('kidDefinition')}"
              + (f" Not to be confused with: {'; '.join(c['nonExamples'])}" if c.get("nonExamples") else "")
              for c in package.get("vocabularyCards", []) if c.get("term")]
    parts += [f"- How to show {m.get('concept')}: {m.get('visualDemonstration')}"
              for m in brief.get("teachingMoments", []) if m.get("concept")]
    wow = package.get("wowFact") or {}
    if wow.get("fact"):
        parts.append(f"- Wow fact: {wow['fact']}")
    terms = featured_terms(export)
    if terms:
        parts.append(f"- Featured terms: {', '.join(terms)}")
    return "\n".join(parts)


def featured_terms(export: dict) -> list[str]:
    return [t["word"] for t in (export.get("creativeBrief") or {}).get("featuredTerms", []) if t.get("word")]


def user_prompt(lines: list[LyricLine], band: GradeBand, types: list[VideoType], brief: str | None,
                direction: str | None = None) -> str:
    rules = band_rules(band)
    formats = "\n".join(
        f"- {t}: Pick it when: {format_rules(t).pick_when} World: {format_rules(t).world} "
        f"On camera: {format_rules(t).on_camera} "
        f"At most {face_limit(band, t, len(lines))} face_to_camera lines."
        for t in types)
    again = repeats(lines)
    lyric = "\n".join(f"{l.index}. [{l.start:.1f}-{l.end:.1f}s]{f' [{l.section}]' if l.section else ''}"
                      f"{f' [repeat of {again[l.index]}]' if l.index in again else ''}"
                      f"{' [unclear]' if l.unclear else ''} {l.text}"
                      for l in lines)
    return (f"Audience: grades {band} (grade 0 is kindergarten).\n"
            f"Look: {rules.look}\nOn-screen text allowed: {rules.on_screen_text}\n"
            f"Sequences use at most {place_limit(band)} places between them; each step lasts at least "
            f"{step_seconds(band):.1f} s.\n"
            f"Avoid: {'; '.join(rules.avoid)}.\n\n"
            f"Allowed video types:\n{formats}\n\n"
            + (f"Teacher's brief: {brief}\n\n" if brief else "")
            + (f"{direction}\n\n" if direction else "")
            + f"Lyrics, one entry per line:\n{lyric}")
