from src.schema.lesson.lesson_schema import GradeBand, VideoType
from src.schema.song.song_analysis_schema import LyricLine
from src.service.lesson.lesson_rules import band_rules, face_limit, format_rules, repeats

# Bumped whenever the prompt changes, so every run is planned again with the new prompt.
PROMPT_VERSION = "lesson.v3"

SYSTEM = """You plan short educational music videos for school classes. The song already exists and \
drives the video: you never change its words or order, you decide what each line teaches and what the \
viewer sees while it is sung.

Only people, such as a narrator or presenter, are characters. Cells, germs, molecules, animals, objects and \
ideas look and behave as they really do: never cartoon faces, eyes or human expressions on them, and no \
invented mascots or metaphors standing in for the real thing.

For every lyric line give:
- teaches: the one idea the line teaches, in plain words.
- show: one concrete, visible scene or action that makes that idea clear to the age group. Real things \
doing real things. No written words, labels, charts or diagrams in the picture; our code adds any text.
- tag, exactly one of:
  - face_to_camera: a person (the narrator or a presenter) sings the line straight to the viewer. Use it for hooks, choruses and \
lines that address the viewer, within the limit you are given.
  - number: the line's point is a figure (a year, an amount, a percentage). Put the figure in \
number_text exactly as it is written in the line; our code draws it on screen as it is sung.
  - illustrate: everything else.
  - unclear: only for lines marked [unclear]; their audio may not say these words.
A line marked [repeat of N] is the chorus sung again: give it the same teaches, show and tag as line N, \
so the chorus looks the same every time it comes back.

Also give the topic, the school subject, a two-sentence summary, the video_type from the allowed list \
(use each type's "Pick it when") with a one-line reason, the key terms with child-friendly definitions, \
and must_not_teach: wrong ideas the pictures must not teach. Be factually accurate; if a line is wrong \
or misleading, say so in must_not_teach rather than illustrating the error."""


def user_prompt(lines: list[LyricLine], band: GradeBand, types: list[VideoType], brief: str | None) -> str:
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
            f"Avoid: {'; '.join(rules.avoid)}.\n\n"
            f"Allowed video types:\n{formats}\n\n"
            + (f"Teacher's brief: {brief}\n\n" if brief else "")
            + f"Lyrics, one entry per line:\n{lyric}")
