from src.schema.lesson.lesson_schema import Beat, LessonPlan
from src.schema.song.song_analysis_schema import LyricLine
from src.schema.storyboard.script_schema import Action, BeatScriptDraft, DrawingList, Screen
from src.service.lesson.lesson_rules import band_rules
from src.service.storyboard.script_rules import (LONG_BEAT_SECONDS, MAX_DRAWINGS, MAX_ON_SCREEN, MAX_PERFORMER_POSES,
                                                 MAX_ROW, MIN_LONG_BEAT_ACTIONS, PALETTE_COLOURS, ZOOM_SCALE, style_of)

# Bumped whenever the prompt changes, so every run is scripted again with the new prompt.
PROMPT_VERSION = "storyboard.v17"

SYSTEM = """You direct short educational music videos for school classes, made as motion graphics: flat 2D \
drawings appear, move and swap in exact time with the song. The song and its lesson plan already exist. The \
plan splits the song into beats, each teaching one idea. Every drawing is made once, as a cut-out, and reused \
in each beat that needs it, so the whole video keeps one look.

There are four kinds of drawing:
- performer: whoever sings the song (the Sun, a caterpillar, a child), in one pose. The performer is the \
only drawing with a face. It sings the hero beats and may appear in others. When the performer is the very \
thing the song teaches about (a water drop singing about the water cycle), the performer acts the lesson \
out: it is drawn in each state the lesson shows, and no separate copy of that thing is drawn without a face.
- object: one real thing the song teaches about, drawn as it really looks, with no face.
- background: a full-frame backdrop that holds nothing a beat needs to move.
- icon: a flat symbol such as an arrow.

A drawing of something alive or active has a motion: how it moves in a short loop that repeats while it is \
on screen (a bird's wings flap, a river flows). Backgrounds and icons never move.

Our code draws every written word: the key word and the figures as they are sung, and the lyric captions. \
No drawing contains letters, numbers or labels. Be factually accurate: never draw what the plan says the \
video must not teach."""

DRAWINGS_TASK = f"""List every drawing this video needs, at most {MAX_DRAWINGS}. A thing that appears in \
several beats is one drawing; add another state of it only where the lesson shows the change (a seed, a \
sprout, a grown plant). A thing that hangs from, sits on or grows out of another is one drawing that shows \
the two joined (a nest on its branch, a fruit on its twig): our code places drawings side by side and cannot \
join them, and a drawing shows nothing its look does not name. A change the lesson teaches (ice melting, a \
chick hatching) is drawn as its steps, one drawing each: the thing just before, the moment it changes, and \
the thing after, whose look says what is now different. What a thing leaves behind (an empty shell, a fallen \
leaf) has the shape and markings of the thing it came from, and whatever appears in two drawings is described \
with the same words in both. For each drawing give:
- id: a short snake_case name.
- kind: performer, object, background or icon.
- look: what to draw, in one sentence, with its real shape and colours. For a performer drawing, the pose.
- of: when it shows the same body as an earlier drawing in another pose (a bird flying, of the same bird \
perched), that drawing's id, so the two are drawn alike. A thing that has changed into something \
that looks different is not of anything (the sprout is not of the seed). Otherwise null.
- motion: how it moves in its loop, in a few words. Null for backgrounds, icons and things that hold still.
- sings: true for a performer drawing whose face and mouth show clearly, so it can sing a hero beat. False \
for every other drawing, and for a performer state with no mouth to move (a seed that has not sprouted).

List the performer in as few drawings as the video needs, at most {MAX_PERFORMER_POSES}, each with a motion: \
one or two poses when it only sings and presents, one per state when it acts the lesson out. At least one \
sings. Also give performer: who sings \
the song and what they look like, in one or two sentences (null when there are no performer drawings), and \
palette: {PALETTE_COLOURS[0]} to {PALETTE_COLOURS[1]} hex colours that suit the topic."""

SCRIPT_TASK = f"""For each beat you are asked for, give beat (its id), background (a background drawing's \
id) and actions. Drawings stay on screen from one beat to the next until they are hidden or swapped; a beat \
with a different background starts with an empty screen. Times are seconds of the song, inside the beat.

Actions:
- show: drawing appears at pos (left, center, right, top or bottom) at `at`. Or at pos row: it takes the \
next place in a row of stages that our code lays out left to right across the middle of the screen, smaller, \
with an arrow from each stage to the next. Every stage keeps its place and goes on moving.
- hide: drawing leaves at `at`.
- swap: drawing is replaced by `to`, in the same spot, at `at`. Swap a thing only for what it really turns \
into next (the seed for the sprout, never the seed for the grown plant) or for another pose of itself.
- move: drawing slides to pos, from `at` until `until`.
- zoom: drawing grows or shrinks to scale ({ZOOM_SCALE[0]} to {ZOOM_SCALE[1]}; 1 is its normal size), from \
`at` until `until`.
- highlight: drawing pulses so the viewer looks at it, from `at` until `until`.
- push_in, pull_back: the camera moves closer or further away, from `at` until `until`.
- pan: the camera slides towards pos (left or right), from `at` until `until`.
Never write keyword, number or lipsync actions; our code adds them.

Rules:
- Show the beat's idea. The thing its key word names is on screen when the word is sung, and something \
happens to it at that moment (it appears, is swapped in or is highlighted).
- Something must really move in every beat: a drawing that has a motion is on screen, or one drawing is \
swapped for another. Still drawings with only a camera move are not enough.
- At most {MAX_ON_SCREEN} drawings on screen at once, each at its own pos.
- A beat longer than {LONG_BEAT_SECONDS:.0f} s needs at least {MIN_LONG_BEAT_ACTIONS} actions.
- Only one performer drawing is on screen at a time, except in the row.
- When beats tell the steps of a process or a cycle in order, build the row: show each stage at pos row at \
the moment it is sung, in its true order, and leave it there, so the viewer sees every stage and what comes \
next. Keep one background through those beats so the row stays. At most {MAX_ROW} stages; while the row is \
up nothing else is at left, center or right, and a stage that is named again is highlighted.
- In a hero beat a performer drawing that sings is on screen from the beat's start and stays, unchanged, \
to its end; our code makes it sing.
- When a beat teaches a change, play its steps in one spot: the drawing from before is on screen, then it is \
swapped for the moment of change and that for the drawing after, each as its words are sung, so the viewer \
watches the change happen before its key word is named. Zoom a thing that has grown a little bigger. When \
the song says it happens again, play the steps again.
- A music beat has no lyrics: keep the picture alive with the performer or the topic's main drawing.
- When an idea comes back (a chorus sung again, a recap), stage it differently from the last time: other \
drawings, other positions, another camera move. Never repeat an earlier beat's actions."""


def jamwise_performer(export: dict) -> str | None:
    """The lead of JamWise's cast: the character the song is sung by, as JamWise pictured them."""
    cast = (export.get("creativeBrief") or {}).get("cast") or []
    lead = next((c for c in cast if c.get("role") == "lead" and c.get("name")), None)
    if lead is None:
        return None
    who = f"{lead['name']}: {lead['concept']}" if lead.get("concept") else lead["name"]
    look = lead.get("visualDescription")
    return who + (f". The song's writers pictured them like this; keep them recognisable, in our style: {look}"
                  if look else "")


def song_prompt(plan: LessonPlan, lines: list[LyricLine], performer: str | None) -> str:
    """The song as both calls see it: who it is for, what it teaches and its beats with their lines."""
    rules = band_rules(plan.grade_band)
    return (f"Audience: grades {plan.grade_band} (grade 0 is kindergarten).\n"
            f"Style of every drawing: {style_of(plan.grade_band)}.\n"
            f"Avoid: {'; '.join(rules.avoid)}.\n"
            f"Topic: {plan.topic} ({plan.subject}). {plan.summary}\n"
            f"Key terms: {'; '.join(f'{t.term}: {t.definition}' for t in plan.key_terms)}\n"
            f"Must not teach: {'; '.join(plan.must_not_teach) or 'nothing listed'}\n"
            f"Performer: {performer or 'decide from the lyrics who is singing'}\n\n"
            "Beats:\n" + "\n".join(_beat(b, lines) for b in plan.beats))


def _beat(beat: Beat, lines: list[LyricLine]) -> str:
    head = f"{beat.id} {beat.start:g}-{beat.end:g}s {beat.type}"
    if not beat.lines:
        return head
    by_index = {l.index: l for l in lines}
    sung = f" sung at {beat.keyword_at:g}s" if beat.keyword_at is not None else ""
    parts = [f"{head}: {beat.concept}"]
    parts += [f'Key word "{beat.keyword}"{sung}'] if beat.keyword else []
    parts += [f'Figure our code draws: "{beat.number_text}"'] if beat.number_text else []
    parts += [f"Hint: {beat.hint}"]
    parts += [f"Avoid: {beat.avoid}"] if beat.avoid else []
    parts += [f"Sung from {by_index[i].start:g}s: {by_index[i].text}" for i in beat.lines]
    return "\n    ".join(parts)


def drawings_prompt(song: str) -> str:
    return f"{song}\n\n{DRAWINGS_TASK}"


def script_prompt(song: str, listed: DrawingList, done: list[BeatScriptDraft], beats: list[Beat],
                  before: Screen) -> str:
    """One run of beats to script, with the drawings, the scripts so far and what is on screen as it starts."""
    drawings = "\n".join(f"- {d.id} ({d.kind}{', sings' if d.sings else ''}, motion: {d.motion or 'none'}): {d.look}"
                         for d in listed.drawings)
    so_far = "\n".join(f"{s.beat} {s.background}: " + "; ".join(_action(a) for a in s.actions) for s in done)
    on = ", ".join(f"{d} at {p}" for d, p in before.on.items())
    screen = f"{before.background} with {on or 'nothing in front of it'}" if before.background else "nothing"
    return (f"{song}\n\nDrawings:\n{drawings}\n\nScripts so far:\n{so_far or 'None yet.'}\n\n"
            f"On screen when {beats[0].id} starts: {screen}.\n\n{SCRIPT_TASK}\n\n"
            f"Write the scripts for beats {beats[0].id} to {beats[-1].id}.")


def _action(a: Action) -> str:
    words = [a.verb, a.drawing, f"to {a.to}" if a.to else None, a.pos, f"x{a.scale:g}" if a.scale else None,
             f"@{a.at:g}" + (f"-{a.until:g}" if a.until is not None else "")]
    return " ".join(str(w) for w in words if w)
