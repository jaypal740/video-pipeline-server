from src.schema.lesson.lesson_schema import Beat, LessonPlan
from src.schema.song.song_analysis_schema import LyricLine
from src.schema.storyboard.script_schema import Action, BeatScript, BeatScriptDraft, DrawingList, Screen
from src.service.lesson.lesson_rules import band_rules
from src.schema.storyboard.script_schema import DrawingKind
from src.service.storyboard.script_rules import (MAX_ON_SCREEN, MAX_ROW, MAX_STILL,
                                                 PALETTE_COLOURS, ZOOM_LEAST, ZOOM_MOST, style_of)

# Bumped whenever the prompt changes, so every run is scripted again with the new prompt.
PROMPT_VERSION = "storyboard.v23"

SYSTEM = """You direct short educational music videos for school classes, made as motion graphics: flat 2D \
drawings appear, move and swap in exact time with the song. The song and its lesson plan already exist. The \
plan splits the song into beats, each teaching one idea. Every drawing is made once, as a cut-out, and reused \
in each beat that needs it, so the whole video keeps one look.

There are four kinds of drawing:
- performer: whoever sings the song (the Sun, a caterpillar, a child), in one pose. The performer is the \
only drawing with a face. It sings the hero beats and may appear in others. When the performer is the very \
thing the song teaches about (a water drop singing about the water cycle), the performer acts the lesson \
out: it is drawn in each state the lesson shows, and no separate copy of that thing is drawn without a face. \
It is one creature, so one drawing of it is on screen at a time: two would look like two animals.
- object: one real thing the song teaches about, drawn as it really looks, with no face.
- background: a full-frame backdrop that holds nothing a beat needs to move.
- icon: a flat symbol for something real the lesson names that has no shape of its own (wind, heat, a \
sound). Never decoration: no glows, sparkles, rings, frames or loose arrows.

A drawing of something alive or active has a motion: how it moves in a short loop that repeats while it is \
on screen (a bird's wings flap, a river flows). Backgrounds and icons never move.

Our code draws every written word: the key word and the figures as they are sung, a stage's name under it, \
and the lyric captions. \
No drawing contains letters, numbers or labels. Be factually accurate: never draw what the plan says the \
video must not teach."""

DRAWINGS_TASK = f"""List every drawing this video needs, and no more than it needs. A thing that appears in \
several beats is one drawing; add another state of it only where the lesson shows the change (a seed, a \
sprout, a grown plant). A thing that hangs from, sits on or grows out of another is one drawing that shows \
the two joined (a nest on its branch, a fruit on its twig): our code places drawings side by side and cannot \
join them, and a drawing shows nothing its look does not name. What one thing does to another (a caterpillar \
eating a leaf, a bee drinking from a flower) is likewise one drawing of it doing so, the two touching, and \
the doing is its motion (chewing); never stand the thing beside it and swap the thing. A drawing is on \
screen once at a time: when several of the same thing must be seen together (three fallen leaves), list one \
drawing for each. Every drawing is something the lesson names; our code draws the arrows between stages. \
The sun, the sky, the weather and the time of day belong in a background, never in a drawing of their own, \
unless the song is about them, and nothing is drawn to stand for time passing (a sun crossing the sky, a \
clock). List one background for each place the writers' plan sets its scenes in, at most six, places that \
look alike sharing one; where it names no places, two to four, one for each part of the song. A background \
is the real place around the subject, seen close enough to know it (under a leafy branch, inside a beehive, \
a flower garden), and a chorus has one open place of its own. A change the lesson teaches (ice melting, a \
chick hatching) is drawn as its steps, one drawing each: the thing just before, the moment it changes, and \
the thing after, whose look says what is now different. When a living thing changes its form (a tadpole \
into a frog), the look of every drawing of the new form says its whole new body, each part with its real \
shape and colour (a frog's wide head, short rounded body and four legs), and the one small thing it keeps so the viewer \
knows who it is (its eyes). Nothing of the old body is left to be assumed: a real animal is never drawn as a \
mix of two forms. What a thing leaves behind (an empty shell, a fallen \
leaf) has the shape and markings of the thing it came from, and whatever appears in two drawings is described \
with the same words in both. When the lesson is a cycle, list the drawing in which its last stage makes \
its first (the grown plant dropping its seed). For each drawing give:
- id: a short snake_case name.
- kind: performer, object, background or icon.
- look: what to draw, in one sentence, with its real shape and colours. For a performer drawing, the pose. \
Say only what is seen, never what is absent: the picture is drawn from these words, and whatever they name \
gets drawn. For a thing that has lost something, say what is there now (a tree of bare brown branches).
- name: when the lesson teaches a process, each of its stages has one drawing that stands for it in the row \
or ring; give that drawing the stage's name as a child should learn it, in one or two words, a key term \
where one fits (egg, larva, pupa, adult). Our code writes it under the drawing there. Null for every other \
drawing.
- of: when it shows the same body as an earlier drawing in another pose (a bird flying, of the same bird \
perched), that drawing's id, so the two are drawn alike. A thing that has changed into something \
that looks different is not of anything (the sprout is not of the seed). Otherwise null.
- motion: how it moves in its loop, in a few words. Null for backgrounds, icons and things that hold still.
- sings: true for a performer drawing whose face and mouth show clearly, so it can sing a hero beat. False \
for every other drawing, and for a performer state with no mouth to move (a seed that has not sprouted).

List the performer in as few drawings as the video needs, each with a motion: \
one or two poses when it only sings and presents, one per state when it acts the lesson out. At least one \
sings. Also give performer: who sings \
the song and what they look like, in one or two sentences (null when there are no performer drawings); when \
the performer changes form through the song, say there only what every form keeps (its name, its eyes, its \
manner), because each drawing's look says the body it has then. And \
palette: {PALETTE_COLOURS[0]} to {PALETTE_COLOURS[1]} hex colours that suit the topic."""

SCRIPT_TASK = f"""For each beat you are asked for, give beat (its id), background (a background drawing's \
id) and actions. Drawings stay on screen from one beat to the next until they are hidden or swapped; a beat \
with a different background starts with an empty screen. Times are seconds of the song, inside the beat.

Actions:
- show: drawing appears at pos (left, center, right, top or bottom) at `at`. Or at pos row: it takes the \
next place in a row of stages that our code lays out left to right across the middle of the screen, smaller, \
with an arrow from each stage to the next. Or at pos ring: it takes the next place round a ring that our \
code lays out clockwise from the top, with an arrow from each stage to the next and one from the last stage \
back to the first. Every stage keeps its place and goes on moving.
- hide: drawing leaves at `at`.
- swap: drawing is replaced by `to`, in the same spot, at `at`. Swap a thing only for what it really turns \
into next (the seed for the sprout, never the seed for the grown plant) or for another pose of itself.
- move: drawing slides to pos, from `at` until `until`.
- zoom: drawing grows or shrinks to scale, from `at` until `until`. 1 is its normal size, {ZOOM_LEAST} the \
smallest; a performer drawing fits the screen up to {ZOOM_MOST[DrawingKind.PERFORMER]}, an object up to \
{ZOOM_MOST[DrawingKind.OBJECT]}.
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
- The picture never holds still for more than {MAX_STILL:g} s of a beat: within that time a drawing is \
shown, swapped, moved or zoomed, and it is the next step of the beat's idea, never a filler. A highlight or \
a camera move does not count.
- The performer is one creature, so one drawing of it is on screen at a time; a second beside it would look \
like another animal. To show it change, swap it. Only the stages of a row or ring are several of its \
drawings together, and while they are up no other drawing of it is on screen.
- Nothing hides anything else. Left, center and right stand clear of each other at normal size. Top and \
bottom overlap a drawing at center, so use them only while center is empty or holds something small. A \
drawing zoomed big reaches into the places beside it. Our code lays the frames out, checks this, and tells \
you what covers what.
- Each beat plays on the background of the place its part of the story happens in; a chorus, and a row or \
ring, keep one open place every time they come.
- A row or ring shows every stage of a process together, but small. So tell the process big first: on the \
line that sings of a stage, that stage is at center, big, and is seen changing from the stage before it. \
Then gather the stages on the line that names or sums up the whole process (in a chorus, its last line): \
clear the middle and show each stage's named drawing at pos row, in its true order, a moment apart, so the \
viewer sees every stage, its name and what comes next. When the process is a cycle, one that starts again from its last stage, use pos \
ring instead. Do it this way even where the writers' plan lights a circle up line by line. At most {MAX_ROW} stages; a stage that is named again is highlighted. While the row is up \
nothing else is at left, center or right. While the ring is up nothing else is at left, right, top or \
bottom. Its middle stays empty; only when the performer is not itself one of the stages may a hero beat's \
singer stand there, at pos center.
- A cycle is closed on screen: when the song says it starts again, show its last stage making its first \
(the bird with its new egg), then the first stage once more.
- In a hero beat the performer is on screen from the beat's start, and our code makes whichever of its \
drawings has a mouth sing. A performer that only presents stays, in one pose, beside what the beat shows. A \
performer that acts the lesson out is staged exactly as in any other beat: the stage it is in is the one \
that sings, and no other drawing of it is added.
- When a beat teaches a change, play its steps in one spot: the drawing from before is on screen, then it is \
swapped for the moment of change and that for the drawing after, each as its words are sung, so the viewer \
watches the change happen before its key word is named. A thing that has grown is zoomed bigger at each \
step, by a quarter or more each time, starting small enough to leave room, so its growth is plain to see. \
When the song says it happens again, \
play the steps again, as many times as the plan says.
- A music beat has no lyrics, and some beats go on long after their lines are sung. Fill that time with \
the lesson: at the song's start, the thing the lesson starts with, coming alive; later, a replay of the stages \
taught so far, big and one at a time, in order, each doing what it does, ending on the picture the next beat \
or the plan's ending needs.
- When a chorus is sung again, stage it as it was staged the first time: the same background, the same \
drawings in the same places and order, timed to the new lines, so the viewer knows the picture and can sing \
along. A hero beat is staged the same way; only a performer that just presents is added beside it."""

LIMITS = f"""The script keeps to these limits, so never ask for what breaks them: a drawing is on screen once at a time; the \
performer is one drawing at a time, and only the stages of a row or ring show several of its drawings \
together, with none outside them; a hero beat has the performer on screen from its start; nothing hides \
another drawing; a beat's picture never holds still for \
more than {MAX_STILL:g} s; the places are left, center, right, top, bottom, row and ring."""


REVIEW_TASK = f"""Below is the finished script for this song. Check it against the plan before any drawing is \
made. Go through what each beat must teach (its concept, its hint and the writers' pictures) and list what a \
child watching with the sound off would not see or would get wrong:
- a step of a change that is missing, out of order or shown fewer times than it is sung;
- a thing that is sung about but is not on screen as it is sung, or the wrong thing on screen under a key word;
- something the lines sing of doing (eating, drinking, laying) that is not seen being done by the one who does it;
- a cycle whose last stage is not shown leading back to its first;
- a chorus that comes back staged differently from its first time;
- a drawing that is decoration and teaches nothing, or shows something the lesson does not name: a sun, a \
moon or a clock put in to show time passing must go;
- a drawing whose look gives a living thing the wrong body for its stage, mixes two forms, or leaves the \
body of a new form unsaid;
- anything else a child would learn wrong from the picture: say what they would wrongly believe.
A stage shown big on its own line and gathered into the row or ring only on the line that sums the process \
up is how our video shows a circle of stages; never ask for the ring earlier. List only what is really \
missing or wrong, one sentence each: the beat's id, what is missing and how to fix \
it with our drawings and actions. Do not ask for written words, sounds, lighting or camera work. A hero \
beat staged like the chorus's first time, the stage itself singing, is right: never ask for a second drawing \
of the performer. An empty list means the script teaches the plan.

{LIMITS}"""

REPAIR_TASK = f"""A review of the script below found the gaps listed under it. Close each gap, changing as \
little as you can:
- beats: only the beats that must change, each given whole, with every action it keeps as well as the new \
ones. A beat that follows a changed beat on the same background starts with what that beat leaves on screen; \
change it too if it no longer fits.
- drawings: only the new drawings those beats need, each with id, kind, look, of, motion and sings as the \
listed drawings have them. Never list a drawing again. A drawing is on screen once at a time, so when several \
of the same thing must be seen together (three fallen leaves) each is a drawing of its own.
Leave a gap as it is when our drawings and actions cannot close it.

{LIMITS}

{SCRIPT_TASK}"""


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


def jamwise_pictures(export: dict) -> str | None:
    """How JamWise planned to show what the song teaches: each scene's place and demonstration, the chorus
    picture and the ending. The script follows these directly, so nothing is lost on the way through the lesson."""
    brief = export.get("creativeBrief") or {}
    shown = {m.get("sceneId"): m.get("visualDemonstration") for m in brief.get("teachingMoments", [])}
    parts = [f"- Scene {s.get('id')}, {s.get('title')}" + (f", set in {s['place']}" if s.get("place") else "")
             + f": teaches {s.get('teaches')}. Shown as: "
             f"{shown.get(s.get('id')) or (s.get('teachingMoment') or {}).get('visualDemonstration')}"
             for s in brief.get("scenes", []) if s.get("teaches")]
    world = brief.get("visualWorld") or {}
    parts += [f"- The world: {world[k]}" for k in ("primarySetting", "secondarySetting") if world.get(k)]
    chorus = world.get("chorusVisual")
    parts += [f"- Chorus picture: {chorus}"] if chorus else []
    parts += [f"- {a.get('statistic')}: {a.get('visual')}" for a in brief.get("numberAnchors", []) if a.get("visual")]
    parts += [f"- Section {s.get('section')}: {'; '.join(s['keyMoments'])}" for s in brief.get("sectionPlans", [])
              if s.get("keyMoments")]
    parts += [f"- Ending: {brief['loopVisual']}"] if brief.get("loopVisual") else []
    if not parts:
        return None
    return ("The song's writers planned these pictures for it. Show what they show: the same things, in the same "
            "order, as many times, in the places they name. Do it with our drawings and actions, and leave out their "
            "written words, faces on things, sounds, lighting and camera work, and anything the plan must not teach.\n"
            + "\n".join(parts))


def song_prompt(plan: LessonPlan, lines: list[LyricLine], performer: str | None, pictures: str | None = None) -> str:
    """The song as every call sees it: who it is for, what it teaches and its beats with their lines."""
    rules = band_rules(plan.grade_band)
    return (f"Audience: grades {plan.grade_band} (grade 0 is kindergarten).\n"
            f"Style of every drawing: {style_of(plan.grade_band)}.\n"
            f"Avoid: {'; '.join(rules.avoid)}.\n"
            f"Topic: {plan.topic} ({plan.subject}). {plan.summary}\n"
            f"Key terms: {'; '.join(f'{t.term}: {t.definition}' for t in plan.key_terms)}\n"
            f"Must not teach: {'; '.join(plan.must_not_teach) or 'nothing listed'}\n"
            f"Performer: {performer or 'decide from the lyrics who is singing'}\n\n"
            + (f"{pictures}\n\n" if pictures else "")
            + "Beats:\n" + "\n".join(_beat(b, lines) for b in plan.beats))


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
    drawings = "\n".join(f"- {d.id} ({d.kind}{', sings' if d.sings else ''}, motion: {d.motion or 'none'}"
                         f"{f', stage name: {d.name}' if d.name else ''}): {d.look}" for d in listed.drawings)
    so_far = "\n".join(f"{s.beat} {s.background}: " + "; ".join(_action(a) for a in s.actions) for s in done)
    on = ", ".join(f"{d} at {p}" for d, p in before.on.items())
    screen = f"{before.background} with {on or 'nothing in front of it'}" if before.background else "nothing"
    return (f"{song}\n\nDrawings:\n{drawings}\n\nScripts so far:\n{so_far or 'None yet.'}\n\n"
            f"On screen when {beats[0].id} starts: {screen}.\n\n{SCRIPT_TASK}\n\n"
            f"Write the scripts for beats {beats[0].id} to {beats[-1].id}.")


def review_prompt(song: str, listed: DrawingList, scripts: list[BeatScript]) -> str:
    """The plan and the finished script, for the review that looks for what the script does not teach."""
    drawings = "\n".join(f"- {d.id} ({d.kind}): {d.look}" for d in listed.drawings)
    script = "\n".join(f"{s.beat} {s.background}: " + "; ".join(_action(a) for a in s.actions) for s in scripts)
    return f"{song}\n\n{REVIEW_TASK}\n\nDrawings:\n{drawings}\n\nScript:\n{script}"


def repair_prompt(song: str, listed: DrawingList, scripts: list[BeatScriptDraft], gaps: list[str]) -> str:
    """The plan, the script as it stands and the gaps a review found in it, for the repair of only those parts."""
    drawings = "\n".join(f"- {d.id} ({d.kind}{', sings' if d.sings else ''}, motion: {d.motion or 'none'}): {d.look}"
                         for d in listed.drawings)
    script = "\n".join(f"{s.beat} {s.background}: " + "; ".join(_action(a) for a in s.actions) for s in scripts)
    return (f"{song}\n\n{REPAIR_TASK}\n\nDrawings:\n{drawings}\n\nScript:\n{script}\n\nGaps:\n"
            + "\n".join(f"- {g}" for g in gaps))


def _action(a: Action) -> str:
    words = [a.verb, a.drawing or a.text, f"to {a.to}" if a.to else None, a.pos, f"x{a.scale:g}" if a.scale else None,
             f"@{a.at:g}" + (f"-{a.until:g}" if a.until is not None else "")]
    return " ".join(str(w) for w in words if w)
