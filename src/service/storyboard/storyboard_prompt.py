from src.schema.lesson.lesson_schema import BandRules, FormatRules, LessonPlan, LessonSequence, VideoType
from src.schema.song.song_analysis_schema import LyricLine
from src.schema.storyboard.storyboard_schema import (BoardHeader, CastMember, ShotDraft, ShotKind, Slot,
                                                     SoundOffWatch, StoryboardDraft, Verdict)
from src.service.storyboard.storyboard_rules import HOST, LIMITS

# Bumped whenever the prompt changes, so every run is boarded again with the new prompt.
PROMPT_VERSION = "storyboard.v11"

SYSTEM = f"""You storyboard short educational music videos for school classes. The song, its lesson plan \
and the shot timings already exist; you decide what each shot shows and how the camera films it. An image \
model draws each shot's opening frame from your words and a video model animates it, so write only what \
can be seen, in plain concrete words.

First declare, once for the whole video:
- shape: one sentence on how the video moves from start to end, at most {LIMITS["shape"]} words.
- cast: every person or thing seen in more than one shot. id in snake_case starting with a letter, unique \
across cast and places; kind person or thing; look: \
the fixed appearance in at most {LIMITS["cast look"]} words (colours, materials, shapes, clothes); size: its \
real size in at most {LIMITS["size"]} words; scale: the rung it is seen at (cosmic, planet, landscape, human, \
object, microscopic, molecular). A thing that is replaced by a new one (a second cable, a new ship) is a new \
cast member with its own id. The host and the host's place are given; never declare them.
- places: id, look in at most {LIMITS["place look"]} words, lighting in at most {LIMITS["lighting"]} words, \
scale (cosmic, planet, landscape, human, object, microscopic, molecular) and walkable: true when a person \
could really stand there (a deck, a shore, a lab), false for space, underwater, inside a cell.
- host_outfit, on every place where the host appears: what the host wears there, fitting that place and \
the topic (beach clothes on a beach, a lab coat in a lab, a coat on a ship's deck), at most \
{LIMITS["outfit"]} words of clothes and colours. Name one of the wardrobe kits you are given, or write \
your own. Leave it out for places the host never appears in.
- motifs: visual codes kept all video, e.g. thing "antibiotic", rule "always a cyan liquid", the rule in at \
most {LIMITS["motif rule"]} words. A motif is about appearance only (colour, material, shape), never about \
where a thing is or what it does. Write each motif's rule into the look of the cast member or place it \
belongs to, so every picture keeps it.
- sequence_places: one entry per sequence of the lesson, in order. home: the place the sequence is \
shown in; sequences the lesson sets in the same place share a home. zooms, when the topic needs them: up \
to two more places that show the same topic at other scales, one closer and one wider, such as the atoms \
inside the Sun's core, or Earth's orbit where a satellite is hit, so small things are seen at their own \
size and far views from where they really are. Every step of the sequence must have a place at its scale \
among the home and zooms. Leave zooms empty when the home shows everything. times, when the sequence \
passes through other hours: the home location declared again as its own place with that lighting (the \
same field at night, at dusk), so a night shot is drawn from a night plate. A shot that opens at night, \
dusk, noon or dawn is set in a place lit for that hour.

Then, one sequence at a time, one shot per slot of that sequence:
- place and subjects (up to 3 ids from cast). A subject must suit the place's scale: a hose-sized cable is \
never the subject of a planet-wide view; leave subjects empty when the place itself is the picture.
- size (wide, medium_wide, medium, medium_close, close, extreme_close), angle (eye, low, high, overhead) and \
position of the main subject (left_third, centre, right_third).
- start_state: the frozen opening moment as physical facts, at most {LIMITS["start_state"]} words. It is \
what the first frame shows, before anything happens: "the key is up, the operator's hand rests beside it", \
never the result.
- action: one visible action in at most {LIMITS["action"]} words.
- end_state: what is visibly different at the end, at most {LIMITS["end_state"]} words.
- camera: move (static, pan, tilt, push_in, pull_back, truck, pedestal, orbit, follow), direction (pan, \
truck and orbit: left or right; tilt and pedestal: up or down; follow: left, right, toward or away) and speed \
(slow, medium, fast). One move per shot.
- motion (low, medium, high), screen_direction (left_to_right, right_to_left, toward, away, none) and \
transition_in (cut, match_cut, dissolve).
- moves: always subject. Something in the picture really moves or changes in every shot; a still \
picture with only the camera moving over it reads as a slideshow.
- step: the number of the sequence's step the shot shows, from 1. Sung shots give the step the sequence \
has reached.
- label: when a term from the sequence's terms is sung during this slot and the shot is framed on the \
thing it names, that term exactly as listed; our code draws it on screen. A shown term, which the song does \
not sing, labels the shot framed on that part wherever it fits. Otherwise leave it empty.

Rules:
- Teach with the sound off: a viewer who cannot hear the song must understand each sequence's fact from the \
pictures and labels alone. Give every step of the sequence at least one shot, following the words as they \
are sung, one clear idea per shot.
- Label every term a sequence sings: a labelled shot is framed on the one thing its label names, as the \
main subject in a side third, so the label is drawn in the open side. To explain parts (layers, organs, \
stages), keep one view and give each part its own labelled shot as it is sung.
- Film the real thing where it really is, never a picture of it on a screen, card, poster or window.
- A locked camera needs medium or high motion in the picture; a gentle change gets a slow pan, tilt or \
truck along with it, so no shot reads as a still.
- A slow change (frost spreading, a plant growing or wilting, the Sun turning) is a time-lapse: its \
end_state is clearly different from its start_state across the whole frame.
- A chorus sung again gets new shots: keep the chorus's place and look, move its idea further, and never \
repeat an earlier shot.
- Only people are characters. Cells, germs, molecules, animals, objects and ideas look and behave as they \
really do: never faces, eyes or expressions on them, no mascots, no invented metaphors.
- Pictures carry no text, labels, charts, numbers or digits; our code draws every word and figure. Write \
numbers as words only when they are counts of visible things.
- Describe what is there, never what is absent: no "no", "not", "without" or "never".
- No quotation marks anywhere.
- A lyric line stays in one place. The first picture shot in a new place is wide or medium_wide, so the \
viewer sees where we are; a sung close-up of the host does not count.
- Within a place, keep the screen direction and change shot size gradually.
- Lock the camera when the subject moves fast; keep the subject calm when the camera moves.
- Number shots keep the main subject in a side third; the figure is drawn in the open side.
- Write a shot for every slot you are given, sung slots included. In a sung shot the host sings to camera \
inside one of this video's walkable places, in the chosen outfit; declare a fitting walkable place if none \
exists. Code only adjusts the framing.
- In the same place, a shot starts where the shot before it ended: its start_state carries on from that \
end_state, so the cut joins up.
- A shot opens in the lighting of its place; a time-lapse may carry the light on within the clip, and the \
next shot is set in the place lit for where it ends. Hours never jump inside a plain clip.
- Follow the lesson's idea for each line; change it only when continuity needs it, and keep what it teaches.
- Use few places and come back to them: the video should feel like one connected film, not a slideshow.
- Each sequence shows its topic unfolding in its home place, shot after shot, toward the change the \
lesson gives it, moving to its zoom place and back only where the topic changes scale. Its first shot \
carries on from the shot before it; its last shot leads into the next sequence."""


def _slot_line(slot: Slot, lines: dict[int, LyricLine], plan: dict, sung: str | None = None) -> str:
    head = f"{slot.id} | {slot.start:.1f}-{slot.end:.1f}s ({slot.end - slot.start:.1f}s) | "
    if slot.kind == ShotKind.MUSIC:
        return head + "music, no lyrics"
    kind = {ShotKind.SUNG: "sung by the host to camera", ShotKind.NUMBER: f"number shot, code draws {slot.number_text}",
            ShotKind.ILLUSTRATE: "picture", ShotKind.UNCLEAR: "picture (the words may be misheard)"}[slot.kind]
    sung = f"sings: {sung}" if sung is not None else " / ".join(f"line {i}: {lines[i].text}" for i in slot.lines)
    ideas = " / ".join(f"teaches {plan[i].teaches}; idea: {plan[i].show}" for i in slot.lines)
    return head + f"{kind} | {sung} | {ideas}"


def _sequence_line(k: int, q: LessonSequence) -> str:
    steps = "; ".join(f"{i + 1}. {step}" for i, step in enumerate(q.steps)) or "none"
    return (f"Sequence {k + 1}, lines {q.first_line}-{q.last_line}: {q.topic} | place: {q.place} | change: "
            f"{q.change} | terms: {', '.join(q.terms) or 'none'}"
            + (f" | shown terms: {', '.join(q.shown_terms)}" if q.shown_terms else "")
            + f" | fact: {q.fact or 'none'} | steps: {steps}")


def user_prompt(plan: LessonPlan, lines: list[LyricLine], slots: list[Slot], band: BandRules, fmt: FormatRules,
                host: CastMember, wardrobe: dict[str, str], sung: dict[str, str] | None = None) -> str:
    """The whole film, given to every call: the lesson, the rules, every sequence and every slot. sung: the
    words sung during each slot, so a line split across shots shows each shot its own words."""
    by_index = {l.index: l for l in lines}
    lesson = {l.index: l for l in plan.lines}
    grouped = "\n".join(_sequence_line(k, q) + "".join(f"\n  {_slot_line(s, by_index, lesson, (sung or {}).get(s.id))}"
                                                       for s in slots if s.sequence == k)
                        for k, q in enumerate(plan.sequences))
    return (f"Topic: {plan.topic} ({plan.subject}). {plan.summary}\n"
            f"Must not teach: {'; '.join(plan.must_not_teach) or 'nothing listed'}.\n\n"
            f"Audience: grades {band.band}. Look: {band.look} Avoid: {'; '.join(band.avoid)}.\n"
            f"Limits: at most {band.max_places} places; at most {band.max_people_in_frame} people in frame; shot "
            f"size changes at most {band.max_size_jump} steps within a place; camera moves allowed: "
            f"{', '.join(band.camera_moves)}; camera speed at most {band.max_camera_speed}.\n\n"
            f"Video type: {fmt.type}. World: {fmt.world} On camera: {fmt.on_camera}\n"
            f"Shape: {fmt.shape} Keep the same: {fmt.keep_same} Camera: {fmt.camera}"
            + (f" A scale change of more than {fmt.max_scale_step} step between shots must carry the viewer: "
               "come in on a dissolve or match_cut, or let the shot before push in or pull back."
               if fmt.max_scale_step else "")
            + f"\n\nHost ({HOST}): {host.look} Size: {host.size}. Wardrobe kits: "
            + "; ".join(f"{kit}: {outfit}" for kit, outfit in wardrobe.items())
            + f". For a {fmt.type} video the usual kit is {fmt.host_outfit}; dress the host for each place."
            + ("" if fmt.type == VideoType.PERSONALISED_HERO else " The host appears only in sung shots.")
            + " Other people are welcome. A person who is clearly seen and comes back in a later shot is a cast "
              "member with kind person, listed in the subjects of every shot they are in. Passers-by, a crowd, or "
              "a hand holding something need no cast entry."
            + "\n\n"
            "Sequences and their slots, in order:\n" + grouped)


def header_prompt(user: str, sequences: list[LessonSequence]) -> str:
    return (f"{user}\n\nWrite the header for the whole video now: shape, cast, places, motifs, and "
            f"sequence_places with {len(sequences)} entries, one per sequence in order. No shots yet; they "
            "come one sequence at a time.")


def _where(header: BoardHeader, k: int) -> str:
    sp = header.sequence_places[k]
    return (sp.home + (f", zooming to {' and '.join(sp.zooms)}" if sp.zooms else "")
            + (f", also at other hours in {' and '.join(sp.times)}" if sp.times else ""))


def sequence_prompt(user: str, header: BoardHeader, done: list[ShotDraft], k: int, sequences: list[LessonSequence],
                    todo: list[str]) -> str:
    """One sequence's shots, written with the whole film in view: the fixed header, every shot so far, and
    where the next sequence goes."""
    after = (f"its last shot leads into sequence {k + 2}, {sequences[k + 1].topic}, in "
             f"{header.sequence_places[k + 1].home}" if k + 1 < len(sequences) else "its last shot ends the video")
    before = "\n".join(s.model_dump_json() for s in done) or "none yet; this sequence opens the video"
    return (f"{user}\n\nThe header, fixed for the whole video (use its ids; add no cast or places):\n"
            f"{header.model_dump_json()}\n\nShots written so far, in order:\n{before}\n\n"
            f"Now write sequence {k + 1} of {len(sequences)}, {sequences[k].topic}, in "
            f"{_where(header, k)}: one shot for each of the slots {', '.join(todo)}. Its first shot "
            f"carries on from the last shot above; {after}.")


WATCH_SYSTEM = """You watch a short educational video for school classes with the sound off: you cannot hear \
the song. You are given each shot as what it shows, with any word or figure drawn on screen. For each \
sequence, write in two plain sentences what a viewer learns from these pictures alone. Write only what the \
pictures and on-screen words really show; do not guess what a song might have said."""

REVIEW_SYSTEM = """You check an educational video against its lesson. For each sequence you are given the \
fact it must teach and what a viewer learned from it with the sound off. clear is true when that viewer \
understood the fact; otherwise clear is false and missing says, in one sentence, what the pictures fail to \
show."""


def watch_prompt(board: StoryboardDraft, slots: list[Slot], count: int) -> str:
    """The shots of every sequence as a viewer sees them: no lyrics, no lesson, only pictures and on-screen
    words."""
    places = {p.id: p.look for p in board.places}
    by_slot = {s.slot: s for s in board.shots}
    parts = []
    for k in range(count):
        shots = []
        for slot in (s for s in slots if s.sequence == k and s.id in by_slot):
            shot = by_slot[slot.id]
            screen = [t for t in (shot.label, slot.number_text) if t]
            shots.append(f"- {places.get(shot.place, shot.place)}: {shot.start_state} {shot.action} "
                         f"{shot.end_state}" + (f" On screen: {', '.join(screen)}." if screen else ""))
        parts.append(f"Sequence {k + 1}:\n" + "\n".join(shots))
    return "\n\n".join(parts)


def review_prompt(sequences: list[LessonSequence], watched: SoundOffWatch) -> str:
    learned = {w.sequence: w.learned for w in watched.sequences}
    return "\n\n".join(f"Sequence {k + 1}\nFact to teach: {q.fact}\nSteps planned: {'; '.join(q.steps)}\n"
                       f"Viewer learned with the sound off: {learned.get(k + 1, 'nothing')}"
                       for k, q in enumerate(sequences))


def rewrite_note(verdict: Verdict, learned: str, fact: str) -> str:
    return (f"\n\nThis sequence was written before, but watched with the sound off a viewer learns only: {learned} "
            f"It misses: {verdict.missing} Rewrite the whole sequence so its pictures and labels alone show its "
            f"fact: {fact}")
