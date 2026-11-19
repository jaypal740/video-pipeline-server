from src.schema.lesson.lesson_schema import BandRules, FormatRules, LessonPlan, LessonSequence, VideoType
from src.schema.song.song_analysis_schema import LyricLine
from src.schema.storyboard.storyboard_schema import (BoardHeader, CastMember, ShotDraft, ShotKind, Slot,
                                                     SoundOffWatch, StoryboardDraft, Verdict)
from src.service.storyboard.storyboard_rules import HOST, LIMITS

# Bumped whenever the prompt changes, so every run is boarded again with the new prompt.
PROMPT_VERSION = "storyboard.v20"

# The sentences that differ between a video with our fixed host and a song that comes with its own
# singer (JamWise), who takes the host's place: (host wording, singer wording).
_KINDS = ("kind person or thing", "kind person or thing")
_GIVEN = ("The host and the host's place are given; never declare them.",
          "Declare each real thing the lesson shows in cast as kind thing, with the look it really has. A thing "
          "that changes form is one cast member per form. Declare the song's singer too, the only person and the "
          "only one with a face, made for this song alone: id singer, kind person, star true, scale "
          "human; look: a girl of about ten, her skin, hair and clothes. Her clothes are a full, neat everyday outfit a "
          "schoolchild would wear: a top with sleeves, such as a t-shirt, hoodie or jacket, with trousers, leggings "
          "or a skirt, and shoes. They are bright, cheerful colours that children like, such as red, orange, pink, "
          "purple or yellow, with one strong main colour. No sleeveless tops or dresses and no bare shoulders. "
          "Nothing she wears is green or close to green, such as teal, lime or olive: she is filmed in front of a "
          "green screen. She is seen only in sung shots.")
_OUTFIT = (f"- host_outfit, on every place where the host appears: what the host wears there, fitting that place "
           "and the topic (beach clothes on a beach, a lab coat in a lab, a coat on a ship's deck), at most "
           f"{LIMITS['outfit']} words of clothes and colours. Name one of the wardrobe kits you are given, or write "
           "your own. Leave it out for places the host never appears in.\n", "")
_CHARACTERS = ("Only people are characters. Cells, germs, molecules, animals, objects and ideas look and behave as "
               "they really do: never faces, eyes or expressions on them, no mascots, no invented metaphors.",
               "Only the song's singer is a character. Cells, germs, molecules, animals, objects and ideas look and "
               "behave as they really do: never faces, eyes, limbs, clothes or expressions on them, no mascots, no "
               "invented metaphors. The lyrics may call a thing by a name the song gave it; never write that name: "
               "write the plain word for the real thing.")
_SINGER = ("the host", "the singer")
_FOLLOW = ("Follow the lesson's idea for each line; change it only when continuity needs it, and keep what it "
           "teaches.",
           "The picture slots of a sequence are cut by time, not by line: spread the sequence's steps over them in "
           "order, moving on to the next step when the words reach it. A step that has several slots is held "
           "across them, each shot showing it further along or closer, so the viewer has time to understand it. "
           "Each shot shows the step its sung words are about: what a line names is on screen while it is sung, "
           "and the thing it names is in the shot.")
# A song with its own singer has its sequences written at the same time, so the header says how each opens.
_OPENS = ("", " opens: one or two plain sentences on how things stand when the sequence opens: where each "
          "thing is and in which form, what has changed for good by then, where a thing that stays put is "
          "fixed, and whether it opens earlier or later than the sequence before it ends. The sequences are "
          "written at the same time, each from its own opens and the next one's, so together they tell one "
          "unbroken story: what one sequence leaves is what the next opens with.")
_SUNG = ("In a sung shot the host sings to camera inside one of this video's walkable places, in the chosen outfit; "
         "declare a fitting walkable place if none exists.",
         "In a sung shot the singer sings to camera at the side of the frame while what her line names is filmed "
         "behind her as it really looks. A line that names one thing shows that thing doing what the line says. A "
         "line that names several things shows them in the order they are sung, the first giving way to the next. "
         "A line that names the whole idea and no single thing shows the moment that best sums the idea up. Sung "
         "shots that follow each other each show a different thing or a different moment, so the lines of a chorus "
         "never share one picture. When a chorus is sung again, each line still shows what it names, at a new "
         "moment or from a new distance. A sung shot stands outside the story the picture shots tell. It is its "
         "own piece of real footage of what the line names, at the point in that thing's life the line is about, "
         "whatever the picture shots before and after it show, and it does not carry on from the shot before it. "
         "The thing the line names is itself in the shot, whole and as the main subject; what it leaves behind, "
         "such as an empty shell or a shed skin, never stands for it. Something visibly happens to it in the shot. "
         "Describe real things by what is seen: no character names, and no face, smile or expression on any animal "
         "or thing. A sung shot shows one change that fits its few seconds, never a time-lapse of several stages. "
         "The singer is never in the picture and is never mentioned in start_state, action or end_state: our code "
         "adds her. List the singer first in subjects, then the main thing shown, in a place "
         "of its sequence, and write start_state, action and end_state about what is filmed alone, as real footage "
         "would show it.")


def _system(kinds: str, given: str, outfit: str, characters: str, singer: str, follow: str, sung: str,
            opens: str) -> str:
    return f"""You storyboard short educational music videos for school classes. The song, its lesson plan \
and the shot timings already exist; you decide what each shot shows and how the camera films it. An image \
model draws each shot's opening frame from your words and a video model animates it, so write only what \
can be seen, in plain concrete words.

First declare, once for the whole video:
- shape: one sentence on how the video moves from start to end, at most {LIMITS["shape"]} words.
- cast: every person or thing seen in more than one shot. id in snake_case starting with a letter, unique \
across cast and places; {kinds}; look: \
the fixed appearance in at most {LIMITS["cast look"]} words (colours, materials, shapes, clothes), opening \
with the plain word for what it is (a leaf, a ship) and describing it on its own, with no character's name, \
no other cast member and not where it sits; size: its \
real size in at most {LIMITS["size"]} words; scale: the rung it is seen at (cosmic, planet, landscape, human, \
object, microscopic, molecular). A thing that is replaced by a new one (a second cable, a new ship) is a new \
cast member with its own id. {given}
- places: id, look in at most {LIMITS["place look"]} words that call things by what they are, never by a \
character's name, lighting in at most {LIMITS["lighting"]} words, \
scale (cosmic, planet, landscape, human, object, microscopic, molecular) and walkable: true when a person \
could really stand there (a deck, a shore, a lab), false for space, underwater, inside a cell.
{outfit}- motifs: visual codes kept all video, e.g. thing "antibiotic", rule "always a cyan liquid", the rule in at \
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
dusk, noon or dawn is set in a place lit for that hour.{opens}

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
- The music shots before the first line lead to what the first line names: the first opens on the place, \
and every one of them ends with that thing in view.
- A chorus sung again gets new shots: keep the chorus's place and look, move its idea further, and never \
repeat an earlier shot.
- {characters}
- Pictures carry no text, labels, charts, numbers or digits; our code draws every word and figure. Write \
numbers as words only when they are counts of visible things.
- Describe what is there, never what is absent: no "no", "not", "without" or "never".
- No quotation marks anywhere.
- A lyric line stays in one place. The first picture shot in a new place is wide or medium_wide, so the \
viewer sees where we are; a sung close-up of {singer} does not count.
- Within a place, keep the screen direction and change shot size gradually.
- Lock the camera when the subject moves fast; keep the subject calm when the camera moves.
- Number shots keep the main subject in a side third; the figure is drawn in the open side.
- Write a shot for every slot you are given, sung slots included. {sung} Code only adjusts the framing.
- In the same place, a shot starts where the shot before it ended: its start_state carries on from that \
end_state, so the cut joins up.
- A shot opens in the lighting of its place; a time-lapse may carry the light on within the clip, and the \
next shot is set in the place lit for where it ends. Hours never jump inside a plain clip.
- {follow}
- Use few places and come back to them: the video should feel like one connected film, not a slideshow.
- Each sequence shows its topic unfolding in its home place, shot after shot, toward the change the \
lesson gives it, moving to its zoom place and back only where the topic changes scale. Its first shot \
carries on from the shot before it; its last shot leads into the next sequence."""


SYSTEM = _system(*(part[0] for part in (_KINDS, _GIVEN, _OUTFIT, _CHARACTERS, _SINGER, _FOLLOW, _SUNG, _OPENS)))
STAR_SYSTEM = _system(*(part[1] for part in (_KINDS, _GIVEN, _OUTFIT, _CHARACTERS, _SINGER, _FOLLOW, _SUNG, _OPENS)))


def _slot_line(slot: Slot, lines: dict[int, LyricLine], plan: dict, sung: str | None = None,
               by_scene: bool = False) -> str:
    """by_scene: a picture slot shows its sequence's steps, so it is given the words sung but no line's idea."""
    head = f"{slot.id} | {slot.start:.1f}-{slot.end:.1f}s ({slot.end - slot.start:.1f}s) | "
    if slot.kind == ShotKind.MUSIC:
        return head + "music, no lyrics"
    kind = {ShotKind.SUNG: "sung by the host to camera", ShotKind.NUMBER: f"number shot, code draws {slot.number_text}",
            ShotKind.ILLUSTRATE: "picture", ShotKind.UNCLEAR: "picture (the words may be misheard)"}[slot.kind]
    sung = f"sings: {sung}" if sung is not None else " / ".join(f"line {i}: {lines[i].text}" for i in slot.lines)
    if by_scene and slot.kind in (ShotKind.ILLUSTRATE, ShotKind.UNCLEAR):
        return head + f"{kind} | {sung}"
    ideas = " / ".join(f"teaches {plan[i].teaches}; idea: {plan[i].show}" for i in slot.lines)
    return head + f"{kind} | {sung} | {ideas}"


def _sequence_line(k: int, q: LessonSequence) -> str:
    steps = "; ".join(f"{i + 1}. {step}" for i, step in enumerate(q.steps)) or "none"
    return (f"Sequence {k + 1}, lines {q.first_line}-{q.last_line}: {q.topic} | place: {q.place} | change: "
            f"{q.change} | terms: {', '.join(q.terms) or 'none'}"
            + (f" | shown terms: {', '.join(q.shown_terms)}" if q.shown_terms else "")
            + f" | fact: {q.fact or 'none'} | steps: {steps}")


def user_prompt(plan: LessonPlan, lines: list[LyricLine], slots: list[Slot], band: BandRules, fmt: FormatRules,
                host: CastMember | None, wardrobe: dict[str, str], sung: dict[str, str] | None = None,
                starred: bool = False) -> str:
    """The whole film, given to every call: the lesson, the rules, every sequence and every slot. sung: the
    words sung during each slot, so a line split across shots shows each shot its own words. starred: the
    song has its own singer, in place of the host."""
    who = ("The song has its own singer (where a rule above says host, that is the song's singer). She sings the "
           "sung shots and is in no other shot; every other shot shows real things only."
           if starred else
           f"Host ({HOST}): {host.look} Size: {host.size}. Wardrobe kits: "
           + "; ".join(f"{kit}: {outfit}" for kit, outfit in wardrobe.items())
           + f". For a {fmt.type} video the usual kit is {fmt.host_outfit}; dress the host for each place."
           + ("" if fmt.type == VideoType.PERSONALISED_HERO else " The host appears only in sung shots."))
    by_index = {l.index: l for l in lines}
    lesson = {l.index: l for l in plan.lines}
    grouped = "\n".join(_sequence_line(k, q)
                        + "".join(f"\n  {_slot_line(s, by_index, lesson, (sung or {}).get(s.id), starred)}"
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
            + f"\n\n{who}"
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
                    todo: list[str], alone: bool = False) -> str:
    """One sequence's shots, written with the whole film in view: the fixed header, every shot so far, and
    where the next sequence goes. alone: every sequence is written at the same time, so no shots are shown
    and the sequence opens its place itself."""
    after = (f"its last shot leads into sequence {k + 2}, {sequences[k + 1].topic}, in "
             f"{header.sequence_places[k + 1].home}" if k + 1 < len(sequences) else "its last shot ends the video")
    if k + 1 < len(sequences) and header.sequence_places[k + 1].opens:
        after += f", which opens like this: {header.sequence_places[k + 1].opens.rstrip('.')}"
    if alone:
        home = header.sequence_places[k].home
        before = ("it opens the video" if k == 0 else
                  f"it follows sequence {k}, {sequences[k - 1].topic}, in {header.sequence_places[k - 1].home}"
                  + ("" if header.sequence_places[k - 1].home == home else ", so its first shot comes in on a dissolve"))
        return (f"{user}\n\nThe header, fixed for the whole video (use its ids; add no cast or places):\n"
                f"{header.model_dump_json()}\n\nNow write sequence {k + 1} of {len(sequences)}, {sequences[k].topic}, "
                f"in {_where(header, k)}: one shot for each of the slots {', '.join(todo)}. The other sequences "
                f"are written at the same time, so their shots are not shown: {before}, and its first picture "
                f"shot in each place is wide or medium_wide; {after}."
                + (f" This sequence opens like this, and its first shot's start_state shows it: "
                   f"{header.sequence_places[k].opens.rstrip('.')}. When that is an earlier or later time than the "
                   "sequence before it ends, its first shot comes in on a dissolve."
                   if header.sequence_places[k].opens else ""))
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
