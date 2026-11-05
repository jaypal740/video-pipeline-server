from src.schema.lesson.lesson_schema import BandRules, FormatRules, LessonPlan, LessonSequence, VideoType
from src.schema.song.song_analysis_schema import LyricLine
from src.schema.storyboard.storyboard_schema import BoardHeader, CastMember, ShotDraft, ShotKind, Slot
from src.service.storyboard.storyboard_rules import HOST, LIMITS

# Bumped whenever the prompt changes, so every run is boarded again with the new prompt.
PROMPT_VERSION = "storyboard.v10"

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
shown in; sequences the lesson sets in the same place share a home. zoom, when the topic needs it: one \
more place that shows the same topic at another scale, such as the air up close above a hilltop or the \
atoms inside the Sun's core, so small things are seen at their own size and far views from where they \
really are. Leave zoom empty when the home shows everything.

Then, one sequence at a time, one shot per slot of that sequence (skip slots marked as repeats; code \
copies them):
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
- moves: subject when something in the picture moves or changes; camera when everything in the picture \
stays as it is and only the camera moves over it.

Rules:
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
- Lighting stays the same within a shot; show time passing with a cut or a dissolve, never inside one clip.
- Follow the lesson's idea for each line; change it only when continuity needs it, and keep what it teaches.
- Use few places and come back to them: the video should feel like one connected film, not a slideshow.
- Each sequence shows its topic unfolding in its home place, shot after shot, toward the change the \
lesson gives it, moving to its zoom place and back only where the topic changes scale. Its first shot \
carries on from the shot before it; its last shot leads into the next sequence."""


def _slot_line(slot: Slot, lines: dict[int, LyricLine], plan: dict) -> str:
    head = f"{slot.id} | {slot.start:.1f}-{slot.end:.1f}s ({slot.end - slot.start:.1f}s) | "
    if slot.repeat_of:
        return head + f"repeat of {slot.repeat_of}: code copies it, write nothing"
    if slot.kind == ShotKind.MUSIC:
        return head + "music, no lyrics"
    kind = {ShotKind.SUNG: "sung by the host to camera", ShotKind.NUMBER: f"number shot, code draws {slot.number_text}",
            ShotKind.ILLUSTRATE: "picture", ShotKind.UNCLEAR: "picture (the words may be misheard)"}[slot.kind]
    sung = " / ".join(f"line {i}: {lines[i].text}" for i in slot.lines)
    ideas = " / ".join(f"teaches {plan[i].teaches}; idea: {plan[i].show}" for i in slot.lines)
    return head + f"{kind} | {sung} | {ideas}"


def _sequence_line(k: int, q: LessonSequence) -> str:
    return (f"Sequence {k + 1}, lines {q.first_line}-{q.last_line}: {q.topic} | place: {q.place} | change: "
            f"{q.change} | terms: {', '.join(q.terms) or 'none'}")


def user_prompt(plan: LessonPlan, lines: list[LyricLine], slots: list[Slot], band: BandRules, fmt: FormatRules,
                host: CastMember, wardrobe: dict[str, str]) -> str:
    """The whole film, given to every call: the lesson, the rules, every sequence and every slot."""
    by_index = {l.index: l for l in lines}
    lesson = {l.index: l for l in plan.lines}
    grouped = "\n".join(_sequence_line(k, q) + "".join(f"\n  {_slot_line(s, by_index, lesson)}"
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
    return sp.home + (f", zooming to {sp.zoom}" if sp.zoom else "")


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
