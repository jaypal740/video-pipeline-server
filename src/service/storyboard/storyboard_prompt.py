from src.schema.lesson.lesson_schema import BandRules, FormatRules, LessonPlan, VideoType
from src.schema.song.song_analysis_schema import LyricLine
from src.schema.storyboard.storyboard_schema import CastMember, Place, ShotKind, Slot, StoryboardDraft
from src.service.storyboard.storyboard_rules import HOST, HOST_PLACE, LIMITS

# Bumped whenever the prompt changes, so every run is boarded again with the new prompt.
PROMPT_VERSION = "storyboard.v5"

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
and scale (cosmic, planet, landscape, human, object, microscopic, molecular).
- motifs: visual codes kept all video, e.g. thing "antibiotic", rule "always a cyan liquid", the rule in at \
most {LIMITS["motif rule"]} words. A motif is about appearance only (colour, material, shape), never about \
where a thing is or what it does. Write each motif's rule into the look of the cast member or place it \
belongs to, so every picture keeps it.

Then one shot per slot you are given (skip slots marked as repeats; code copies them):
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
- Sung shots are the host singing to camera from the host's own place ({HOST_PLACE}, given); code sets their \
framing and place.
- Lighting stays the same within a shot; show time passing with a cut or a dissolve, never inside one clip.
- Follow the lesson's idea for each line; change it only when continuity needs it, and keep what it teaches.
- Use few places and come back to them: the video should feel like one connected film, not a slideshow."""


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


def user_prompt(plan: LessonPlan, lines: list[LyricLine], slots: list[Slot], band: BandRules, fmt: FormatRules,
                host: CastMember, host_place: Place) -> str:
    by_index = {l.index: l for l in lines}
    lesson = {l.index: l for l in plan.lines}
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
            + f"\n\nHost ({HOST}): {host.look} Size: {host.size}. Sings from {HOST_PLACE}: {host_place.look}, "
            f"{host_place.lighting}."
            + ("" if fmt.type == VideoType.PERSONALISED_HERO else " The host appears only in sung shots.")
            + "\n\n"
            "Slots, in order:\n" + "\n".join(_slot_line(s, by_index, lesson) for s in slots))


def patch_prompt(user: str, draft: StoryboardDraft, problems: dict[str, list[str]]) -> str:
    listed = "\n".join(f"- {slot_id}: " + "; ".join(found) for slot_id, found in sorted(problems.items()))
    return (f"{user}\n\nThe storyboard so far:\n{draft.model_dump_json()}\n\n"
            "Rewrite only these shots, keeping the cast, places and every other shot as they are. Return just "
            f"these shots, each with every problem fixed:\n{listed}")
