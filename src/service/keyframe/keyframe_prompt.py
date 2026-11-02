from src.schema.storyboard.storyboard_schema import CastKind, CastMember, Place, Shot, ShotKind, ShotSize
from src.schema.image.image_check_schema import Brief, Correction
from src.service.sheet.sheet_prompt import NO_OVERLAY_TEXT, Style

# A retry's prompt is the failed prompt plus a short correction written from the judge's findings.
CORRECTION_WORDS = 60
CORRECTION = """You turn a judge's findings about a generated picture into a short correction for the image \
model that drew it, at most 60 words. You are given the prompt the picture was drawn from, the opening moment \
it must show and what the judge found wrong. Say what the picture should show instead, in positive words that \
move it toward the opening moment: never write "no", "not", "without" or "remove", because the image model \
draws what is named. Fix only what the findings say is wrong. The opening moment and the place describe the \
world the picture is set in, even when it differs from everyday experience (a black sky in full sunlight \
where there is no air): when a finding asks for something they do not, keep what they ask for."""

# Which sheet view a subject is drawn from: the front for wide framings, three-quarter when closer.
_FRONT_SIZES = {ShotSize.WIDE, ShotSize.MEDIUM_WIDE}
_SIZE_WORDS = {ShotSize.WIDE: "Wide shot", ShotSize.MEDIUM_WIDE: "Medium-wide shot", ShotSize.MEDIUM: "Medium shot",
               ShotSize.MEDIUM_CLOSE: "Medium close shot", ShotSize.CLOSE: "Close shot",
               ShotSize.EXTREME_CLOSE: "Extreme close shot"}
_ANGLE_WORDS = {"eye": "eye level", "low": "low angle", "high": "high angle", "overhead": "overhead"}
_POSITION_WORDS = {"left_third": "the main subject in the left third", "centre": "the main subject centred",
                   "right_third": "the main subject in the right third"}


def view_for(member: CastMember, shot: Shot) -> str:
    if member.kind != CastKind.PERSON:                          # a thing has one view, whatever the shot
        return "front"
    if shot.kind == ShotKind.SUNG:
        return "to_camera"
    return "front" if shot.size in _FRONT_SIZES else "three_quarter"


def keyframe_prompt(style: Style, shot: Shot, place: Place, subjects: list[CastMember]) -> str:
    """The frozen opening moment of one shot, at most about 90 words: style, framing, the place (image 1),
    the subjects named by their reference image, then start_state. The action and the end never appear."""
    named = ", ".join(f"{m.id.replace('_', ' ')} is image {i}" for i, m in enumerate(subjects, start=2))
    lines = [f"{style.name} illustration, {_SIZE_WORDS[shot.size].lower()}, {_ANGLE_WORDS[shot.angle]}.",
             f"The place is image 1: {place.look.rstrip('.')}, {place.lighting.rstrip('.')}."]
    if subjects:
        lines.append(f"{named[0].upper()}{named[1:]}, {_POSITION_WORDS[shot.position]}.")
    lines.append(shot.start_state.rstrip(".") + ".")
    lines.append(f"The design features {style.features}. Keep every design exactly as in the references.")
    return " ".join(lines)


def correction_request(prompt: str, moment: str, findings: list[str]) -> str:
    """What the correction writer reads: the failed prompt (the place and its light), the opening moment
    and what the judge found wrong."""
    return (f"The image model was asked for:\n{prompt}\n\nThe opening moment it must show:\n{moment}\n\n"
            "What the judge found wrong:\n" + "\n".join(f"- {f}" for f in findings))


def check_correction(correction: Correction) -> list[str]:
    words = correction.text.lower().replace(",", " ").replace(".", " ").split()
    problems = [f"{len(words)} words, at most {CORRECTION_WORDS}"] if len(words) > CORRECTION_WORDS else []
    negative = [w for w in ("no", "not", "without", "remove") if w in words]
    return problems + ([f"uses {negative}; say what should be seen instead"] if negative else [])


def ref_label(kind: str, ref_id: str) -> str:
    """How a reference image is labelled for the judge, so the rules can name it instead of numbering it. The
    kind keeps a place and a cast member with similar ids (air_molecules_view, air_molecules) apart."""
    return f"the {kind} {ref_id}"


def keyframe_brief(shot: Shot, place: Place, subjects: list[CastMember]) -> Brief:
    """What the judge scores a keyframe against. Who and what the shot shows comes from its opening moment,
    which the judge reads and writes its own checks from; the rules here hold for every shot."""
    who = ", ".join(m.id.replace("_", " ") for m in subjects) or "none, the moment and the place carry the shot"
    description = (f"The opening moment above, before anything has happened. The place is {place.look}, "
                   f"{place.lighting}; where the opening moment differs from the place's usual look, the moment "
                   f"wins. Cast members with a reference image: {who}. Framing: a "
                   f"{_SIZE_WORDS[shot.size].lower()} with {_POSITION_WORDS[shot.position]}.")
    rules = [f"The setting is the same place as in the image labelled '{ref_label('place', place.id)}': same ground, "
             "sky, landmarks and light; things the opening moment adds to it or changes in it are fine"]
    for m in subjects:                                          # the judge finds each reference by its label
        rules.append(f"The {m.id.replace('_', ' ')} is the same one as in the image labelled '{ref_label('cast member', m.id)}': "
                     f"same {'face, hair and clothes' if m.kind == CastKind.PERSON else 'shape, colours and materials'}")
    fatal = [NO_OVERLAY_TEXT, "No person or character beyond the ones the opening moment and the cast list name"]
    rules += fatal + ["No face, eyes, mouth or human expression on anything that is not a person",
                      "Nothing warped, melted together or broken; suitable for school children"]
    return Brief(description=description, rules=rules, fatal=fatal, moment=shot.start_state)


# A chained picture is an edit of the picture before it: the previous shot's last frame for a linked opening,
# the shot's own opening for its ending. The judge finds that picture by these labels.
LINK_LABEL = "the previous shot's last frame"
START_LABEL = "the start keyframe"


def linked_prompt(style: Style, shot: Shot, subjects: list[CastMember]) -> str:
    """The opening of a shot that carries on from the shot before in the same place: image 1 is where that
    shot ended, kept as it is apart from the new framing and what the opening moment says."""
    named = ", ".join(f"{m.id.replace('_', ' ')} is image {i}" for i, m in enumerate(subjects, start=2))
    lines = [f"{style.name} illustration, {_SIZE_WORDS[shot.size].lower()}, {_ANGLE_WORDS[shot.angle]}.",
             "Image 1 is the last frame of the shot before, in the same place: keep its place, light, designs and "
             "the state of everything in it, framed anew."]
    if subjects:
        lines.append(f"{named[0].upper()}{named[1:]}, {_POSITION_WORDS[shot.position]}.")
    lines.append(shot.start_state.rstrip(".") + ".")
    lines.append(f"The design features {style.features}. Keep every design exactly as in the references.")
    return " ".join(lines)


def end_prompt(style: Style, shot: Shot) -> str:
    """The last frame of a shot, drawn from its opening (image 1): the same framing and designs, with the
    shot's end_state. The video model then animates from one to the other."""
    return (f"{style.name} illustration. Image 1 is the opening frame of this shot. Draw the same shot a few seconds "
            f"later: the same place, the same camera framing, the same designs, colours and line style. Now: "
            f"{shot.end_state.rstrip('.')}. Keep everything that does not change exactly as in image 1.")


def linked_brief(shot: Shot, place: Place, subjects: list[CastMember]) -> Brief:
    """A linked opening is judged like any keyframe, but its setting and state come from the picture before."""
    brief = keyframe_brief(shot, place, subjects)
    brief.rules[0] = (f"The place, light, designs and the state of things are the same as in the image labelled "
                      f"'{LINK_LABEL}'; only the framing and what the opening moment says may differ")
    return brief


def end_brief(shot: Shot) -> Brief:
    """An ending is judged against end_state, in the framing and style of the shot's own opening."""
    fatal = [NO_OVERLAY_TEXT, f"No person or character that is not in the image labelled '{START_LABEL}'"]
    return Brief(description=(f"The last frame of a short shot whose first frame is the image labelled '{START_LABEL}'. "
                              f"The action between them: {shot.action} Judge whether this picture shows the ending "
                              "below, in the same place, framing and style as the start keyframe."),
                 rules=[f"Same place, camera framing, line style and colours as the image labelled '{START_LABEL}'",
                        "Every thing and person kept from the start keyframe has the same design", *fatal,
                        "No face, eyes, mouth or human expression on anything that is not a person"],
                 fatal=fatal, moment=f"(This is the ending of the shot, not its opening.) {shot.end_state}")
