from src.schema.storyboard.storyboard_schema import CastKind, CastMember, Place, Shot, ShotKind, ShotSize
from src.schema.image.image_check_schema import Brief
from src.service.sheet.sheet_prompt import Style

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


def ref_label(kind: str, ref_id: str) -> str:
    """How a reference image is labelled for the judge, so the rules can name it instead of numbering it. The
    kind keeps a place and a cast member with similar ids (air_molecules_view, air_molecules) apart."""
    return f"the {kind} {ref_id}"


def keyframe_brief(shot: Shot, place: Place, subjects: list[CastMember]) -> Brief:
    """What the judge scores a keyframe against. Who and what the shot shows comes from its opening moment,
    which the judge reads and writes its own checks from; the rules here hold for every shot."""
    who = ", ".join(m.id.replace("_", " ") for m in subjects) or "none, the moment and the place carry the shot"
    description = (f"The opening moment above, before anything has happened. The place is {place.look}, "
                   f"{place.lighting}. Cast members with a reference image: {who}. Framing: a "
                   f"{_SIZE_WORDS[shot.size].lower()} with {_POSITION_WORDS[shot.position]}.")
    rules = [f"The setting is the same place as in the image labelled '{ref_label('place', place.id)}': same ground, "
             "sky, landmarks and light; things the opening moment adds to it are fine"]
    for m in subjects:                                          # the judge finds each reference by its label
        rules.append(f"The {m.id.replace('_', ' ')} is the same one as in the image labelled '{ref_label('cast member', m.id)}': "
                     f"same {'face, hair and clothes' if m.kind == CastKind.PERSON else 'shape, colours and materials'}")
    rules += ["No letters, digits, labels, arrows, charts or diagram marks anywhere",
              "No face, eyes, mouth or human expression on anything that is not a person",
              "Nothing warped, melted together or broken; suitable for school children"]
    return Brief(description=description, rules=rules, moment=shot.start_state)
