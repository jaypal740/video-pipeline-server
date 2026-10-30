from src.schema.storyboard.storyboard_schema import CastKind, CastMember, Place, Shot, ShotKind, ShotSize
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
    if shot.kind == ShotKind.SUNG:
        return "to_camera"
    if member.kind != CastKind.PERSON:
        return "front"
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


def keyframe_questions(shot: Shot, place: Place, subjects: list[CastMember]) -> list[str]:
    people = [m for m in subjects if m.kind == CastKind.PERSON]
    things = [m for m in subjects if m.kind != CastKind.PERSON]
    count = (f"exactly {len(people)} {'person' if len(people) == 1 else 'people'}" if people else "no people")
    found = [f"Count first: {count}, and these things once each: {', '.join(m.id.replace('_', ' ') for m in things) or 'none'}; "
             "no extra people, creatures or characters"]
    for i, m in enumerate(subjects, start=2):
        found.append(f"The {m.id.replace('_', ' ')} matches its reference sheet (image {i}) in colours, "
                     f"{'clothing, ' if m.kind == CastKind.PERSON else ''}materials and shape")
    found += [
        f"The place matches its reference plate (image {len(subjects) + 2}) with the same lighting",
        f"This is the opening moment before anything has happened, matching: {shot.start_state}",
        f"The framing is about a {_SIZE_WORDS[shot.size].lower()} with {_POSITION_WORDS[shot.position]}",
        "No letters, digits, labels, arrows, charts or diagram marks anywhere",
        "Nothing warped, duplicated or cut off; suitable for school children",
    ]
    if things:
        found.append("No face, eyes, mouth or human expression on anything that is not a person")
    return found
