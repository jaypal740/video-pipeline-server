from src.schema.storyboard.storyboard_schema import CastKind, CastMember, Place

# A plain ground the model never mistakes for a scene; the same for every sheet so sheets look like a set.
GROUND = "flat light grey"
STYLE_LINE = "The same drawing style as image 1; a different subject."

PERSON_VIEWS = ["front", "three_quarter"]
HOST_VIEWS = ["front", "three_quarter", "to_camera"]
THING_VIEWS = ["front"]
PLATE = "plate"

_VIEW = {
    "front": "seen from the front, standing straight with arms relaxed, the whole body in frame",
    "three_quarter": "seen from a three-quarter angle, the whole body in frame",
    "to_camera": "a medium close view from the chest up, facing the camera, mouth closed, calm",
    "thing": "the whole object in frame, seen from its most recognisable side",
}


def views_for(member: CastMember) -> list[str]:
    return PERSON_VIEWS if member.kind == CastKind.PERSON else THING_VIEWS


def sheet_prompt(band_look: str, member: CastMember, view: str, styled: bool) -> str:
    """A reference sheet for one cast member: the look, isolated on a plain ground, in one view."""
    what = "one person" if member.kind == CastKind.PERSON else "one thing"
    pose = _VIEW[view] if member.kind == CastKind.PERSON else _VIEW["thing"]
    text = (f"{band_look} A reference sheet of {what}, isolated on a {GROUND} ground, {pose}. "
            f"{member.look.rstrip('.')}. {member.size.rstrip('.')}.")
    return f"{text} {STYLE_LINE}" if styled else text


def plate_prompt(band_look: str, place: Place, styled: bool) -> str:
    """An empty wide view of a place, with its lighting, for every keyframe set there."""
    text = (f"{band_look} A wide empty view of a place, nobody in it, the middle left clear. "
            f"{place.look.rstrip('.')}, {place.lighting.rstrip('.')}.")
    return f"{text} {STYLE_LINE}" if styled else text


def sheet_questions(member: CastMember, view: str) -> list[str]:
    """What the image check asks about a sheet. Built from the look; no banned-word lists."""
    whole = "the whole body" if member.kind == CastKind.PERSON else "the whole object"
    found = [
        f"Exactly one subject, and nothing else but a plain single-colour ground: no scene, props or extra figures",
        f"The subject matches this description in colours, materials, shapes and clothing: {member.look}",
        f"{whole.capitalize()} is in frame and the view is {view.replace('_', '-')}",
        "No letters, digits, labels, logos or watermarks anywhere",
        "Nothing warped, duplicated or cut off; suitable for school children",
    ]
    if member.kind != CastKind.PERSON:
        found.append("The subject has no face, eyes, mouth or human expression; it looks like the real thing")
    return found


def plate_questions(place: Place) -> list[str]:
    return [
        "An empty place with no people, animals or characters in it",
        f"The place matches this description: {place.look}",
        f"The lighting matches this description: {place.lighting}",
        "The middle of the picture is clear enough to put a subject in later",
        "No letters, digits, labels, signs or watermarks anywhere",
        "Nothing warped or cut off; suitable for school children",
    ]
