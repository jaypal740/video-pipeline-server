from src.schema.image.image_check_schema import Brief
from src.schema.storyboard.script_schema import AnimationScript, Drawing, DrawingKind
from src.service.drawing.cutout import KEY_WORDS

# What a drawing made from another one is told: the reference is image 1.
SAME = ("Image 1 shows the same subject in another pose or state: take from it only the drawing style and whatever "
        "of its colours and features this drawing still shows. Draw the pose, shape and size this drawing "
        "describes, not the pose of image 1.")
NO_TEXT = "No letters, digits, labels, logos or watermarks anywhere in the picture"
NO_ONE = "No people, animals or characters anywhere in the picture"
WHOLE = "The picture shows one single subject, whole, with no part cut off by the edge of the frame"
FACE_SHOWS = "The character's face and mouth are clearly visible"
SAME_ONE = "The subject is the same one as in the reference picture, drawn in the same style and colours"
NO_FACE = "No cartoon face is added: no eyes or mouth drawn onto a thing that has none in real life"
ONLY = ("The picture shows only what the description names: nothing is added under, behind or beside the subject "
        "(no ground, plant, stand or prop the description does not mention)")


def aspect_of(drawing: Drawing) -> str:
    return "16:9" if drawing.kind == DrawingKind.BACKGROUND else "1:1"


def lead(board: AnimationScript) -> Drawing | None:
    """The performer drawing the others are drawn from: one that sings and is no pose of another, with the
    most poses of its own."""
    performers = [d for d in board.drawings if d.kind == DrawingKind.PERFORMER]
    mains = [d for d in performers if d.sings and d.of is None] or performers
    return max(mains, key=lambda d: sum(o.of == d.id for o in performers), default=None)


def reference_of(board: AnimationScript, drawing: Drawing) -> Drawing | None:
    """The drawing attached as a reference when this one is drawn: what it is a pose of, else the lead for a
    performer drawing, so the performer stays one character."""
    main = lead(board)
    if main is not None and drawing.id == main.id:
        return None
    if drawing.of:
        return next(d for d in board.drawings if d.id == drawing.of)
    return main if drawing.kind == DrawingKind.PERFORMER else None


def drawing_prompt(board: AnimationScript, drawing: Drawing, reference: Drawing | None) -> str:
    """One drawing in the video's style and palette: a backdrop that fills the frame, or one subject alone
    on the key colour for code to cut out."""
    look, palette = drawing.look.rstrip("."), ", ".join(board.palette)
    if drawing.kind == DrawingKind.BACKGROUND:
        return (f"{board.style}. {look}. A full-frame backdrop in pale, soft colours, with open calm space in the "
                "middle and nothing in the foreground, so drawings laid over it stand out. No characters, animals, "
                f"text, letters or numbers. Colours drawn from this palette, lightened: {palette}.")
    if drawing.kind == DrawingKind.PERFORMER:
        subject = f"The character is {(board.performer or '').rstrip('.')}. In this drawing: {look}."
        subject += " Its face is turned to the viewer and its mouth is clearly visible." if drawing.sings else ""
    elif drawing.kind == DrawingKind.ICON:
        subject = f"A flat symbol: {look}."
    else:
        subject = f"{look}. It is drawn as the real thing looks, with no cartoon face added."
    return (f"{board.style}. {subject}{f' {SAME}' if reference else ''} One single subject, whole and centred, with a "
            "clear margin all around it, drawn still with no motion lines or speed marks. Draw only what is "
            "described: add nothing under, behind or beside it. The background is one "
            f"flat solid pure {KEY_WORDS} colour filling the whole frame: no shadow, no ground, no gradient, "
            f"nothing else in the picture. Colours drawn from this palette: {palette}. No text, letters or numbers.")


def must_not_show(board: AnimationScript, drawing: Drawing, avoid: dict[str, str]) -> list[str]:
    """The wrong ideas the lesson forbids (`avoid`, by beat) in the beats that show this drawing. A backdrop
    or a symbol has none: it is in too many beats to carry any one beat's lesson."""
    if drawing.kind in (DrawingKind.BACKGROUND, DrawingKind.ICON):
        return []
    lines = [avoid.get(b.beat, "") for b in board.beats if any(drawing.id in (a.drawing, a.to) for a in b.actions)]
    return list(dict.fromkeys(line.rstrip(".") for line in lines if line))


def drawing_brief(board: AnimationScript, drawing: Drawing, avoid: list[str] | None = None) -> Brief:
    """What the judge scores a drawing against, and its hard rules. The judge is told this one drawing's
    look and the wrong ideas its beats forbid (`avoid`), not the performer's whole story: a performer who
    changes through the song (caterpillar, then butterfly) is in each drawing only what that drawing says.
    Text, an added face and an added thing fail a try on their own; a subject touching the frame's edge is
    caught by code when it is cut out. Only a pose of another drawing is compared with it."""
    if drawing.kind == DrawingKind.BACKGROUND:
        return Brief(description=f"{board.style}. A backdrop in pale, soft colours: {drawing.look}",
                     rules=[NO_TEXT, NO_ONE], fatal=[NO_TEXT, NO_ONE])
    rules, fatal = [NO_TEXT, WHOLE, ONLY], [NO_TEXT, ONLY]
    description = f"{board.style}. {drawing.look.rstrip('.')}."
    if drawing.kind == DrawingKind.PERFORMER:
        if drawing.sings:
            rules, fatal = rules + [FACE_SHOWS], fatal + [FACE_SHOWS]
    else:
        rules, fatal = rules + [NO_FACE], fatal + [NO_FACE]
    if drawing.of:
        rules = rules + [SAME_ONE]
    if avoid:
        description += f" It must not show any of these wrong ideas: {'; '.join(avoid)}."
    return Brief(description=f"{description} The subject is alone on a flat solid {KEY_WORDS} background, which is "
                             "what was asked for.", rules=rules, fatal=fatal)
