from pydantic import BaseModel

from src.schema.image.image_check_schema import Brief
from src.schema.lesson.lesson_schema import BandRules
from src.schema.storyboard.storyboard_schema import CastKind, CastMember, Place


class Style(BaseModel):
    """How a band draws: the medium's name and its design features, from the band file."""

    name: str
    features: str
    medium: str = "illustration"                                # what a still is called in a prompt
    motion: str = "animation"                                   # ... and a clip

    @classmethod
    def of(cls, band: BandRules) -> "Style":
        return cls(name=band.style_name, features=band.style)


# What a song's singer sings about is filmed as it really looks, behind her.
REAL = Style(name="Realistic documentary", medium="photograph", motion="footage",
             features="true-to-life colours, textures and forms in natural light, with a softly blurred background")

# A song's own singer is drawn in 3D, whatever the band's look: she is cut out and laid over the film.
SINGER_STYLE = Style(name="A stylised 3D animated film",
                     features="soft rounded forms, large expressive eyes, detailed hair and fabric, natural child "
                              "proportions and warm even film lighting")
SINGER_VIEWS = ["front", "stage"]                               # who she is, and how she stands to be cut out
STAGE = "stage"

# Prompt rules learned on the first real run and from Google's and fal's guides: say what fills the frame,
# never what to leave out; no "reference sheet", "turnaround" or measurements (the model writes them on the
# picture); no "X or Y"; camera words plus the aspect ratio do the framing.
PERSON_VIEWS = ["front", "three_quarter"]
# A host's character sheet: the body from four sides, then the face for lip-sync and the emotions.
BODY_VIEWS = ["front", "three_quarter", "side", "back"]
FACE_VIEWS = ["to_camera", "smiling", "angry"]
HOST_VIEWS = BODY_VIEWS + FACE_VIEWS
THING_VIEWS = ["front"]
PLATE = "plate"
ASPECTS = {**{v: "3:4" for v in BODY_VIEWS}, **{v: "1:1" for v in FACE_VIEWS}, "thing": "1:1", PLATE: "16:9", "stage": "16:9"}

STYLE_LINE = ("Drawn in exactly the same illustration style as image 1, the same linework, shading and palette, "
              "showing a different subject.")
SAME_LINE = "The same person as in image 1, with the same face, hair and clothes."
# Text the code adds later must not be in the picture; writing that belongs to the place is set dressing.
NO_OVERLAY_TEXT = ("No captions, labels, arrows, charts, diagram marks or large readable words laid on the picture; "
                   "small writing that belongs to the place, such as a poster or a sign in the background, is fine")
NO_TEXT_ON_SUBJECT = "No letters, digits, labels, logos or watermarks on the subject or laid on the picture"

SAME_PERSON_LINE = ("The same person as in image 1, in the same pose and from the same angle, with the same face, "
                    "hair and body, now dressed as described.")

_STANCE = {"front": "standing facing the viewer, relaxed, arms at the sides",
           "three_quarter": "standing with the whole body turned 45 degrees to the right of the viewer, so the left "
                            "shoulder is closer to the viewer and the right ear is hidden, head turned the same way, "
                            "relaxed, arms at the sides",
           "side": "standing in full side profile facing to the left, only one eye and one ear visible, relaxed, "
                   "arms at the sides",
           "back": "standing seen from directly behind, the back of the head and the back of the clothes toward the "
                   "viewer, relaxed, arms at the sides"}
_FACE = {"to_camera": "a relaxed neutral expression, lips gently closed",
         "smiling": "a warm open smile, eyes bright",
         "angry": "an angry expression, brows drawn together and mouth set in a firm frown"}


_VIEW_WORDS = {"front": "seen from the front", "three_quarter": "turned three-quarters toward the camera",
               "side": "a full side profile, one eye and one ear visible", "back": "seen from directly behind, the "
               "back of the head and no face visible", "to_camera": "facing the camera, neutral",
               "smiling": "facing the camera, smiling", "angry": "facing the camera, angry",
               "stage": "from the waist up, facing the camera, with the raised hand on the right side of the picture"}


def views_for(member: CastMember) -> list[str]:
    if member.star and member.kind == CastKind.PERSON:
        return SINGER_VIEWS
    return PERSON_VIEWS if member.kind == CastKind.PERSON else THING_VIEWS


def aspect_for(member: CastMember | None, view: str) -> str:
    if view == PLATE or member is None or member.kind == CastKind.PERSON:
        return ASPECTS[view]
    return ASPECTS["thing"]


def sheet_prompt(style: Style, member: CastMember, view: str, reference: str | None) -> str:
    """One cast member alone on a plain ground, in one view, character first: who they are, then the design
    features, then the framing. reference: None (drawn from words alone), "style" (image 1 sets the drawing
    style) or "same" (image 1 is this very person)."""
    look = member.look.rstrip(".")
    look = look[0].lower() + look[1:]
    if member.kind != CastKind.PERSON:
        # the look alone says what it is: an id is a name ("fern" for a leaf), and the model draws the name
        text = (f"{style.name} illustration of {look}. Shown large, filling most of the frame, at the centre of a "
                f"plain pale-grey studio surface, seen from its most recognisable side, slightly from above. The "
                f"design features {style.features}. Even soft lighting with a faint contact shadow beneath; the "
                "rest of the frame is blank grey, right to its edges.")
    elif view == STAGE:                                          # a plain screen, so code can cut her out:
        screen = "blue" if "green" in look.lower() else "green"   # never the colour of what she wears
        text = (f"{style.name} character design of {look}, seen from the waist up in the middle of the frame, the "
                f"body turned a little toward the right side of the picture, the arm on the right side of the picture "
                f"raised with an open hand presenting the empty space to the right, "
                f"the face to the camera with a bright friendly smile and the mouth closed. The design features "
                f"{style.features}. The whole background is one perfectly flat, uniform, pure bright {screen}, "
                "evenly lit from edge to edge.")
    elif view in FACE_VIEWS:
        text = (f"{style.name} character design of {look}, a head-and-shoulders portrait facing the viewer directly "
                f"with {_FACE[view]}, eyes looking into the lens. The design features {style.features}. The face "
                "fills about two-thirds of the frame, centred on a plain pale-grey studio backdrop with soft even "
                "frontal lighting.")
    else:
        text = (f"{style.name} character design of {look}, {_STANCE[view]}. The design features {style.features}. "
                "Full figure from head to toe, centred on a plain pale-grey studio backdrop with a soft floor shadow.")
    if reference == "style":
        return f"{text} {STYLE_LINE}"
    if reference == "same":
        return f"{text} {SAME_LINE}"
    if reference == "same_person":
        return f"{text} {SAME_PERSON_LINE}"
    return text


def plate_prompt(style: Style, place: Place, reference: str | None) -> str:
    """An empty wide view of a place, with its lighting, for every keyframe set there."""
    lighting = place.lighting.rstrip(".")
    text = (f"{style.name} illustration of an empty place, deserted and still: {place.look.rstrip('.')}. "
            f"{lighting[0].upper()}{lighting[1:]}. The design features {style.features}. A wide establishing view "
            "at eye level; the middle of the frame is open and uncluttered, and the painting covers the whole "
            "frame right to its edges.")
    return f"{text} {STYLE_LINE}" if reference == "style" else text


def sheet_brief(member: CastMember, view: str) -> Brief:
    """What the judge scores a sheet against, and the hard rules. The description names the kind of
    thing, its shape, colours and materials; small details are for the score, not for a rule."""
    if member.kind == CastKind.PERSON:
        angle = _VIEW_WORDS.get(view, view.replace("_", " "))
        shown = ("the face, hair and top garment" if view in FACE_VIEWS else "hair, clothes and footwear as far as "
                 "this angle shows them (a back view shows no face; a side view shows one side)")
        description = (f"A single person, {angle}: {member.look}. Judge {shown}. Ignore pose, expression, exact "
                       "skin shade and how garments are worn.")
        whole = ("The face and shoulders are in frame" if view in FACE_VIEWS else "The head, arms and body down to "
                 "the waist are in frame" if view == STAGE else "The whole body from head to feet is in frame")
        fatal = ["Exactly one person on a plain single-colour ground (a soft floor shadow is fine): no scene, props or extra figures",
                 NO_TEXT_ON_SUBJECT]
        rules = [f"{whole} and the view is: {angle}", *fatal,
                 "Nothing warped or duplicated; suitable for school children"]
    else:
        description = (f"{member.look.rstrip('.')}. Judge the kind of object, how many there are as the "
                       "description says, its overall shape, its main colours and materials; small details "
                       "such as edge sharpness, reflections and texture only shade the score.")
        fatal = [NO_TEXT_ON_SUBJECT]
        rules = ["As many items as the description says (one, unless it describes several of the same item) on a "
                 "plain single-colour ground (a soft contact shadow is fine): no scene and nothing it does not describe",
                 "The whole subject is in frame", *fatal,
                 *(["The subject has no face, eyes, mouth or human expression; it looks like the real thing"]
                   if member.kind == CastKind.THING else []),    # a song's own character is drawn with its face
                 "Nothing warped and no copies beyond what the description says; suitable for school children"]
    return Brief(description=description, rules=rules, fatal=fatal)


def plate_brief(place: Place) -> Brief:
    """What the judge scores an empty place plate against. A wide view is cropped by its frame by nature,
    so cropping is never a rule; the scene and its light are what count."""
    return Brief(description=(f"An empty wide view of a place: {place.look}. Light roughly {place.lighting}. Judge "
                              "whether this is that place, with that kind of light, as a backdrop for later shots."),
                 rules=["An empty place with no people, animals or characters in it", NO_OVERLAY_TEXT,
                        "Nothing warped or broken; suitable for school children"],
                 fatal=["An empty place with no people, animals or characters in it", NO_OVERLAY_TEXT])
