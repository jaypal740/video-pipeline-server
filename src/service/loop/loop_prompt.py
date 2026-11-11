from src.schema.image.image_check_schema import Brief
from src.schema.storyboard.script_schema import AnimationScript, Drawing
from src.service.drawing.cutout import KEY_WORDS
from src.service.drawing.drawing_prompt import SAME_ONE, drawing_brief

# A loop is asked for at this length; the video model draws from the drawing back to the same drawing.
LOOP_SECONDS = 4.0


def loop_prompt(drawing: Drawing) -> str:
    """The drawing animated in place, ending as it began, still alone on its key colour so every frame cuts
    out the way the drawing did."""
    return (f"A flat 2D cartoon drawing animated in place. It shows: {drawing.look.rstrip('.')}. Its motion: "
            f"{(drawing.motion or '').rstrip('.')}. It moves smoothly and ends exactly as it began, so the clip "
            "repeats without a jump. It stays in the same spot at the same size, drawn in the same flat style with "
            "the same outlines and colours; nothing is added and nothing disappears. The background stays one flat "
            f"solid {KEY_WORDS} colour, unchanged, with no shadows. The camera does not move.")


def loop_brief(board: AnimationScript, drawing: Drawing, avoid: list[str] | None = None) -> Brief:
    """What the judge scores a loop's frame against: the drawing's own brief, told that the frame is the
    reference drawing part-way through its motion."""
    brief = drawing_brief(board, drawing, avoid)
    return Brief(description=f"{brief.description} This is one frame from a short animation of the reference "
                             f"drawing, in which it does this: {(drawing.motion or '').rstrip('.')}. Its pose may "
                             "differ from the reference as that motion needs.",
                 rules=brief.rules if SAME_ONE in brief.rules else brief.rules + [SAME_ONE], fatal=brief.fatal)
