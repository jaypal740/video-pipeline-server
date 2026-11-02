import re

from src.schema.image.image_check_schema import Brief
from src.schema.storyboard.storyboard_schema import (CameraMove, CastKind, CastMember, Direction, Motion, Place,
                                                     ScreenDirection, Shot, ShotKind, Speed)
from src.service.sheet.sheet_prompt import NO_OVERLAY_TEXT, Style

# A clip prompt is built by code from the storyboard, never rewritten by a model: the action, the ending,
# the camera in spatial words, under about 60 words. What was learned on the old pipeline: short prompts
# with visible verbs move more; "hold still" clauses freeze the clip; direction words like "clockwise"
# send motion the wrong way, so moves are named by where the picture goes.
PROMPT_WORDS = 60
NEGATIVE = "text, captions, subtitles, watermark, extra people, extra characters, distortion, blur"
# Added after the shot's own words. On the 2026-11-02 style test it removed H3's glow and flares and its
# overacting faces; a mouth stays closed because only the lip-sync stage gives anyone words.
STYLE_LINE = ("Faces stay calm with the mouth closed unless the action says otherwise. Flat, even cartoon colours "
              "with crisp outlines and soft, even light.")

_MOVE = {CameraMove.STATIC: "the camera holds its framing", CameraMove.PAN: "the camera pans {d}",
         CameraMove.TILT: "the camera tilts {d}", CameraMove.PUSH_IN: "the camera pushes in slowly",
         CameraMove.PULL_BACK: "the camera pulls back", CameraMove.TRUCK: "the camera moves {d} alongside",
         CameraMove.PEDESTAL: "the camera rises {d}", CameraMove.ORBIT: "the camera circles {d} around the subject",
         CameraMove.FOLLOW: "the camera follows the subject"}
_DIRECTION = {Direction.LEFT: "to the left", Direction.RIGHT: "to the right", Direction.UP: "upward",
              Direction.DOWN: "downward", Direction.TOWARD: "toward the subject", Direction.AWAY: "away from the subject"}
_SPEED = {Speed.SLOW: "slowly", Speed.MEDIUM: "steadily", Speed.FAST: "quickly"}
_MOTION = {Motion.LOW: "gentle, small movement", Motion.MEDIUM: "steady, clear movement",
           Motion.HIGH: "lively, large movement"}
_SCREEN = {ScreenDirection.LEFT_TO_RIGHT: "moving left to right across the frame",
           ScreenDirection.RIGHT_TO_LEFT: "moving right to left across the frame",
           ScreenDirection.TOWARD: "coming toward the camera", ScreenDirection.AWAY: "moving away from the camera"}
_QUOTE = re.compile(r"[\"“”]")
_DIGIT = re.compile(r"\d+")


def camera_words(shot: Shot) -> str:
    words = _MOVE[shot.camera.move].format(d=_DIRECTION.get(shot.camera.direction, ""))
    words = " ".join(words.split())                             # a move with no direction leaves no double space
    if shot.camera.move != CameraMove.STATIC and shot.camera.speed and "slowly" not in words:
        words += f", {_SPEED[shot.camera.speed]}"
    return words


def clip_prompt(style: Style, shot: Shot, place: Place, subjects: list[CastMember], host_look: str | None) -> str:
    """What the video model is told, from the storyboard: the action, how it ends, the camera, how much
    moves. A sung shot asks for a calm host with the lips together, so a later lip-sync pass has a still
    mouth to animate. Quotes and digits never appear: both make the model draw text."""
    if shot.kind == ShotKind.SUNG:
        who = host_look or "the person"
        parts = [f"{who[0].upper()}{who[1:]} faces the camera with the lips together and a calm, friendly face, "
                 "breathing gently, the head and shoulders moving a little.", f"{camera_words(shot).capitalize()}."]
    else:
        parts = [f"{shot.action.rstrip('.')}.", f"By the last frame, "
                 f"{shot.end_state.rstrip('.')[0].lower()}{shot.end_state.rstrip('.')[1:]}."]
        if "camera" in shot.action.lower():                    # the board already said what the camera does
            parts.append(f"{_MOTION[shot.motion][0].upper()}{_MOTION[shot.motion][1:]}.")
        else:
            parts.append(f"{camera_words(shot).capitalize()}, {_MOTION[shot.motion]}.")
        if shot.screen_direction in _SCREEN:
            parts.append(f"The subject is {_SCREEN[shot.screen_direction]}.")
        parts.append(STYLE_LINE)
    return f"{style.name} animation. {clean(' '.join(parts))}"   # the style name keeps its own digits (2D)


def clean(text: str) -> str:
    return " ".join(_DIGIT.sub("", _QUOTE.sub("", text)).split())


def clip_brief(shot: Shot, place: Place, subjects: list[CastMember]) -> Brief:
    """What the judge scores a clip against, from three frames of it: the start, the middle and the end."""
    who = ", ".join(m.id.replace("_", " ") for m in subjects) or "none, the place carries the shot"
    description = ("Images 1, 2 and 3 are the first, middle and last frames of one short clip, in order; image 4 is "
                   "the keyframe it was animated from. Judge the clip as a whole: the first frame shows the opening "
                   f"moment, the middle shows the action under way, the last shows the ending. The place is {place.look}, "
                   f"{place.lighting}. Cast with a reference: {who}. Small camera drift is never a fault.")
    fatal = [NO_OVERLAY_TEXT, "No person or character beyond the ones the opening moment and the cast list name"]
    rules = ["Every character and thing keeps the design of the keyframe: same face, hair, clothes, shape and colours",
             *fatal, "Nothing warped, melted together or broken; suitable for school children"]
    moment = f"Opening: {shot.start_state} Action: {shot.action} Ending: {shot.end_state}"
    return Brief(description=description, rules=rules, fatal=fatal, moment=moment)


def host_look_for(subjects: list[CastMember]) -> str | None:
    return next((m.look for m in subjects if m.kind == CastKind.PERSON), None)
