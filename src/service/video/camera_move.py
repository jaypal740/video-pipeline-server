from src.schema.storyboard.storyboard_schema import CameraMove, Direction, Shot, Speed

# A shot where only the camera moves is made by ffmpeg over its keyframe, not by a video model: nothing in
# the picture may change, which a model never guarantees. The picture is enlarged 4x first so the slow
# zoom and slide stay smooth. How far the camera travels, as a share of the frame, by its speed:
TRAVEL = {Speed.SLOW: 0.10, Speed.MEDIUM: 0.15, Speed.FAST: 0.22}
UPSCALE = 4


def move_filter(shot: Shot, frames: int, width: int, height: int, fps: int) -> str:
    """The ffmpeg filter that turns one still into `frames` frames of the shot's camera move."""
    amount = TRAVEL.get(shot.camera.speed, TRAVEL[Speed.SLOW])
    t = f"(on/{max(frames - 1, 1)})"
    centre_x, centre_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    move, way = shot.camera.move, shot.camera.direction
    if move == CameraMove.PUSH_IN:
        zoom, x, y = f"1+{amount}*{t}", centre_x, centre_y
    elif move == CameraMove.PULL_BACK:
        zoom, x, y = f"{1 + amount}-{amount}*{t}", centre_x, centre_y
    elif move in (CameraMove.PAN, CameraMove.TRUCK):
        zoom, y = f"{1 + amount}", centre_y
        x = f"(iw-iw/zoom)*(1-{t})" if way == Direction.LEFT else f"(iw-iw/zoom)*{t}"
    elif move in (CameraMove.TILT, CameraMove.PEDESTAL):
        zoom, x = f"{1 + amount}", centre_x
        y = f"(ih-ih/zoom)*{t}" if way == Direction.DOWN else f"(ih-ih/zoom)*(1-{t})"
    else:                                                       # a move ffmpeg cannot fake: a slow push in
        zoom, x, y = f"1+{amount / 2}*{t}", centre_x, centre_y
    return (f"scale={width * UPSCALE}:{height * UPSCALE}:force_original_aspect_ratio=increase,"
            f"crop={width * UPSCALE}:{height * UPSCALE},"
            f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={width}x{height}:fps={fps},setsar=1,format=yuv420p")
