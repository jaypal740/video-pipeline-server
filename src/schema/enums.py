from enum import StrEnum


class StageName(StrEnum):
    INTAKE = "intake"
    LESSON = "lesson"
    BEATS = "beats"
    STORYBOARD = "storyboard"
    VISUAL_IDENTITY = "visual_identity"
    LOOPS = "loops"
    RENDER = "render"
    KEYFRAMES = "keyframes"
    VIDEO = "video"
    LIPSYNC = "lipsync"
    ASSEMBLY = "assembly"
    QC = "qc"


class FilmType(StrEnum):
    """What a run makes. narration and music_video run the same realistic stages (a spoken track with the
    grade's host; a JamWise song with its own singer); 2d runs the two_d stages (drawings, loops, code render)."""

    NARRATION = "narration"
    MUSIC_VIDEO = "music_video"
    TWO_D = "2d"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_GATE = "waiting_gate"
    DONE = "done"
    FAILED = "failed"


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


class AssetKind(StrEnum):
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    DATA = "data"
