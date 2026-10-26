from enum import StrEnum


class StageName(StrEnum):
    INTAKE = "intake"
    LESSON = "lesson"
    BEATS = "beats"
    STORYBOARD = "storyboard"
    VISUAL_IDENTITY = "visual_identity"
    KEYFRAMES = "keyframes"
    VIDEO = "video"
    LIPSYNC = "lipsync"
    ASSEMBLY = "assembly"
    QC = "qc"


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
