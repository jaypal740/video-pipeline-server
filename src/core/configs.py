from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    fal_key: str | None = Field(default=None, validation_alias="FAL_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    claude_api_key: str | None = Field(default=None, validation_alias="CLAUDE_API_KEY")
    db_url: str = Field(default="sqlite:///data/pipeline.db", validation_alias="PIPELINE_DB_URL")
    asset_dir: str = Field(default="data/assets", validation_alias="PIPELINE_ASSET_DIR")
    device: str = Field(default="cpu", validation_alias="PIPELINE_DEVICE")
    spend_cap_usd: float = Field(default=25.0, validation_alias="PIPELINE_SPEND_CAP_USD")
    log_level: str = Field(default="INFO", validation_alias="PIPELINE_LOG_LEVEL")
    log_dir: str | None = Field(default="data/logs", validation_alias="PIPELINE_LOG_DIR")
    lesson_model: str = Field(default="claude-opus-5-5", validation_alias="PIPELINE_LESSON_MODEL")
    storyboard_model: str = Field(default="claude-opus-5-5", validation_alias="PIPELINE_STORYBOARD_MODEL")
    sheet_model: str = Field(default="nano-banana-2.1", validation_alias="PIPELINE_SHEET_MODEL")
    host_model: str = Field(default="gpt-image-2", validation_alias="PIPELINE_HOST_MODEL")
    keyframe_model: str = Field(default="nano-banana-2.1", validation_alias="PIPELINE_KEYFRAME_MODEL")
    image_check_model: str = Field(default="claude-haiku-5-5", validation_alias="PIPELINE_IMAGE_CHECK_MODEL")
    # When every try of a keyframe fails, its best try is judged once more by this stronger model before the
    # shot fails; None turns the second opinion off.
    image_second_opinion_model: str | None = Field(default="claude-sonnet-5-5",
                                                   validation_alias="PIPELINE_IMAGE_SECOND_OPINION_MODEL")
    # When the image model returns nothing (a filter, an outage), the same draw is tried once on this model.
    image_fallback_model: str | None = Field(default="gpt-image-2", validation_alias="PIPELINE_IMAGE_FALLBACK_MODEL")
    # Video clips: one per shot, animated between its start and end keyframes, asked for at the slot's own
    # length (length_factor 1.0) and fitted to the slot at assembly. Sung shots are skipped here: the
    # lip-sync stage makes them from the keyframe and the song. Chosen on the 2026-11-02 bake-offs.
    video_model: str = Field(default="minimax-h3-max-turbo", validation_alias="PIPELINE_VIDEO_MODEL")
    video_fallback_model: str | None = Field(default="kling-o3-standard", validation_alias="PIPELINE_VIDEO_FALLBACK_MODEL")
    video_workers: int = Field(default=4, validation_alias="PIPELINE_VIDEO_WORKERS")
    video_length_factor: float = Field(default=1.0, validation_alias="PIPELINE_VIDEO_LENGTH_FACTOR")
    video_sung_mode: str = Field(default="skip", validation_alias="PIPELINE_VIDEO_SUNG_MODE")
    # Lip-sync: the host's sung shots, drawn from the keyframe and the song from the shot's start.
    lipsync_model: str = Field(default="minimax-h3-max-lipsync", validation_alias="PIPELINE_LIPSYNC_MODEL")
    # Assembly: every clip at this size and frame rate; a dissolve overlaps two shots by this many seconds.
    assembly_width: int = Field(default=1280, validation_alias="PIPELINE_ASSEMBLY_WIDTH")
    assembly_height: int = Field(default=720, validation_alias="PIPELINE_ASSEMBLY_HEIGHT")
    assembly_fps: int = Field(default=24, validation_alias="PIPELINE_ASSEMBLY_FPS")
    assembly_dissolve_s: float = Field(default=0.4, validation_alias="PIPELINE_ASSEMBLY_DISSOLVE_S")
    # Any one ffmpeg or ffprobe call that runs longer than this is stopped and fails its stage.
    ffmpeg_timeout_s: float = Field(default=600.0, validation_alias="PIPELINE_FFMPEG_TIMEOUT_S")
    image_workers: int = Field(default=4, validation_alias="PIPELINE_IMAGE_WORKERS")
    # The judge's 0-10 score an image needs to pass; below it the image is drawn again. The judge gives a 6
    # to "the right picture with one small difference", which is good enough for a viewer.
    image_pass_score: int = Field(default=6, validation_alias="PIPELINE_IMAGE_PASS_SCORE")
    # A keyframe whose tries all failed uses its best-scoring try at or above this; below it, the shot fails.
    image_floor_score: int = Field(default=5, validation_alias="PIPELINE_IMAGE_FLOOR_SCORE")
    # Draws per image (the first draw plus retries); when none passes, the best try at or above the floor is
    # used, and below it the image fails. The stop-loss: a stage stops drawing when more than this share of
    # its first STOP_LOSS_AFTER finished images failed every draw.
    image_max_tries: int = Field(default=2, validation_alias="PIPELINE_IMAGE_MAX_TRIES")
    image_stop_loss: float = Field(default=0.5, validation_alias="PIPELINE_IMAGE_STOP_LOSS")
    storage: str = Field(default="local", validation_alias="PIPELINE_STORAGE")
    r2_account_id: str | None = Field(default=None, validation_alias="R2_ACCOUNT_ID")
    r2_access_key_id: str | None = Field(default=None, validation_alias="R2_ACCESS_KEY_ID")
    r2_secret_access_key: str | None = Field(default=None, validation_alias="R2_SECRET_ACCESS_KEY")
    r2_bucket: str | None = Field(default=None, validation_alias="R2_BUCKET")
