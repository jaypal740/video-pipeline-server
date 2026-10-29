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
    sheet_model: str = Field(default="nano-banana-pro", validation_alias="PIPELINE_SHEET_MODEL")
    host_model: str = Field(default="nano-banana-pro", validation_alias="PIPELINE_HOST_MODEL")
    keyframe_model: str = Field(default="nano-banana-2.1", validation_alias="PIPELINE_KEYFRAME_MODEL")
    image_check_model: str = Field(default="claude-haiku-5-5", validation_alias="PIPELINE_IMAGE_CHECK_MODEL")
    storage: str = Field(default="local", validation_alias="PIPELINE_STORAGE")
    r2_account_id: str | None = Field(default=None, validation_alias="R2_ACCOUNT_ID")
    r2_access_key_id: str | None = Field(default=None, validation_alias="R2_ACCESS_KEY_ID")
    r2_secret_access_key: str | None = Field(default=None, validation_alias="R2_SECRET_ACCESS_KEY")
    r2_bucket: str | None = Field(default=None, validation_alias="R2_BUCKET")
