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
