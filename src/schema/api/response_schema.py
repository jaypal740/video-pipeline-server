from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel

from src.schema.song.song_schema import SongStatus

T = TypeVar("T")


class DataResponse(BaseModel, Generic[T]):
    """Every successful response is wrapped as {"data": ...}."""

    data: T


class HealthView(BaseModel):
    status: str


class SongView(BaseModel):
    """A library song as products see it; its local audio path stays private."""

    id: str
    name: str
    status: SongStatus
    lyrics: str
    created_at: datetime
