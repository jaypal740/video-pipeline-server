from fastapi import APIRouter, Depends

from src.api.dependencies import get_container, require_api_key
from src.core.container import Container
from src.schema.api.response_schema import DataResponse, SongView
from src.schema.song.song_schema import Song

router = APIRouter(prefix="/v1", tags=["song"], dependencies=[Depends(require_api_key)])


def to_view(song: Song) -> SongView:
    return SongView(id=song.id, name=song.name, status=song.status, lyrics=song.lyrics, created_at=song.created_at)


@router.get("/songs", response_model=DataResponse[list[SongView]])
def list_songs(container: Container = Depends(get_container)) -> DataResponse[list[SongView]]:
    """Our library songs (defaults and test songs), oldest first."""
    return DataResponse(data=[to_view(s) for s in container.song_service().list()])


@router.get("/songs/{song_ref}", response_model=DataResponse[SongView])
def get_song(song_ref: str, container: Container = Depends(get_container)) -> DataResponse[SongView]:
    """One library song by its id or name."""
    return DataResponse(data=to_view(container.song_service().get(song_ref)))
