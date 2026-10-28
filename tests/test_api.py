import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.container import Container
from src.schema.song.song_schema import Song


@pytest.fixture
def client(container: Container):
    with TestClient(create_app(container)) as client:
        yield client


@pytest.fixture
def key(container: Container) -> str:
    return container.auth_service().create("jamwise")[1]


def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def test_health_needs_no_key(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.json() == {"data": {"status": "ok"}}


def test_songs_need_a_valid_key(client, container, key):
    assert client.get("/v1/songs").status_code == 401
    assert client.get("/v1/songs", headers=auth("vp_000000000000_nope")).status_code == 401
    container.auth_service().revoke(key[:15])
    assert client.get("/v1/songs", headers=auth(key)).status_code == 403


def test_library_songs_are_listed_without_their_audio_path(client, container, key):
    song = container.song_repository().songs.add(Song(name="ocean", audio_path="/private/a.mp3", lyrics="Salt."))
    listed = client.get("/v1/songs", headers=auth(key)).json()["data"]
    assert [s["id"] for s in listed] == [song.id] and "audio_path" not in listed[0]
    one = client.get("/v1/songs/ocean", headers=auth(key))
    assert one.status_code == 200 and one.json()["data"]["lyrics"] == "Salt."
    assert client.get("/v1/songs/missing", headers=auth(key)).status_code == 404
