import fal_client
import pytest

from src.service.image import fal_image_service as module
from src.service.image.fal_image_service import FalImageService


class FakeClient:
    """fal's client: the primary model returns an error, any other model returns an image."""

    def __init__(self):
        self.calls = []

    def subscribe(self, endpoint, arguments, with_logs, client_timeout):
        self.calls.append((endpoint, arguments))
        if "nano-banana" in endpoint:
            raise fal_client.FalClientError("no_media_generated: The model did not generate the expected output")
        return {"images": [{"url": "https://fal.test/one.png", "content_type": "image/png"}]}


@pytest.fixture
def service(monkeypatch):
    service = FalImageService(api_key="x", fallback_model="gpt-image-2")
    client = FakeClient()
    monkeypatch.setattr(service, "_get_client", lambda: client)
    monkeypatch.setattr(module.httpx, "get", lambda url, timeout: type("R", (), {"content": b"png-bytes"})())
    return service, client


def test_a_draw_the_model_refuses_is_made_on_the_fallback_model(service):
    service, client = service
    reply = service.generate("nano-banana-2.1", "a purple enzyme", ["https://ref"], "1:1", seed=7)
    assert reply.model == "gpt-image-2" and reply.data == b"png-bytes" and reply.seed is None
    assert len(client.calls) == 2 and "nano-banana" in client.calls[0][0] and "gpt-image-2" in client.calls[1][0]
    assert client.calls[1][1]["image_size"] == {"width": 1024, "height": 1024}      # the fallback's own arguments


def test_without_a_fallback_the_error_is_raised(monkeypatch):
    service = FalImageService(api_key="x", fallback_model=None)
    monkeypatch.setattr(service, "_get_client", lambda: FakeClient())
    with pytest.raises(fal_client.FalClientError, match="no_media_generated"):
        service.generate("nano-banana-2.1", "a purple enzyme", [], "1:1", seed=7)
