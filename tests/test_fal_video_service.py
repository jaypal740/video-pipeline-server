import fal_client
import pytest

from src.service.video import fal_video_service as module
from src.service.video.fal_video_service import FalVideoService
from src.service.video.video_prices import video_model


class FakeHandle:
    def __init__(self, request_id):
        self.request_id = request_id


class FakeClient:
    """fal's client: records submissions; `fail` maps an endpoint to the error its first call raises."""

    def __init__(self, fail=None):
        self.calls, self.fail, self.n = [], fail or {}, 0

    def submit(self, endpoint, arguments):
        self.calls.append((endpoint, arguments))
        exc = self.fail.get(endpoint)
        if exc is not None and sum(1 for e, _ in self.calls if e == endpoint) == 1:
            raise exc
        self.n += 1
        return FakeHandle(f"req-{self.n}")

    def status(self, endpoint, request_id):
        return fal_client.Completed(logs=[], metrics={})

    def result(self, endpoint, request_id):
        return {"video": {"url": f"https://fal.test/{request_id}.mp4", "content_type": "video/mp4"}, "seed": 42}


class FakeReply:
    def __init__(self, content_type="video/mp4"):
        self.headers, self.content = {"content-type": content_type}, b"mp4-bytes"

    def raise_for_status(self):
        pass


def make(monkeypatch, client, fallback="minimax-h3", content_type="video/mp4"):
    service = FalVideoService(api_key="x", fallback_model=fallback)
    monkeypatch.setattr(service, "_get_client", lambda: client)
    monkeypatch.setattr(module.httpx, "get", lambda url, timeout, follow_redirects: FakeReply(content_type))
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    return service


def test_each_model_is_asked_in_its_own_argument_names(monkeypatch):
    client = FakeClient()
    service = make(monkeypatch, client)
    reply = service.generate("kling-3-pro", "the ship sails", "https://img/k.png", seconds=4, seed=7, negative="text")
    assert reply.request_id == "req-1" and reply.cost_usd == pytest.approx(0.448) and reply.seconds_requested == 4
    endpoint, args = client.calls[0]
    assert endpoint == video_model("kling-3-pro").endpoint
    assert args["duration"] == "4" and args["start_image_url"] == "https://img/k.png" and args["generate_audio"] is False
    assert args["negative_prompt"] == "text" and "seed" not in args               # Kling has no seed
    service.generate("wan-3", "the ship sails", "https://img/k.png", seconds=1, seed=7, negative="text")
    _, args = client.calls[1]
    assert args["duration"] == 2 and args["seed"] == 7 and args["audio"] is False     # 2 s minimum, seeded
    assert args["enable_prompt_expansion"] is False and "negative_prompt" not in args
    service.generate("minimax-h3", "the ship sails", "https://img/k.png", seconds=3)
    assert client.calls[2][1]["duration"] == 5 and client.calls[2][1]["resolution"] == "768P"


def test_a_transient_error_is_retried_on_the_same_model_and_a_content_error_falls_back(monkeypatch):
    kling = video_model("kling-3-pro").endpoint
    client = FakeClient(fail={kling: fal_client.FalClientTimeoutError(10, "req-x")})
    reply = make(monkeypatch, client).generate("kling-3-pro", "p", "https://img", seconds=5)
    assert reply.model == "kling-3-pro" and [e for e, _ in client.calls] == [kling, kling]

    blocked = fal_client.FalClientError("no_media_generated")
    client = FakeClient(fail={kling: blocked})
    reply = make(monkeypatch, client).generate("kling-3-pro", "p", "https://img", seconds=5)
    assert reply.model == "minimax-h3" and client.calls[1][0] == video_model("minimax-h3").endpoint
    assert reply.seconds_requested == 5 and reply.cost_usd == pytest.approx(0.30)

    with pytest.raises(fal_client.FalClientError):
        make(monkeypatch, FakeClient(fail={kling: blocked}), fallback=None).generate("kling-3-pro", "p", "https://img", 5)


def test_a_download_that_is_not_a_video_is_refused(monkeypatch):
    with pytest.raises(RuntimeError, match="instead of a video"):
        make(monkeypatch, FakeClient(), content_type="text/html").generate("wan-3", "p", "https://img", seconds=3)
