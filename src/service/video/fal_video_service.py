import logging
import time

import fal_client
import httpx
from pydantic import BaseModel

from src.service.image.fal_errors import transient
from src.service.video.video_prices import video_model

logger = logging.getLogger(__name__)

# A clip takes one to several minutes in fal's queue plus generation.
TIMEOUT_S = 900
POLL_S = 3.0


class VideoReply(BaseModel):
    data: bytes
    content_type: str
    model: str
    seed: int | None
    seconds_requested: float
    cost_usd: float
    seconds: float = 0.0
    request_id: str


class FalVideoService:
    """Image-to-video through fal. One call, one clip from one start image. The request is submitted and its
    id logged before we wait, so a timeout never loses a paid clip: the same id is polled again."""

    name = "fal"

    def __init__(self, api_key: str | None, fallback_model: str | None = None) -> None:
        self._api_key = api_key
        self._fallback = fallback_model
        self._client: fal_client.SyncClient | None = None

    def generate(self, model: str, prompt: str, image_url: str, seconds: float, seed: int | None = None,
                 negative: str | None = None, end_image_url: str | None = None) -> VideoReply:
        """A clip from `model`. A transient fal error is retried once on the same model; an error the model
        will give again (a filter, no media) is retried once on the fallback model, when one is set."""
        try:
            return self._generate(model, prompt, image_url, seconds, seed, negative, end_image_url)
        except fal_client.FalClientError as exc:
            if transient(exc):
                logger.warning("fal %s: %s; trying the same model once more", model, str(exc)[:160])
                return self._generate(model, prompt, image_url, seconds, seed, negative, end_image_url)
            if not self._fallback or self._fallback == model:
                raise
            logger.warning("fal %s gave no clip (%s); drawing on %s instead", model, str(exc)[:160], self._fallback)
            return self._generate(self._fallback, prompt, image_url, seconds, seed, negative, end_image_url)

    def _generate(self, model: str, prompt: str, image_url: str, seconds: float, seed: int | None,
                  negative: str | None, end_image_url: str | None) -> VideoReply:
        spec = video_model(model)
        seconds = spec.seconds(seconds)
        arguments = {"prompt": prompt[:spec.prompt_max], spec.image_arg: image_url,
                     "duration": str(int(seconds)) if spec.duration_as_str else seconds, **spec.fixed}
        if spec.end_frame and end_image_url:
            arguments["end_image_url"] = end_image_url
        if spec.seeded and seed is not None:
            arguments["seed"] = seed
        if spec.negative_prompt and negative:
            arguments["negative_prompt"] = negative
        started = time.monotonic()
        handle = self._get_client().submit(spec.endpoint, arguments=arguments)
        logger.info("fal %s: submitted %s, %ss, seed %s, end frame %s", model, handle.request_id, seconds, seed,
                    bool(end_image_url))
        result = self._wait(spec.endpoint, handle.request_id)
        video = result["video"]
        data = self._download(video["url"])
        elapsed = round(time.monotonic() - started, 1)
        cost = round(spec.price_per_s * seconds, 4)
        logger.info("fal %s: %s, %ss, seed %s, $%.3f, %.0fs", model, handle.request_id, seconds, seed, cost, elapsed)
        return VideoReply(data=data, content_type=video.get("content_type") or "video/mp4", model=model,
                          seed=result.get("seed", seed), seconds_requested=seconds, cost_usd=cost, seconds=elapsed,
                          request_id=handle.request_id)

    def _wait(self, endpoint: str, request_id: str) -> dict:
        """Poll a submitted request until it completes; the id is known, so a timeout can be resumed."""
        client = self._get_client()
        deadline = time.monotonic() + TIMEOUT_S
        while True:
            status = client.status(endpoint, request_id)
            if isinstance(status, fal_client.Completed):
                return client.result(endpoint, request_id)
            if time.monotonic() > deadline:
                raise fal_client.FalClientTimeoutError(TIMEOUT_S, request_id)
            time.sleep(POLL_S)

    @staticmethod
    def _download(url: str) -> bytes:
        reply = httpx.get(url, timeout=120, follow_redirects=True)
        reply.raise_for_status()
        if not reply.headers.get("content-type", "").startswith("video/"):
            raise RuntimeError(f"fal returned {reply.headers.get('content-type')!r} instead of a video from {url}")
        return reply.content

    def upload(self, data: bytes, content_type: str) -> str:
        """A short-lived fal URL for a start image that lives in local storage."""
        return self._get_client().upload(data, content_type)

    def _get_client(self) -> fal_client.SyncClient:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("FAL_KEY is not set")
            self._client = fal_client.SyncClient(key=self._api_key, default_timeout=TIMEOUT_S)
        return self._client
