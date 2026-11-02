import logging
import time

import fal_client
import httpx
from pydantic import BaseModel

from src.service.image.fal_errors import transient
from src.service.lipsync.lipsync_prices import lipsync_model

logger = logging.getLogger(__name__)

TIMEOUT_S = 900
POLL_S = 3.0


class LipsyncReply(BaseModel):
    data: bytes
    content_type: str
    model: str
    seed: int | None
    audio_seconds: float
    cost_usd: float
    seconds: float = 0.0
    request_id: str


class FalLipsyncService:
    """Lip-sync through fal: one clip of the person in one image singing one stretch of audio. The request
    is submitted and its id logged before we wait, so a timeout never loses a paid clip."""

    name = "fal"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client: fal_client.SyncClient | None = None

    def generate(self, model: str, image_url: str, audio_url: str, audio_seconds: float,
                 seed: int | None = None) -> LipsyncReply:
        """A clip from `model`. A transient fal error is retried once; any other error is raised."""
        try:
            return self._generate(model, image_url, audio_url, audio_seconds, seed)
        except fal_client.FalClientError as exc:
            if not transient(exc):
                raise
            logger.warning("fal %s: %s; trying once more", model, str(exc)[:160])
            return self._generate(model, image_url, audio_url, audio_seconds, seed)

    def _generate(self, model: str, image_url: str, audio_url: str, audio_seconds: float,
                  seed: int | None) -> LipsyncReply:
        spec = lipsync_model(model)
        arguments = {"image_url": image_url, "audio_url": audio_url, **spec.fixed}
        if spec.seeded and seed is not None:
            arguments["seed"] = seed
        started = time.monotonic()
        handle = self._get_client().submit(spec.endpoint, arguments=arguments)
        logger.info("fal %s: submitted %s, %.2fs of audio, seed %s", model, handle.request_id, audio_seconds, seed)
        result = self._wait(spec.endpoint, handle.request_id)
        video = result["video"]
        data = self._download(video["url"])
        elapsed = round(time.monotonic() - started, 1)
        cost = round(spec.price_per_s * audio_seconds, 4)
        logger.info("fal %s: %s, %.2fs of audio, seed %s, $%.3f, %.0fs", model, handle.request_id, audio_seconds,
                    seed, cost, elapsed)
        return LipsyncReply(data=data, content_type=video.get("content_type") or "video/mp4", model=model,
                            seed=result.get("seed", seed), audio_seconds=audio_seconds, cost_usd=cost,
                            seconds=elapsed, request_id=handle.request_id)

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
        """A short-lived fal URL for a keyframe or an audio cut that lives locally; one retry on a failed upload."""
        try:
            return self._get_client().upload(data, content_type)
        except (httpx.HTTPError, fal_client.FalClientError) as exc:
            logger.warning("fal upload failed (%s); trying once more", str(exc)[:160])
            return self._get_client().upload(data, content_type)

    def _get_client(self) -> fal_client.SyncClient:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("FAL_KEY is not set")
            self._client = fal_client.SyncClient(key=self._api_key, default_timeout=TIMEOUT_S)
        return self._client
