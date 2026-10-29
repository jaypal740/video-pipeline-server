import logging

import fal_client
import httpx
from pydantic import BaseModel

from src.service.image.image_prices import image_model

logger = logging.getLogger(__name__)

RESOLUTION = "1K"
# Waits for the queue and the generation; Nano Banana Pro can take half a minute per image.
TIMEOUT_S = 300


class ImageReply(BaseModel):
    data: bytes
    content_type: str
    model: str
    seed: int | None
    cost_usd: float


class FalImageService:
    """Image generation through fal. One call, one image; references are fetched by fal from URLs, and a
    file that has no URL is uploaded to fal's own storage first."""

    name = "fal"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client: fal_client.SyncClient | None = None

    def generate(self, model: str, prompt: str, refs: list[str], aspect: str, seed: int | None,
                 thinking: str = "medium") -> ImageReply:
        spec = image_model(model)
        arguments = {"prompt": prompt, "num_images": 1, "aspect_ratio": aspect, "resolution": RESOLUTION,
                     "output_format": "png", "thinking_level": thinking}
        if seed is not None:
            arguments["seed"] = seed
        if refs:
            arguments["image_urls"] = refs
        endpoint = spec.edit if refs else spec.generate
        result = self._get_client().subscribe(endpoint, arguments=arguments, with_logs=False,
                                              client_timeout=TIMEOUT_S)
        image = result["images"][0]
        data = httpx.get(image["url"], timeout=60).content
        cost = spec.prices[RESOLUTION] + spec.per_reference * len(refs)
        logger.info("fal %s: %s, %d refs, seed %s, $%.3f", model, aspect, len(refs), seed, cost)
        return ImageReply(data=data, content_type=image.get("content_type") or "image/png", model=model,
                          seed=result.get("seed", seed), cost_usd=cost)

    def upload(self, data: bytes, content_type: str) -> str:
        """A short-lived fal URL for bytes that live in local storage."""
        return self._get_client().upload(data, content_type)

    def _get_client(self) -> fal_client.SyncClient:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("FAL_KEY is not set")
            self._client = fal_client.SyncClient(key=self._api_key, default_timeout=TIMEOUT_S)
        return self._client
