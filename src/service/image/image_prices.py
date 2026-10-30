from pydantic import BaseModel

# USD per image on fal, checked 2026-10-29 on each model's fal page. fal returns no dollar cost, so every
# image model we call must be listed here. Only the Nano Banana family exposes a seed.


class ImageModel(BaseModel):
    """One image model as we call it: fal endpoints and the price per image at each resolution. family
    picks the request shape: Nano Banana takes an aspect ratio, a resolution tier and a thinking level;
    GPT Image takes a pixel size and a quality level, and has no seed."""

    generate: str
    edit: str
    prices: dict[str, float]
    family: str = "nano_banana"
    seeded: bool = True
    # fal bills each attached reference as input tokens; about this much per 1K image.
    per_reference: float = 0.0
    # The thinking levels this endpoint accepts, lowest first; a request is clamped to the nearest.
    thinking_levels: list[str] = ["minimal", "medium", "high"]

    def thinking(self, wanted: str) -> str:
        order = ["minimal", "medium", "high"]
        allowed = [level for level in order if level in self.thinking_levels]
        return min(allowed, key=lambda level: abs(order.index(level) - order.index(wanted)))


IMAGE_MODELS: dict[str, ImageModel] = {
    "nano-banana-2.1": ImageModel(generate="google/nano-banana-2.1", edit="google/nano-banana-2.1/edit",
                                  prices={"1K": 0.044, "2K": 0.063, "4K": 0.138}, per_reference=0.002),
    "nano-banana-pro": ImageModel(generate="fal-ai/nano-banana-pro", edit="fal-ai/nano-banana-pro/edit",
                                  prices={"1K": 0.15, "2K": 0.15, "4K": 0.30}, thinking_levels=["high"]),
    "nano-banana-2": ImageModel(generate="fal-ai/nano-banana-2", edit="fal-ai/nano-banana-2/edit",
                                prices={"1K": 0.08, "2K": 0.12, "4K": 0.16}, thinking_levels=["minimal", "high"]),
    # Medium quality. fal lists 1024x1024 $0.061 and 1920x1080 $0.053; the 1024x1536 portrait price is an
    # estimate from the token rates until fal's page lists it.
    "gpt-image-2": ImageModel(generate="openai/gpt-image-2", edit="openai/gpt-image-2/edit", family="gpt_image",
                              prices={"1:1": 0.061, "3:4": 0.09, "16:9": 0.053}, seeded=False, thinking_levels=[]),
}

# GPT Image sizes for our three aspects; 1024x1536 is the model's native portrait size.
GPT_IMAGE_SIZES = {"1:1": (1024, 1024), "3:4": (1024, 1536), "16:9": (1920, 1080)}
GPT_IMAGE_QUALITY = "medium"


def image_model(name: str) -> ImageModel:
    try:
        return IMAGE_MODELS[name]
    except KeyError:
        raise ValueError(f"No fal endpoint or price for image model {name!r}; add it to image_prices.py") from None
