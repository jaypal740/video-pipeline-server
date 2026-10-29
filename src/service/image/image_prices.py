from pydantic import BaseModel

# USD per image on fal, checked 2026-10-29 on each model's fal page. fal returns no dollar cost, so every
# image model we call must be listed here. Only the Nano Banana family exposes a seed.


class ImageModel(BaseModel):
    """One image model as we call it: fal endpoints and the price per image at each resolution."""

    generate: str
    edit: str
    prices: dict[str, float]
    seeded: bool = True
    # fal bills each attached reference as input tokens; about this much per 1K image.
    per_reference: float = 0.0


IMAGE_MODELS: dict[str, ImageModel] = {
    "nano-banana-2.1": ImageModel(generate="google/nano-banana-2.1", edit="google/nano-banana-2.1/edit",
                                  prices={"1K": 0.044, "2K": 0.063, "4K": 0.138}, per_reference=0.002),
    "nano-banana-pro": ImageModel(generate="fal-ai/nano-banana-pro", edit="fal-ai/nano-banana-pro/edit",
                                  prices={"1K": 0.15, "2K": 0.15, "4K": 0.30}),
    "nano-banana-2": ImageModel(generate="fal-ai/nano-banana-2", edit="fal-ai/nano-banana-2/edit",
                                prices={"1K": 0.08, "2K": 0.12, "4K": 0.16}),
}


def image_model(name: str) -> ImageModel:
    try:
        return IMAGE_MODELS[name]
    except KeyError:
        raise ValueError(f"No fal endpoint or price for image model {name!r}; add it to image_prices.py") from None
