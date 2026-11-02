import math

from pydantic import BaseModel

# USD per second of clip on fal, checked 2026-11-02 on each model's fal page and OpenAPI schema. fal returns
# no dollar cost, so every video model we call must be listed here. Argument names differ per endpoint,
# so each entry also says how to ask for the start image, the length and silence.


class VideoModel(BaseModel):
    """One image-to-video model as we call it: the fal endpoint, the clip lengths it accepts, the price per
    second at the resolution we use, and the request arguments that differ between endpoints."""

    endpoint: str
    min_s: float
    max_s: float
    price_per_s: float
    image_arg: str                              # the start-frame argument
    duration_as_str: bool = False               # Kling wants "5", the others want 5
    fixed: dict = {}                            # arguments sent on every call: resolution, audio off, no rewriting
    end_frame: bool = True                      # accepts end_image_url
    fractional: bool = False                    # takes any length, such as 2.3 s, not only whole seconds
    negative_prompt: bool = False
    seeded: bool = False
    prompt_max: int = 2500

    def seconds(self, wanted: float) -> float:
        """The length to request for a wanted length, within the model's range: exact to a tenth of a second
        when the model takes any length, else rounded up to whole seconds."""
        if self.fractional:
            return min(self.max_s, max(self.min_s, math.ceil(wanted * 10) / 10))
        return int(min(self.max_s, max(self.min_s, math.ceil(wanted))))


VIDEO_MODELS: dict[str, VideoModel] = {
    # The fallback: cleanest pictures but slow; 3 s minimum; audio off by default; no negative prompt, no seed.
    "kling-o3-standard": VideoModel(endpoint="fal-ai/kling-video/o3/standard/image-to-video", min_s=3, max_s=15,
                                    price_per_s=0.084, image_arg="image_url", duration_as_str=True,
                                    fixed={"generate_audio": False}),
    # The default. Any length from 0.92 s, so a clip can match its slot; 768p; prompt rewriting off.
    "minimax-h3-max-turbo": VideoModel(endpoint="minimax/h3-max-turbo/image-to-video", min_s=0.92, max_s=15,
                                       price_per_s=0.04, image_arg="image_url", fractional=True,
                                       fixed={"resolution": "768P", "prompt_expansion_mode": "disabled"},
                                       seeded=True, prompt_max=50000),
}


def video_model(name: str) -> VideoModel:
    try:
        return VIDEO_MODELS[name]
    except KeyError:
        raise ValueError(f"No fal endpoint or price for video model {name!r}; add it to video_prices.py") from None
