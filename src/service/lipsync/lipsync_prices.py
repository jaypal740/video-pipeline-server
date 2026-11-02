from pydantic import BaseModel

# USD per second of audio sent on fal, checked 2026-11-02 on the model's fal page and OpenAPI schema. fal
# returns no dollar cost, so every lip-sync model we call must be listed here.


class LipsyncModel(BaseModel):
    """One image+audio-to-video lip-sync model as we call it: the fal endpoint, the shortest audio it takes,
    the price per second of audio and the arguments sent on every call."""

    endpoint: str
    min_s: float
    price_per_s: float
    fixed: dict = {}
    seeded: bool = False

    def seconds(self, slot: float) -> float:
        """The audio to send for a slot: the slot, or the model's minimum when the slot is shorter."""
        return round(max(slot, self.min_s), 3)


LIPSYNC_MODELS: dict[str, LipsyncModel] = {
    # Takes no prompt; the clip runs as long as the audio sent, which must be at least 5 s.
    "minimax-h3-max-lipsync": LipsyncModel(endpoint="minimax/h3-max/lip-sync/image-to-video", min_s=5.0,
                                           price_per_s=0.08, fixed={"resolution": "768P"}, seeded=True),
}


def lipsync_model(name: str) -> LipsyncModel:
    try:
        return LIPSYNC_MODELS[name]
    except KeyError:
        raise ValueError(f"No fal endpoint or price for lip-sync model {name!r}; add it to lipsync_prices.py") from None
