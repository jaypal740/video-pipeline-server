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
    source: str = "image"                                       # "video": it re-times the mouth of a clip we send
    min_video_s: float = 0.0                                    # the shortest clip a "video" model takes

    def seconds(self, slot: float) -> float:
        """The audio to send for a slot: the slot, or the model's minimum when the slot is shorter."""
        return round(max(slot, self.min_s), 3)


LIPSYNC_MODELS: dict[str, LipsyncModel] = {
    # Takes no prompt; the clip runs as long as the audio sent, which must be at least 5 s.
    "minimax-h3-max-lipsync": LipsyncModel(endpoint="minimax/h3-max/lip-sync/image-to-video", min_s=5.0,
                                           price_per_s=0.08, fixed={"resolution": "768P"}, seeded=True),
    # Video in, video out: the person keeps moving as in the clip sent and only the mouth follows the audio.
    # Price noted on the 2026-11-16 singer test. cut_off: the result is as long as the shorter of the two.
    "sync-lipsync-v3": LipsyncModel(endpoint="fal-ai/sync-lipsync/v3", min_s=0.0, price_per_s=0.133,
                                    fixed={"sync_mode": "cut_off"}, source="video"),
    # The same, from the family of the image model above; it answered in 42 s on 2026-11-16 while sync v3 sat
    # in its queue past 15 minutes. Price from its fal page that day. 480P is enough: she is laid in small.
    "minimax-h3-max-video-lipsync": LipsyncModel(endpoint="minimax/h3-max/lip-sync/video-to-video", min_s=0.0,
                                                 price_per_s=0.10, source="video", min_video_s=5.0, seeded=True,
                                                 fixed={"resolution": "480P", "duration_mode": "cut_off"}),
}


def lipsync_model(name: str) -> LipsyncModel:
    try:
        return LIPSYNC_MODELS[name]
    except KeyError:
        raise ValueError(f"No fal endpoint or price for lip-sync model {name!r}; add it to lipsync_prices.py") from None
