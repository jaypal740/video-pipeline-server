import logging

from src.core.configs import Settings
from src.schema.image.image_check_schema import Checklist
from src.service.llm.llm_service import LabelledImage, LlmCall, LlmService

logger = logging.getLogger(__name__)

SYSTEM = """You check one generated picture against a short checklist for an educational video for school \
children. Image 1 is the picture to check; any further images are the reference sheets it was drawn from. \
Look carefully and answer every question with ok true or false. For every false answer give a one-sentence \
reason that says what you actually see. Judge only what is visible; do not guess at intent."""


class ImageCheckService:
    """Quality layer 1: a yes/no checklist read off one image by a vision model. The questions come from
    the caller, built from the storyboard; this service only asks them and returns the answers."""

    def __init__(self, llm: LlmService, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    def check(self, image: bytes, refs: list[LabelledImage], questions: list[str],
              model: str | None = None) -> tuple[Checklist, LlmCall]:
        model = model or self._settings.image_check_model
        images = [LabelledImage(label="the picture to check", data=image)] + refs
        user = "Checklist:\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
        checklist, call = self._llm.inspect(model, SYSTEM, user, images, Checklist)
        if len(checklist.answers) != len(questions):
            raise ValueError(f"image check answered {len(checklist.answers)} of {len(questions)} questions")
        return checklist, call
