import logging

from src.core.configs import Settings
from src.schema.image.image_check_schema import Brief, Checklist, Fill
from src.service.llm.llm_service import LabelledImage, LlmCall, LlmService

logger = logging.getLogger(__name__)

# One answer plus follow-ups that ask only for the parts it left out.
CHECK_TRIES = 3

SYSTEM = """You check one generated picture for an educational video for school children. Image 1 is the \
picture to check; any further images are the reference sheets it was drawn from.

If you are given an opening moment, first list what it asks for, one item each: every person (with how \
many when the words say), every object (with how many when the words say, such as "a second prism" or \
"many rays"), and anything it says is absent (such as an empty beach). End the list with one item for no \
people, creatures or characters beyond the ones listed. For each item say in one sentence what you see, \
then true or false. Judge an item as a viewer would: is it clearly there and recognisable? Its exact place \
in the frame, age, size, shade or angle fail it only when the point of the moment depends on them (beams of \
different lengths, light from one side). Someone or something the moment names belongs in the picture even \
when no reference image shows it. With no opening moment, leave that list empty.

Then give a score from 0 to 10 for how well the picture meets the description you are given, judging the \
whole picture: the kind of thing shown, its overall shape, its main colours and materials, the setting and \
the moment. No generated picture is perfect, so score what a viewer would see: 10 is a flawless match; 7 is \
a good match with small differences a viewer would not notice (edge sharpness, a slightly different shade, \
how a sleeve sits, something touching the frame edge); 5 is recognisably the right thing with one clear \
mistake; 3 or less is the wrong thing. Say in one sentence what matches and what does not.

Then answer each rule true or false, with a one-sentence reason for every false that says what you actually \
see. Judge only what is visible; do not guess at intent. The description and the opening moment are what \
was asked for: a rule never forbids something they ask for."""


class ImageCheckService:
    """Quality layer 1: a score and a few hard rules read off one image by a vision model. The brief comes
    from the caller, built from the storyboard; this service only asks and returns the answers."""

    def __init__(self, llm: LlmService, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    def check(self, image: bytes, refs: list[LabelledImage], brief: Brief,
              model: str | None = None) -> tuple[Checklist, LlmCall]:
        model = model or self._settings.image_check_model
        images = [LabelledImage(label="the picture to check", data=image)] + refs
        moment = f"Opening moment:\n{brief.moment}\n\n" if brief.moment else ""
        user = (f"{moment}Description to score:\n{brief.description}\n\nRules:\n"
                + "\n".join(f"{i}. {r}" for i, r in enumerate(brief.rules, start=1)))
        checklist, call = self._llm.inspect(model, SYSTEM, user, images, Checklist)
        if len(checklist.answers) > len(brief.rules):            # line the answers up with the rules we asked
            logger.warning("image check answered %d of %d rules; using the first %d", len(checklist.answers),
                           len(brief.rules), len(brief.rules))
            checklist.answers = checklist.answers[:len(brief.rules)]
        for _ in range(CHECK_TRIES - 1):                         # a reply that left parts out: ask for those only
            gaps = checklist.missing(brief)
            if not gaps:
                break
            logger.warning("image check left out %s; asking for those only", ", ".join(gaps))
            fill, more = self._llm.inspect(model, SYSTEM, self._fill_request(user, brief, gaps), images, Fill)
            call.attempts, call.cost_usd = call.attempts + more.attempts, call.cost_usd + more.cost_usd
            call.input_tokens, call.output_tokens = (call.input_tokens + more.input_tokens,
                                                     call.output_tokens + more.output_tokens)
            self._merge(checklist, fill, brief, gaps)
        gaps = checklist.missing(brief)
        if gaps:                                                 # still missing: those parts pass unchecked
            logger.warning("image check left out %s after %d tries; passing them unchecked", ", ".join(gaps),
                           CHECK_TRIES)
            if "score" in gaps:
                checklist.score, checklist.score_reason = self._settings.image_pass_score, "not given by the judge"
        return checklist, call

    def problems(self, checklist: Checklist) -> list[str]:
        return checklist.problems(self._settings.image_pass_score)

    @staticmethod
    def _fill_request(user: str, brief: Brief, gaps: list[str]) -> str:
        wanted = []
        for gap in gaps:
            if gap == "moment":
                wanted.append("- the list of what the opening moment asks for, each item judged")
            elif gap == "score":
                wanted.append("- the score and its one sentence")
            else:
                n = int(gap.removeprefix("rule "))
                wanted.append(f"- rule {n}. {brief.rules[n - 1]}")
        return (f"{user}\n\nYour answer left some parts out. Answer only these, in this order, and leave every other "
                "field empty:\n" + "\n".join(wanted))

    @staticmethod
    def _merge(checklist: Checklist, fill: Fill, brief: Brief, gaps: list[str]) -> None:
        if "moment" in gaps:
            checklist.asked_for = fill.asked_for
        if "score" in gaps and fill.score is not None and fill.score_reason:
            checklist.score, checklist.score_reason = fill.score, fill.score_reason
        wanted = len(brief.rules) - len(checklist.answers)       # the missing rules, in order, follow the answered ones
        checklist.answers += fill.answers[:wanted]
