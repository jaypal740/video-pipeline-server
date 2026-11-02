from pydantic import BaseModel, Field


class Brief(BaseModel):
    """What the judge is given: the description the picture is scored against, and the hard rules. A
    keyframe also gives its opening moment, from which the judge writes that shot's own checks."""

    description: str
    rules: list[str]
    moment: str | None = None


class Answer(BaseModel):
    rule: str
    ok: bool
    reason: str


class AskedFor(BaseModel):
    """One thing the opening moment asks for, written by the judge: who or what is shown, how many when
    the words say, or what must be absent. The reason comes before the verdict, so the verdict follows it."""

    item: str
    reason: str
    ok: bool


class Checklist(BaseModel):
    """What the judge returns: the opening moment's own checks (empty when there is no moment), a 0-10
    score for how well the picture meets its description, one sentence on what matches and what does
    not, and a yes/no per hard rule with a reason for every no."""

    asked_for: list[AskedFor]
    score: int = Field(ge=0, le=10)
    score_reason: str
    answers: list[Answer]

    def missing(self, brief: Brief) -> list[str]:
        """The parts of the brief this answer left out: the moment's list, the score, unanswered rules."""
        gaps = ["moment"] if brief.moment and not self.asked_for else []
        if not self.score_reason.strip():
            gaps.append("score")
        return gaps + [f"rule {i}" for i in range(len(self.answers) + 1, len(brief.rules) + 1)]

    def problems(self, pass_score: int) -> list[str]:
        found = [f"the moment asks for {a.item}: {a.reason}" for a in self.asked_for if not a.ok]
        found += [f"{a.rule}: {a.reason}" for a in self.answers if not a.ok]
        if self.score < pass_score:
            found.append(f"matches the description {self.score}/10 (needs {pass_score}): {self.score_reason}")
        return found


class Fill(BaseModel):
    """A follow-up answer holding only the parts the judge's first answer left out; the rest stays empty."""

    asked_for: list[AskedFor]
    score: int | None = Field(ge=0, le=10)
    score_reason: str | None
    answers: list[Answer]
