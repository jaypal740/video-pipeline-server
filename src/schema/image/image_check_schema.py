from pydantic import BaseModel


class Answer(BaseModel):
    question: str
    ok: bool
    reason: str


class Checklist(BaseModel):
    """What the image check returns: one yes/no per question, with a reason for every no."""

    answers: list[Answer]

    def problems(self) -> list[str]:
        return [f"{a.question}: {a.reason}" for a in self.answers if not a.ok]
