import re
from pathlib import Path

from num2words import num2words

_MARKUP = re.compile(r"\*\*(.*?)\*\*")
_NUMBER = re.compile(r"^(\$?)(\d[\d,]*(?:\.\d+)?)(%|st|nd|rd|th)?$")
_EDGE_PUNCT = re.compile(r"^[^\w$]+|[^\w%]+$")


def read_lyrics(path: Path) -> list[str]:
    """The lines to align: blank lines and # comments dropped, **bold** vocabulary markup removed."""
    lines = []
    for raw in Path(path).read_text().splitlines():
        line = _MARKUP.sub(r"\1", raw).strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def spoken(word: str) -> list[str]:
    """How a written word is said, as alignable tokens. The aligner only knows letters, so a line
    starting "1858." or "99%" used to start at its next word: 1858 -> eighteen fifty eight."""
    core = _EDGE_PUNCT.sub("", word)
    m = _NUMBER.match(core)
    if not m:
        return [word]
    dollar, digits, suffix = m.groups()
    value = float(digits.replace(",", "")) if "." in digits else int(digits.replace(",", ""))
    if suffix in ("st", "nd", "rd", "th"):
        text = num2words(value, to="ordinal")
    elif isinstance(value, int) and 1100 <= value <= 2099 and "," not in digits and not dollar and not suffix:
        text = num2words(value, to="year")
    else:
        text = num2words(value)
    if suffix == "%":
        text += " percent"
    if dollar:
        text += " dollars"
    return text.replace("-", " ").replace(",", "").split()
