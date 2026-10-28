import re

from num2words import num2words

_MARKUP = re.compile(r"\*\*(.*?)\*\*")
_SECTION_TAG = re.compile(r"^\[[^\]]*\]$")
_SECTION_NUMBER = re.compile(r"[\s\d]+$")
_DASH = re.compile(r"\s*(?:\u2014|--)\s*")
_NUMBER = re.compile(r"^(\$?)(\d[\d,]*(?:\.\d+)?)(%|st|nd|rd|th)?$")
_EDGE_PUNCT = re.compile(r"^[^\w$]+|[^\w%]+$")


def parse_lyrics(text: str) -> list[tuple[str | None, str]]:
    """The lines to align, each with the section it sits in: [Chorus 2] -> "chorus", None before any tag.
    Blank lines, # comments and the tags themselves dropped, **bold** vocabulary markup removed,
    em dashes and -- turned into spaces."""
    section, lines = None, []
    for raw in text.splitlines():
        line = _DASH.sub(" ", _MARKUP.sub(r"\1", raw)).strip()
        if not line or line.startswith("#"):
            continue
        if _SECTION_TAG.match(line):
            section = _SECTION_NUMBER.sub("", line[1:-1]).strip().lower() or None
        else:
            lines.append((section, line))
    return lines


def clean_lyrics(text: str) -> list[str]:
    """The lines to align, without their sections."""
    return [line for _, line in parse_lyrics(text)]


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
