from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.schema.song.song_analysis_schema import LyricLine, TimedWord

# The lyrics as JamWise burns them into its videos (apps/api/src/modules/generation/render/VideoComposition.tsx):
# the line being sung, alone, in a dark box at the bottom, each word lit while it is sung; nothing between
# lines. Sizes are JamWise's on its 1080 x 1080 frame, that square fitted to our frame's height, so the
# captions look the same and wrap where JamWise's do.
FONT = Path(__file__).resolve().parents[2] / "templates" / "fonts" / "Inter-Bold.ttf"
JAMWISE_SIZE = 1080
FONT_PX, OUTLINE_PX, PAD_X, PAD_Y, RADIUS = 36, 2, 32, 16, 12
LINE_HEIGHT = 1.3
BOX_SHARE = 0.9                   # the box is at most this share of JamWise's frame
BOTTOM_SHARE = 0.08               # the box sits this share of JamWise's frame above the bottom edge
SUNG = (255, 255, 255, 255)
ACTIVE = (168, 85, 247, 255)      # #a855f7
UPCOMING = (255, 255, 255, 140)   # white at 55%
OUTLINE = (0, 0, 0, 255)
BOX = (0, 0, 0, 153)              # black at 60%


def caption_states(lines: list[LyricLine], words: list[TimedWord], offset: float,
                   length: float) -> list[tuple[float, float, int | None, int | None]]:
    """The caption over the film, as (start, end, line index, lit word index) in film seconds; line None is
    nothing on screen. A line shows from its start to its end, or to the next line's start if sooner; a word
    is lit from its start to its end, and is sung (white) after. offset: the song time the film starts at."""
    timed = [l for l in lines if any(w.line_index == l.index for w in words)]
    cuts = {0.0, length}
    for i, line in enumerate(timed):
        stop = min(line.end, timed[i + 1].start) if i + 1 < len(timed) else line.end
        cuts |= {line.start - offset, stop - offset}
        cuts |= {t - offset for w in words if w.line_index == line.index for t in (w.start, w.end)}
    edges = sorted(t for t in cuts if 0.0 <= t <= length)
    states = []
    for a, b in zip(edges, edges[1:]):
        t = (a + b) / 2 + offset
        line = next((l for i, l in enumerate(timed)
                     if l.start <= t < (min(l.end, timed[i + 1].start) if i + 1 < len(timed) else l.end)), None)
        lit = None if line is None else next(
            (w.index for w in words if w.line_index == line.index and w.start <= t <= w.end), None)
        state = (line.index if line else None, lit)
        if states and states[-1][2:] == state:
            states[-1] = (states[-1][0], b, *state)
        else:
            states.append((a, b, *state))
    return states


def caption_image(words: list[TimedWord], t: float | None, width: int, height: int) -> Image.Image:
    """The frame-sized see-through picture of one line at song time t: words before t sung, the word at t
    lit, the rest upcoming. No words gives an empty picture."""
    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not words:
        return frame
    unit = height / JAMWISE_SIZE
    font = ImageFont.truetype(str(FONT), round(FONT_PX * unit))
    outline, pad_x, pad_y = max(1, round(OUTLINE_PX * unit)), round(PAD_X * unit), round(PAD_Y * unit)
    space = font.getlength(" ")
    rows, row, room = [], [], height * BOX_SHARE - 2 * pad_x
    for w in words:
        if row and sum(font.getlength(x.text) for x in row) + space * len(row) + font.getlength(w.text) > room:
            rows.append(row)
            row = []
        row.append(w)
    rows.append(row)
    spans = [sum(font.getlength(x.text) for x in r) + space * (len(r) - 1) for r in rows]
    step = font.size * LINE_HEIGHT
    box_w, box_h = max(spans) + 2 * pad_x, step * len(rows) + 2 * pad_y
    left, bottom = (width - box_w) / 2, height - height * BOTTOM_SHARE
    top = bottom - box_h
    ImageDraw.Draw(frame).rounded_rectangle((left, top, left + box_w, bottom), round(RADIUS * unit), fill=BOX)
    ascent, descent = font.getmetrics()
    strokes, fills = Image.new("RGBA", frame.size, (0, 0, 0, 0)), Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw_strokes, draw_fills = ImageDraw.Draw(strokes), ImageDraw.Draw(fills)
    for k, (r, span) in enumerate(zip(rows, spans)):
        x = (width - span) / 2
        y = top + pad_y + k * step + (step - ascent - descent) / 2 + ascent
        for w in r:
            colour = SUNG if t is not None and t > w.end else ACTIVE if t is not None and t >= w.start else UPCOMING
            draw_strokes.text((x, y), w.text, font=font, fill=OUTLINE, anchor="ls", stroke_width=outline,
                              stroke_fill=OUTLINE)
            draw_fills.text((x, y), w.text, font=font, fill=colour, anchor="ls")
            x += font.getlength(w.text) + space
    return Image.alpha_composite(Image.alpha_composite(frame, strokes), fills)


def caption_track(lines: list[LyricLine], words: list[TimedWord], offset: float, length: float, width: int,
                  height: int, work: Path) -> Path:
    """The captions as pictures with their durations, written for ffmpeg's concat reader; returns the list.
    Each distinct picture is drawn once."""
    drawn: dict[tuple[int | None, int | None], Path] = {}
    entries = []
    for a, b, line, lit in caption_states(lines, words, offset, length):
        if (line, lit) not in drawn:
            own = [w for w in words if w.line_index == line] if line is not None else []
            lit_word = next((w for w in own if w.index == lit), None)
            t = (lit_word.start + lit_word.end) / 2 if lit_word else a + offset + (b - a) / 2
            path = work / f"caption-{len(drawn)}.png"
            caption_image(own, t, width, height).save(path)
            drawn[(line, lit)] = path
        entries.append(f"file '{drawn[(line, lit)].name}'\nduration {b - a:.4f}")
    listing = work / "captions.txt"
    listing.write_text("\n".join(entries + [entries[-1].splitlines()[0]]) + "\n")   # concat needs the last twice
    return listing
