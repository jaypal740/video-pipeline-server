import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.service.assembly.captions import FONT
from src.service.render.timeline import (BODY, CLOSEST, NAME_PX, WORD_GAP, WORD_PAD, WORD_PX, WORD_RIM, WORD_WIDE, Frame,
                                         _back)

# A key word is drawn by code, never by the image model: dark letters on a white plate with a dark rim, like
# the drawings' own outlines. The arrows between the stages of a row or ring are drawn the same way.
INK, PLATE = (28, 32, 28, 255), (255, 255, 255, 255)


class Painter:
    """Paints the film's frames. backgrounds: full-frame pictures; stills: cut-out drawings that hold still;
    loops: a moving drawing's cut-out frames (n, height, width, 4), played at `fps` and repeated. A loop's
    last frame is its first again, so it is left out when the loop comes round."""

    def __init__(self, width: int, height: int, fps: float, backgrounds: dict[str, Image.Image],
                 stills: dict[str, Image.Image], loops: dict[str, np.ndarray]) -> None:
        self._width, self._height, self._fps = width, height, fps
        big = (round(width * CLOSEST), round(height * CLOSEST))             # room for the closest the camera goes
        self._backgrounds = {name: ImageOps.fit(picture.convert("RGB"), big, Image.LANCZOS)
                             for name, picture in backgrounds.items()}
        self._stills, self._loops = stills, loops
        self._sized: dict[str, Image.Image] = {}                             # a still at the size it was last drawn
        self._words: dict[tuple[str, float], Image.Image] = {}
        self._arrows: dict[tuple[int, int], Image.Image] = {}

    def paint(self, frame: Frame) -> Image.Image:
        width, height, zoom = self._width, self._height, frame.zoom
        back = self._backgrounds[frame.background]
        seen = 0.5 / zoom                                                   # half the share of the scene in view
        box = ((0.5 + frame.pan - seen) * back.width, (0.5 - seen) * back.height,
               (0.5 + frame.pan + seen) * back.width, (0.5 + seen) * back.height)
        canvas = back.transform((width, height), Image.EXTENT, box, Image.BILINEAR)
        for x, y, length, turn, popped in frame.arrows:
            arrow = self._arrow(max(8, round(length * width * zoom * (0.6 + 0.4 * _back(popped)))), round(turn))
            self._lay(canvas, arrow, width / 2 + (x - 0.5 - frame.pan) * width * zoom, height / 2 + (y - 0.5) * height * zoom,
                      min(1.0, 2 * popped))
        for item in frame.items:
            side = max(2, round(item.size * height * zoom))
            picture = self._picture(item.drawing, item.age, side)
            x = width / 2 + (item.x - 0.5 - frame.pan) * width * zoom
            y = height / 2 + (item.y - 0.5) * height * zoom
            self._lay(canvas, picture, x, y, item.opacity)
            if item.name:                                                   # just under its subject
                name = self._word(item.name, NAME_PX)
                self._lay(canvas, name, x, y + side * BODY / 2 + name.height / 2, item.opacity)
        for text, x, y, popped in frame.words:
            plate = self._word(text)
            reach = plate.width / 2 + WORD_GAP * width
            x = min(max(x * width, reach), width - reach)
            grow = 0.6 + 0.4 * _back(popped)
            if popped < 1:
                plate = plate.resize((max(2, round(plate.width * grow)), max(2, round(plate.height * grow))), Image.BILINEAR)
            self._lay(canvas, plate, x, y * height, min(1.0, 2 * popped))
        return canvas

    def _picture(self, drawing: str, age: float, side: int) -> Image.Image:
        if drawing in self._loops:
            frames = self._loops[drawing]
            now = Image.fromarray(np.asarray(frames[int(age * self._fps) % max(1, len(frames) - 1)]), "RGBA")
            return now.resize((side, side), Image.BILINEAR)
        sized = self._sized.get(drawing)
        if sized is None or sized.width != side:
            sized = self._sized[drawing] = self._stills[drawing].resize((side, side), Image.LANCZOS)
        return sized

    def _arrow(self, length: int, turn: int) -> Image.Image:
        """An arrow that long, turned that many degrees clockwise from pointing right, drawn once for each."""
        if (length, turn) not in self._arrows:
            tall, rim = length * 0.7, max(1, round(WORD_RIM * self._height * 0.75))
            arrow = Image.new("RGBA", (length, round(tall)), (0, 0, 0, 0))
            shape = [(0, 0.32), (0.55, 0.32), (0.55, 0), (1, 0.5), (0.55, 1), (0.55, 0.68), (0, 0.68)]
            ImageDraw.Draw(arrow).polygon([(x * (length - 1), y * (tall - 1)) for x, y in shape], fill=PLATE, outline=INK,
                                          width=rim)
            self._arrows[length, turn] = arrow.rotate(-turn, Image.BICUBIC, expand=True) if turn else arrow
        return self._arrows[length, turn]

    def _word(self, text: str, px: float = WORD_PX) -> Image.Image:
        """A word on its plate, in letters `px` of the frame's height, drawn once."""
        if (text, px) not in self._words:
            h = self._height * px / WORD_PX
            font = ImageFont.truetype(str(FONT), round(WORD_PX * h))
            pad, rim = round(WORD_PAD * h), max(2, round(WORD_RIM * h))
            left, top, right, bottom = font.getbbox(text)
            plate = Image.new("RGBA", (right - left + 2 * pad + 2 * rim, bottom - top + 2 * pad + 2 * rim), (0, 0, 0, 0))
            draw = ImageDraw.Draw(plate)
            draw.rounded_rectangle((0, 0, plate.width - 1, plate.height - 1), radius=pad, fill=PLATE, outline=INK, width=rim)
            draw.text((rim + pad - left, rim + pad - top), text, font=font, fill=INK)
            widest = WORD_WIDE * self._width
            if plate.width > widest:
                plate = plate.resize((round(widest), round(plate.height * widest / plate.width)), Image.LANCZOS)
            self._words[text, px] = plate
        return self._words[text, px]

    @staticmethod
    def _lay(canvas: Image.Image, picture: Image.Image, x: float, y: float, opacity: float) -> None:
        """A cut-out laid on the frame with its centre at (x, y)."""
        mask = picture.getchannel("A")
        if opacity < 1:
            mask = mask.point(lambda a: round(a * opacity))
        canvas.paste(picture.convert("RGB"), (round(x - picture.width / 2), round(y - picture.height / 2)), mask)
