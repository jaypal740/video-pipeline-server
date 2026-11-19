import io

import numpy as np
from PIL import Image, ImageDraw

# Every drawing that is not a backdrop is drawn on this flat colour, which code then makes transparent.
# Tried on 2026-11-11: the image model keeps it flat, and so does the video model when it animates the drawing.
KEY = (255, 0, 255)
KEY_WORDS = "magenta (#FF00FF)"
# Distance from the key colour, in RGB: nearer than NEAR is fully clear, further than FAR fully kept.
NEAR, FAR = 90.0, 170.0
# The key colour enclosed by the subject (the gap between an arm and the body) is surround too when it is
# this near the pure key and at least this many pixels; a pink tongue or a small speck is left alone.
PURE, MIN_HOLE, MAX_HOLES = 40.0, 30, 30
# A usable cut-out has this share of its frame edge clear, and its subject fills this much of the frame. A
# drawing needs more of its edge clear than a frame of its loop: one that touches the edge is cut off there
# for good, and leaves its loop no room to move.
MIN_CLEAR_EDGE, STILL_CLEAR_EDGE = 0.9, 0.98
SUBJECT_SHARE = (0.03, 0.85)


def cut_out(data: bytes) -> tuple[bytes, list[str]]:
    """A drawing cut out (see `cut`), as a PNG, and what makes it unusable as a cut-out."""
    rgba, found = cut(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), STILL_CLEAR_EDGE)
    out = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG")
    return out.getvalue(), found


def cut(picture: np.ndarray, clear_edge: float = MIN_CLEAR_EDGE) -> tuple[np.ndarray, list[str]]:
    """A picture (height, width, 3) with its key-coloured surround made transparent (height, width, 4), and
    what makes it unusable as a cut-out. The surround is the key colour joined to the frame's edge, and pure
    key colour enclosed by the subject; a colour merely like the key inside the subject stays. The rim fades
    out and loses the key's tint."""
    rgb = picture.astype(np.float32)
    dist = np.linalg.norm(rgb - np.array(KEY, dtype=np.float32), axis=2)
    outside = _joined(dist < FAR)
    for _ in range(MAX_HOLES):
        holes = np.argwhere((dist < PURE) & ~outside)
        if len(holes) < MIN_HOLE:
            break
        outside |= _joined((dist < FAR) & ~outside, tuple(holes[0]))
    alpha = np.where(outside, np.clip((dist - NEAR) / (FAR - NEAR), 0, 1), 1.0)
    rim = outside & (alpha > 0)
    for channel in (0, 2):                                  # magenta is red and blue: bring both down to the green
        rgb[..., channel] = np.where(rim, np.minimum(rgb[..., channel], rgb[..., 1]), rgb[..., channel])
    edge = np.concatenate([alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]])
    clear, share = float((edge == 0).mean()), float((alpha > 0.5).mean())
    found = []
    if clear < clear_edge:
        found.append(f"only {clear:.0%} of the frame's edge is flat {KEY_WORDS}; the subject touches the edge or the "
                     "surround is not flat")
    if not SUBJECT_SHARE[0] <= share <= SUBJECT_SHARE[1]:
        found.append(f"the subject fills {share:.0%} of the frame; it must be whole, with a margin, and not tiny")
    return np.dstack([rgb, alpha * 255]).astype(np.uint8), found


def _joined(mask: np.ndarray, seed: tuple[int, int] | None = None) -> np.ndarray:
    """The true part of a mask connected to `seed` (row, column), or to the frame's edge when none is given."""
    padded = Image.fromarray(np.pad(mask, 1, constant_values=seed is None).astype(np.uint8) * 255, "L").copy()   # writable
    row, column = seed or (-1, -1)
    ImageDraw.floodfill(padded, (int(column) + 1, int(row) + 1), 128)
    return (np.asarray(padded) == 128)[1:-1, 1:-1]
