import numpy as np

from src.schema.song.song_analysis_schema import LyricLine, TimedWord
from src.service.assembly.captions import ACTIVE, BOX, SUNG, UPCOMING, caption_image, caption_states

LINES = [LyricLine(index=0, text="The Sun shines", start=1.0, end=2.5, confidence=0.9),
         LyricLine(index=1, text="On us", start=4.0, end=5.0, confidence=0.9)]
WORDS = [TimedWord(index=0, line_index=0, text="The", start=1.0, end=1.4),
         TimedWord(index=1, line_index=0, text="Sun", start=1.5, end=1.9),
         TimedWord(index=2, line_index=0, text="shines", start=2.0, end=2.5),
         TimedWord(index=3, line_index=1, text="On", start=4.0, end=4.4),
         TimedWord(index=4, line_index=1, text="us", start=4.5, end=5.0)]


def test_one_line_shows_while_sung_its_word_lit_and_nothing_between_lines():
    states = [(round(a, 3), round(b, 3), line, lit) for a, b, line, lit, _ in
              caption_states(LINES, WORDS, offset=0.5, length=5.0)]
    assert states == [(0.0, 0.5, None, None), (0.5, 0.9, 0, 0), (0.9, 1.0, 0, None), (1.0, 1.4, 0, 1),
                      (1.4, 1.5, 0, None), (1.5, 2.0, 0, 2), (2.0, 3.5, None, None), (3.5, 3.9, 1, 3),
                      (3.9, 4.0, 1, None), (4.0, 4.5, 1, 4), (4.5, 5.0, None, None)]


def test_a_line_gives_way_to_the_next_when_they_overlap():
    late = [LINES[0].model_copy(update={"end": 4.5}), LINES[1]]
    states = [s[:4] for s in caption_states(late, WORDS, offset=0.0, length=5.0)]
    assert (2.5, 4.0, 0, None) in states and states[-2][2] == 1


def colours(image) -> set:
    return {tuple(p) for p in np.array(image).reshape(-1, 4)}


def test_sung_lit_and_upcoming_words_are_drawn_in_a_dark_box_at_the_bottom():
    image = caption_image(WORDS[:3], 1.7, 1280, 720)
    pixels = np.array(image)
    found = colours(image)
    assert SUNG in found and ACTIVE in found and BOX in found
    assert tuple(pixels[0, 0]) == (0, 0, 0, 0) and pixels[:550, :, 3].max() == 0     # the top stays clear
    box_rows = np.where(pixels[:, 640, 3] > 0)[0]
    assert 720 - 720 * 0.08 - 2 <= box_rows.max() <= 720 - 720 * 0.08 + 1
    faded = pixels[(pixels[..., 3] == 255) & (pixels[..., 0] > 100) & (pixels[..., 0] < 200)
                   & (pixels[..., 0] == pixels[..., 1])]
    assert len(faded) > 0 and UPCOMING not in found        # upcoming white at 55%, seen over the box as grey


def test_a_long_line_wraps_inside_the_box_and_no_words_draw_nothing():
    long = [TimedWord(index=i, line_index=0, text="photosynthesis", start=i, end=i + 0.5) for i in range(12)]
    one, many = np.array(caption_image(long[:1], 0.2, 1280, 720)), np.array(caption_image(long, 0.2, 1280, 720))
    height = lambda p: np.ptp(np.where(p[:, :, 3].max(axis=1) > 0)[0])
    width = lambda p: np.ptp(np.where(p[:, :, 3].max(axis=0) > 0)[0])
    assert height(many) > 2 * height(one) and width(many) <= 720 * 0.9 + 1
    assert np.array(caption_image([], 0.2, 1280, 720))[..., 3].max() == 0


def test_screen_words_show_for_their_whole_stretch_beside_the_captions():
    screen = [(1.2, 2.2, ("photosphere",), "right"), (4.0, 5.0, ("93 million miles", "core"), "left")]
    states = caption_states(LINES, WORDS, offset=0.0, length=5.0, screen=screen)
    shown = {(round(a, 3), round(b, 3)): s for a, b, _, _, s in states if s}
    assert min(a for a, _ in shown) == 1.2 and max(b for _, b in shown) == 5.0
    assert all(s == screen[0] for (a, _), s in shown.items() if a < 2.2)
    assert all(s == screen[1] for (a, _), s in shown.items() if a >= 4.0)


def test_screen_words_are_drawn_in_the_open_third_and_stack():
    one = np.array(caption_image([], None, 1280, 720, (0.0, 1.0, ("photosphere",), "right")))
    two = np.array(caption_image([], None, 1280, 720, (0.0, 1.0, ("photosphere", "corona"), "left")))
    cols = lambda p: np.where(p[:, :, 3].max(axis=0) > 0)[0]
    rows = lambda p: np.where(p[:, :, 3].max(axis=1) > 0)[0]
    assert cols(one).min() > 1280 * 0.55 and cols(two).max() < 1280 * 0.45       # right third, then left third
    assert abs(rows(one).mean() - 720 * 0.42) < 20 and np.ptp(rows(two)) > 1.8 * np.ptp(rows(one))
    assert SUNG in {tuple(p) for p in one.reshape(-1, 4)}
