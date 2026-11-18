from src.schema.lesson.lesson_schema import GradeBand
from src.service.lesson.lesson_rules import band_rules
from src.service.two_d.common import style_of


def test_2d_films_keep_the_flat_cartoon_look_whatever_the_band_files_say():
    for band in GradeBand:
        assert style_of(band).startswith("A bold 2D cartoon with thick clean dark outlines, flat saturated colours")
        assert band_rules(band).style not in style_of(band)
