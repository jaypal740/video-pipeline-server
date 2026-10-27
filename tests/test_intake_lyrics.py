from src.service.intake.intake_lyrics import read_lyrics, spoken


def test_read_lyrics_drops_comments_blanks_and_markup(tmp_path):
    path = tmp_path / "lyrics.txt"
    path.write_text("# heard by Whisper\n\nWarm water **evaporates**.\n  Clouds form  \n")
    assert read_lyrics(path) == ["Warm water evaporates.", "Clouds form"]


def test_numbers_are_spoken_as_words():
    assert spoken("1858.") == ["eighteen", "fifty", "eight"]
    assert spoken("99%") == ["ninety", "nine", "percent"]
    assert spoken("7,000") == ["seven", "thousand"]
    assert spoken("6th") == ["sixth"]
    assert spoken("2.5") == ["two", "point", "five"]
    assert spoken("$5") == ["five", "dollars"]


def test_ordinary_words_are_left_alone():
    assert spoken("bacteria,") == ["bacteria,"]
    assert spoken("great-grandmother") == ["great-grandmother"]
