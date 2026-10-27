from src.service.intake.intake_alignment import lyric_lines, song_confidence, spoken_tokens, timed_words

LINES = ["In 1858 a ship", "It holds."]


def test_tokens_remember_which_written_word_they_came_from():
    tokens, owners = spoken_tokens(LINES)
    assert tokens == ["In", "eighteen", "fifty", "eight", "a", "ship", "It", "holds."]
    assert owners[1:4] == [(0, 1)] * 3


def test_a_spoken_number_folds_back_onto_its_written_word():
    _, owners = spoken_tokens(LINES)
    aligned = [(0.0, 0.2, 0.9), (0.3, 0.5, 0.8), (0.5, 0.7, 0.8), (0.7, 0.9, 0.8), (1.0, 1.1, 0.9),
               (1.1, 1.4, 0.9), (2.0, 2.1, 0.9), (2.1, 2.5, 0.9)]
    words = timed_words(LINES, owners, aligned)
    assert [w.text for w in words] == ["In", "1858", "a", "ship", "It", "holds."]
    assert (words[1].start, words[1].end, words[1].confidence) == (0.3, 0.9, 0.8)


def test_an_untimed_word_takes_the_gap_between_its_neighbours():
    lines = ["one - two"]
    _, owners = spoken_tokens(lines)
    words = timed_words(lines, owners, [(1.0, 1.2, 0.9), None, (1.6, 1.8, 0.9)])
    assert (words[1].start, words[1].end, words[1].confidence) == (1.2, 1.6, None)


def test_a_low_confidence_line_is_marked_unclear():
    lines = ["real words", "purple elephants"]
    _, owners = spoken_tokens(lines)
    words = timed_words(lines, owners, [(0, 1, 0.85), (1, 2, 0.8), (3, 4, 0.2), (4, 5, 0.15)])
    out = lyric_lines(lines, words)
    assert [l.unclear for l in out] == [False, True]
    assert (out[0].start, out[0].end) == (0, 2)


def test_song_confidence_weights_long_lines_more():
    lines = ["a b c", "oh"]
    _, owners = spoken_tokens(lines)
    words = timed_words(lines, owners, [(0, 1, 0.9), (1, 2, 0.9), (2, 3, 0.9), (3, 4, 0.1)])
    assert song_confidence(lyric_lines(lines, words)) == 0.7
