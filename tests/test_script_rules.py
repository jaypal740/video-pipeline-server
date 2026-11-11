from src.schema.lesson.lesson_schema import Beat
from src.schema.song.song_analysis_schema import TimedWord
from src.schema.storyboard.script_schema import Action, BeatScriptDraft, Drawing, DrawingList, Screen
from src.service.storyboard.script_rules import drawing_problems, finish, play

LIST = [Drawing(id="meadow", kind="background", look="A green meadow"),
        Drawing(id="night", kind="background", look="A dark sky"),
        Drawing(id="egg", kind="object", look="A pale egg on a leaf"),
        Drawing(id="caterpillar", kind="object", look="A green caterpillar", motion="crawls slowly"),
        Drawing(id="leaf", kind="object", look="A leaf"),
        Drawing(id="twig", kind="object", look="A twig"),
        Drawing(id="arrow", kind="icon", look="A curved arrow"),
        Drawing(id="pip", kind="performer", look="Waving", motion="sways", sings=True),
        Drawing(id="pip_point", kind="performer", look="Pointing up", motion="bobs")]
DRAWINGS = {d.id: d for d in LIST}


def beat(id="b01", start=10.0, end=14.0, type="explain", **more):
    return Beat(id=id, start=start, end=end, type=type, lines=[0], concept="An egg hatches.", hint="h", **more)


def act(verb, at, drawing=None, **more):
    return Action(verb=verb, at=at, drawing=drawing, **more)


def script(beat_id, background, *actions):
    return BeatScriptDraft(beat=beat_id, background=background, actions=list(actions))


def problems(actions, b=None, before=None, background="meadow"):
    b = b or beat()
    return play([script(b.id, background, *actions)], [b], DRAWINGS, before or Screen())[0]


def listed(drawings=None, performer="Pip, a green caterpillar", palette=("#FFC857", "#2E86AB", "#F25F5C", "#FFFFFF")):
    return DrawingList(performer=performer, palette=list(palette), drawings=drawings or LIST)


def test_a_sound_list_of_drawings_passes():
    assert drawing_problems(listed(), [beat(type="hero")]) == []


def test_the_list_needs_a_background_a_palette_and_ids_of_its_own():
    found = drawing_problems(listed([LIST[2], LIST[2]], palette=["red"]), [beat()])
    assert any("own id" in p for p in found) and any("palette" in p for p in found)
    assert any("at least one background" in p for p in found)


def test_hero_beats_need_a_performer_who_is_described_and_moves():
    assert any("hero beats" in p for p in drawing_problems(listed(LIST[:3]), [beat(type="hero")]))
    assert any("hero beats" in p for p in drawing_problems(listed(LIST[:7] + LIST[8:]), [beat(type="hero")]))
    loud = Drawing(id="horn", kind="object", look="A horn", sings=True)
    assert any("only a performer drawing sings" in p for p in drawing_problems(listed(LIST + [loud]), [beat()]))
    assert any("performer must say" in p for p in drawing_problems(listed(performer=None), [beat()]))
    still = Drawing(id="pip", kind="performer", look="Waving")
    assert any("pip needs a motion" in p for p in drawing_problems(listed([LIST[0], still]), [beat()]))
    poses = [Drawing(id=f"pip_{i}", kind="performer", look="Pose", motion="sways") for i in range(13)]
    assert any("13 performer poses" in p for p in drawing_problems(listed([LIST[0]] + poses), [beat()]))


def test_backgrounds_and_icons_do_not_move_and_of_names_a_drawing_of_the_same_kind():
    moving = Drawing(id="sky", kind="background", look="Sky", motion="clouds drift")
    assert any("sky is a background" in p for p in drawing_problems(listed([moving]), [beat()]))
    wrong = Drawing(id="egg_open", kind="object", look="A hatched egg", of="meadow")
    assert any("egg_open's of" in p for p in drawing_problems(listed(LIST + [wrong]), [beat()]))
    right = Drawing(id="egg_open", kind="object", look="A hatched egg", of="egg")
    assert drawing_problems(listed(LIST + [right]), [beat()]) == []


def test_the_stages_of_a_process_line_up_in_a_row_several_performer_drawings_among_them():
    stages = [act("show", 10.0, "egg", pos="row"), act("show", 11.0, "pip", pos="row"),
              act("show", 12.0, "pip_point", pos="row"), act("show", 13.0, "arrow", pos="top")]
    found, screens = play([script("b01", "meadow", *stages)], [beat()], DRAWINGS, Screen())
    assert found == [] and list(screens[0].on.items())[:3] == [("egg", "row"), ("pip", "row"), ("pip_point", "row")]
    assert any("center is taken by egg" in p for p in problems(stages[:1] + [act("show", 11.0, "leaf", pos="center")]))
    assert any("row is taken by leaf" in p for p in problems([act("show", 10.0, "leaf", pos="left"), stages[0]]))
    assert any("joins the row with show" in p for p in problems([act("show", 10.0, "leaf", pos="top"),
                                                                 act("move", 11.0, "leaf", pos="row", until=12.0)]))
    full = Screen(background="meadow", on={d: "row" for d in ("egg", "leaf", "twig", "arrow", "pip")})
    assert any("the row is full" in p for p in problems([act("show", 10.0, "caterpillar", pos="row")], before=full))


def test_a_beat_plays_and_leaves_its_drawings_on_screen():
    b = beat()
    found, screens = play([script("b01", "meadow", act("show", 10.0, "egg", pos="center"),
                                  act("swap", 12.0, "egg", to="caterpillar"))], [b], DRAWINGS, Screen())
    assert found == [] and screens[0].on == {"caterpillar": "center"} and screens[0].background == "meadow"


def test_drawings_stay_into_the_next_beat_until_the_background_changes():
    before = Screen(background="meadow", on={"caterpillar": "center"})
    move = [act("move", 10.5, "caterpillar", pos="left", until=12.0), act("push_in", 12.0, until=14.0)]
    assert problems(move, before=before) == []
    assert any("caterpillar is not on screen" in p for p in problems(move, before=before, background="night"))


def test_a_drawing_is_shown_once_in_a_free_spot_and_four_fit_on_screen():
    before = Screen(background="meadow", on={"caterpillar": "center"})
    assert any("already on screen" in p for p in problems([act("show", 10.0, "caterpillar", pos="left")], before=before))
    assert any("center is taken by caterpillar" in p for p in problems([act("show", 10.0, "egg", pos="center")],
                                                                       before=before))
    crowd = [act("show", 10.0 + i, d, pos=p) for i, (d, p) in enumerate(
        [("egg", "left"), ("leaf", "right"), ("twig", "top"), ("arrow", "bottom")])]
    assert any("at most 4" in p for p in problems(crowd, before=before))


def test_actions_are_timed_inside_their_beat_and_only_the_models_verbs_are_allowed():
    shown = act("show", 10.0, "caterpillar", pos="center")
    assert any("inside the beat" in p for p in problems([shown, act("hide", 15.0, "caterpillar")]))
    assert any("until must be after" in p for p in problems([shown, act("zoom", 11.0, "caterpillar", scale=1.5)]))
    assert any("scale must be" in p for p in problems([shown, act("zoom", 11.0, "caterpillar", scale=9, until=12.0)]))
    assert any("left or right" in p for p in problems([shown, act("pan", 11.0, pos="top", until=12.0)]))
    assert any("code adds keyword" in p for p in problems([shown, act("keyword", 11.0, text="egg")]))
    assert any("'meadow' is not a listed performer" in p for p in problems([shown, act("show", 11.0, "meadow")]))
    assert any("background 'egg'" in p for p in problems([shown, act("push_in", 11.0, until=12.0)], background="egg"))


def test_something_must_really_move_in_every_beat():
    still = [act("show", 10.0, "egg", pos="center"), act("push_in", 10.5, until=14.0)]
    assert any("nothing really moves" in p for p in problems(still))
    assert problems(still + [act("swap", 12.0, "egg", to="leaf")]) == []                 # a swap is a real change
    assert problems([act("show", 10.0, "caterpillar", pos="center"), act("pan", 11.0, pos="left", until=13.0)]) == []


def test_a_long_beat_needs_two_actions():
    b = beat(end=18.0)
    assert any("at least 2" in p for p in problems([act("show", 10.0, "caterpillar", pos="center")], b))


def test_a_hero_beat_keeps_its_singer_on_screen_from_the_start():
    b = beat(type="hero")
    late = [act("show", 12.0, "pip", pos="center"), act("push_in", 12.5, until=14.0)]
    assert any("from its start" in p for p in problems(late, b))
    gone = [act("show", 10.0, "pip", pos="center"), act("swap", 12.0, "pip", to="pip_point")]
    assert any("pip sings the whole hero beat" in p for p in problems(gone, b))
    mute = [act("show", 10.0, "pip_point", pos="center"), act("push_in", 11.0, until=14.0)]
    assert any("pip_point cannot sing" in p for p in problems(mute, b))
    sung = script("b01", "meadow", act("show", 10.2, "pip", pos="center"), act("push_in", 11.0, until=14.0))
    found, screens = play([sung], [b], DRAWINGS, Screen())
    assert found == [] and screens[0].singer == "pip"
    assert sung.actions[0].at == 10.0                        # 0.2 s after the start is moved to the start


def test_only_one_performer_drawing_is_on_screen_at_a_time():
    before = Screen(background="meadow", on={"pip": "left"})
    assert any("only one performer drawing" in p for p in problems([act("show", 10.0, "pip_point", pos="right")],
                                                                   before=before))
    both = [act("show", 10.0, "caterpillar", pos="center"), act("swap", 11.0, "caterpillar", to="pip_point")]
    assert any("only one performer drawing" in p for p in problems(both, before=before))
    assert problems([act("swap", 10.0, "pip", to="pip_point")], before=before) == []     # a new pose replaces the old


def test_each_beat_asked_for_gets_one_script():
    found, screens = play([script("b02", "meadow")], [beat()], DRAWINGS, Screen())
    assert found == ["give exactly one script for each of beats b01 to b01, in order"] and screens == []


def test_code_adds_the_key_word_the_figure_and_the_lip_sync():
    words = [TimedWord(index=i, line_index=0, text=t, start=11.0 + i, end=11.5 + i) for i, t in enumerate(["In", "1858"])]
    b = beat(type="hero", keyword="colony", keyword_at=13.5, number_text="1858")
    sung = script("b01", "meadow", act("push_in", 11.0, until=14.0), act("show", 10.0, "pip", pos="center"))
    done = finish([sung], [b], [Screen(background="meadow", on={"pip": "center"}, singer="pip")], words, 15.0, [])
    assert (done[0].start, done[0].end, done[0].type) == (10.0, 14.0, "hero")
    assert [(a.verb, a.at, a.until) for a in done[0].actions] == [
        ("show", 10.0, None), ("lipsync", 10.0, 14.0), ("push_in", 11.0, 14.0), ("number", 12.0, 14.0),
        ("keyword", 13.5, 15.0)]                             # held 2 s, past the beat but not past the song
    assert done[0].actions[1].drawing == "pip" and done[0].actions[3].text == "1858"
    plain = finish([sung], [beat(keyword="egg", number_text="Egg")], [Screen()], [], 15.0, [])
    assert [a.verb for a in plain[0].actions] == ["show", "keyword", "push_in"]      # unsung: at the beat's start
    early = finish([sung], [beat(number_text="1858")], [Screen()],
                   [TimedWord(index=0, line_index=0, text="1858", start=8.5, end=9.0)], 15.0, [])
    assert [(a.at, a.until) for a in early[0].actions if a.verb == "number"] == [(10.0, 12.0)]  # kept in its beat
    pieces = finish([sung], [beat(keyword="Met-a-MOR-pho-SIS")], [Screen()], [], 15.0, ["metamorphosis", "egg"])
    assert [a.text for a in pieces[0].actions if a.verb == "keyword"] == ["metamorphosis"]   # the key term's spelling
