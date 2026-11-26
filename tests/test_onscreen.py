"""Custom on-screen keyboard: layout, snapping, typing, suggestions (no Qt, no real keys)."""

import pytest

from gesture_control import config
from gesture_control.onscreen import (
    BACKSPACE, ENTER, LANG, LAYOUTS, SHIFT, SPACE, SUGGEST, KeyboardModel, OnscreenKeyboard,
    Rect, Suggester, build_keys, choose_key, key_height, keyboard_rect, load_words,
)

SCREEN = (1920, 1080)


class FakeSender:
    def __init__(self):
        self.out = []

    def text(self, s):
        self.out.append(s)

    def key(self, name):
        self.out.append(f"<{name}>")

    @property
    def typed(self):
        return "".join(self.out)


def make_keyboard(words_en=("hello", "help", "world"), words_ru=("привет",)):
    sender = FakeSender()
    kb = OnscreenKeyboard(sender, {"en": Suggester(list(words_en)), "ru": Suggester(list(words_ru))})
    kb.set_screen(*SCREEN)
    kb.set_visible(True)
    return kb, sender


def key_index(kb, kind, value=None):
    for i, k in enumerate(kb.view().keys):
        if k.kind == kind and (value is None or k.value == value):
            return i
    raise KeyError((kind, value))


def tap(kb, kind, value=None):
    """Hovers the center of a key and presses it, like the controller does."""
    k = kb.view().keys[key_index(kb, kind, value)]
    cx, cy = k.rect.center
    assert kb.hover(cx, cy) is not None
    assert kb.press()


# -- geometry ----------------------------------------------------------------------
def test_keyboard_is_at_the_bottom_center():
    r = keyboard_rect(*SCREEN)
    assert r.y + r.h < SCREEN[1] and r.y > SCREEN[1] / 2
    assert r.x == pytest.approx(SCREEN[0] - r.x - r.w, abs=1)


@pytest.mark.parametrize("lang", list(LAYOUTS))
def test_keys_fit_inside_and_do_not_overlap(lang):
    area = keyboard_rect(*SCREEN)
    keys = build_keys(lang, area)
    for k in keys:
        assert area.x <= k.rect.x and k.rect.x + k.rect.w <= area.x + area.w + 1e-6
        assert area.y <= k.rect.y and k.rect.y + k.rect.h <= area.y + area.h + 1e-6
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap_x = min(a.rect.x + a.rect.w, b.rect.x + b.rect.w) - max(a.rect.x, b.rect.x)
            overlap_y = min(a.rect.y + a.rect.h, b.rect.y + b.rect.h) - max(a.rect.y, b.rect.y)
            assert overlap_x <= 0 or overlap_y <= 0


@pytest.mark.parametrize("lang", list(LAYOUTS))
def test_every_layout_has_the_required_keys(lang):
    kinds = {k.kind for k in build_keys(lang, keyboard_rect(*SCREEN))}
    assert {BACKSPACE, ENTER, SHIFT, LANG, SPACE, SUGGEST, "char"} <= kinds
    chars = {k.value for k in build_keys(lang, keyboard_rect(*SCREEN)) if k.kind == "char"}
    assert set("0123456789,.?!") <= chars


def test_keys_are_large():
    area = keyboard_rect(*SCREEN)
    assert key_height(area, "en") > SCREEN[1] * 0.05


# -- snapping ------------------------------------------------------------------------
def test_snaps_to_nearest_key_and_none_outside():
    area = keyboard_rect(*SCREEN)
    keys = build_keys("en", area)
    kh = key_height(area, "en")
    q = next(i for i, k in enumerate(keys) if k.value == "q")
    cx, cy = keys[q].rect.center
    assert choose_key((cx + 3, cy - 2), keys, area, None, kh) == q
    assert choose_key((10, 10), keys, area, q, kh) is None


def test_hysteresis_keeps_current_key_near_the_border():
    area = keyboard_rect(*SCREEN)
    keys = build_keys("en", area)
    kh = key_height(area, "en")
    q = next(i for i, k in enumerate(keys) if k.value == "q")
    w = next(i for i, k in enumerate(keys) if k.value == "w")
    (qx, qy), (wx, _) = keys[q].rect.center, keys[w].rect.center
    just_past_middle = ((qx + wx) / 2 + 2, qy)
    assert choose_key(just_past_middle, keys, area, None, kh) == w
    assert choose_key(just_past_middle, keys, area, q, kh) == q
    assert choose_key((wx, qy), keys, area, q, kh) == w


def test_hover_returns_key_center_only_while_visible():
    kb, _ = make_keyboard()
    k = kb.view().keys[key_index(kb, "char", "a")]
    assert kb.hover(k.rect.x + 2, k.rect.y + 2) == tuple(round(c) for c in k.rect.center)
    kb.set_visible(False)
    assert kb.hover(*k.rect.center) is None
    assert not kb.press()


# -- typing ----------------------------------------------------------------------------
def test_type_letters_shift_space_backspace_enter():
    kb, sender = make_keyboard()
    tap(kb, SHIFT)
    for ch in "hi":
        tap(kb, "char", ch)
    tap(kb, SPACE)
    tap(kb, "char", "x")
    tap(kb, BACKSPACE)
    tap(kb, ENTER)
    assert sender.typed == "Hi x<backspace><enter>"


def test_language_switch_types_russian():
    kb, sender = make_keyboard()
    tap(kb, LANG)
    assert kb.model.lang == "ru"
    tap(kb, "char", "д")
    tap(kb, "char", "а")
    assert sender.typed == "да"
    tap(kb, LANG)
    assert kb.model.lang == "en"


def test_suggestion_completes_the_word_and_adds_a_space():
    kb, sender = make_keyboard()
    tap(kb, "char", "h")
    tap(kb, "char", "e")
    assert kb.model.suggestions == ["hello", "help"]
    assert kb.view().labels[key_index(kb, SUGGEST, 1)] == "help"
    tap(kb, SUGGEST, 1)
    assert sender.typed == "help "
    assert kb.model.word == "" and kb.model.suggestions == []


def test_empty_suggestion_does_nothing():
    kb, sender = make_keyboard()
    tap(kb, SUGGEST, 0)
    assert sender.typed == ""


def test_press_at_for_real_mouse_clicks():
    kb, sender = make_keyboard()
    k = kb.view().keys[key_index(kb, "char", "z")]
    assert kb.press_at(k.rect.x + 1, k.rect.y + 1)
    assert not kb.press_at(0, 0)
    assert sender.typed == "z"


# -- suggestions ---------------------------------------------------------------------
def test_suggester_order_case_and_limit():
    s = Suggester(["the", "they", "then", "there", "theory"])
    assert s.suggest("the") == ["they", "then", "there"]
    assert s.suggest("The") == ["They", "Then", "There"]
    assert s.suggest("") == []


def test_word_lists_ship_with_the_app():
    for lang, path in config.WORDS_PATHS.items():
        words = load_words(path)
        assert len(words) > 200, lang
        assert all(w == w.strip().lower() and not w.startswith("#") for w in words)


def test_load_words_missing_file(tmp_path):
    assert load_words(tmp_path / "none.txt") == []


def test_model_word_tracking():
    m = KeyboardModel({"en": Suggester(["can", "cannot"])})
    area = keyboard_rect(*SCREEN)
    keys = {k.value: k for k in build_keys("en", area)}
    for ch in "ca":
        m.press(keys[ch])
    assert m.word == "ca" and m.suggestions == ["can", "cannot"]
    m.press(keys[","])
    assert m.word == "" and m.suggestions == []
    assert Rect(0, 0, 10, 10).contains(5, 5)
