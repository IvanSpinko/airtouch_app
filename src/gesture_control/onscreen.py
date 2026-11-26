"""Custom on-screen keyboard (stage 6): large keys, cursor snapping, word suggestions.

Parts:

- layouts and key geometry (``LAYOUTS``, ``build_keys``, ``keyboard_rect``) — pure
  functions in physical screen pixels, testable without Qt;
- ``Suggester`` — completes the word being typed from a word frequency list;
- ``KeyboardModel`` — what a key press does (Shift, language, current word);
- ``choose_key`` — cursor snapping to the nearest key with hysteresis;
- ``OnscreenKeyboard`` — thread-safe facade: the engine thread snaps the cursor
  and presses keys, the Qt window (``KeyboardWindow``) only draws it;
- ``KeySender`` — sends characters as Unicode (independent of the Windows input
  language) and Backspace / Enter through pynput.

Key presses are sent to the window that has the keyboard focus. The keyboard
window never takes the focus, so text goes to the app the user is working in.
No administrator rights are needed (unlike osk.exe).

Check: ``python -m gesture_control.onscreen`` shows the keyboard; click keys with
the real mouse and the text goes to the focused window.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gesture_control import config

# Special keys (all other entries of a layout row are characters).
BACKSPACE, ENTER, SHIFT, LANG, SPACE = "backspace", "enter", "shift", "lang", "space"
SUGGEST = "suggest"
SPECIAL_KEYS = (BACKSPACE, ENTER, SHIFT, LANG, SPACE)

# Rows of each layout, top to bottom (the suggestion row is added above them).
LAYOUTS: dict[str, list[list[str]]] = {
    "en": [
        [*"1234567890", BACKSPACE],
        [*"qwertyuiop"],
        [*"asdfghjkl", ENTER],
        [SHIFT, *"zxcvbnm", ",", "."],
        [LANG, "'", "?", SPACE, "!", "-"],
    ],
    "ru": [
        [*"1234567890", BACKSPACE],
        [*"йцукенгшщзхъ"],
        [*"фывапролджэ", ENTER],
        [SHIFT, *"ячсмитьбю", ",", "."],
        [LANG, "ё", "?", SPACE, "!", "-"],
    ],
}
LAYOUT_NAMES = {"en": "EN", "ru": "RU"}
SPECIAL_LABELS = {BACKSPACE: "⌫", ENTER: "Enter", SHIFT: "⇧ Shift", SPACE: "Space"}
# Characters that continue a word (others end it and reset the suggestions).
WORD_CHARS_EXTRA = "'-"


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def grown(self, margin: float) -> "Rect":
        return Rect(self.x - margin, self.y - margin, self.w + 2 * margin, self.h + 2 * margin)


@dataclass(frozen=True)
class Key:
    kind: str          # "char", one of SPECIAL_KEYS, or SUGGEST
    value: str | int   # the character, the special key name, or the suggestion index
    rect: Rect         # physical screen pixels, gaps already removed


def key_units(name: str) -> float:
    return config.ONSCREEN_WIDE_KEYS.get(name, 1.0)


def keyboard_rect(screen_w: int, screen_h: int) -> Rect:
    """Keyboard window: centered at the bottom of the screen."""
    w = screen_w * config.ONSCREEN_WIDTH_FRACTION
    h = screen_h * config.ONSCREEN_HEIGHT_FRACTION
    x = (screen_w - w) / 2
    y = screen_h - h - screen_h * config.ONSCREEN_BOTTOM_MARGIN_FRACTION
    return Rect(round(x), round(y), round(w), round(h))


def key_height(area: Rect, lang: str) -> float:
    rows = len(LAYOUTS[lang]) + config.ONSCREEN_SUGGESTION_ROW_UNITS
    return (area.h - 2 * config.ONSCREEN_PAD_PX) / rows


def build_keys(lang: str, area: Rect) -> list[Key]:
    """All keys of ``lang`` laid out in ``area``: suggestions first, then the rows."""
    rows = LAYOUTS[lang]
    pad = config.ONSCREEN_PAD_PX
    inner_w = area.w - 2 * pad
    kh = key_height(area, lang)
    unit = inner_w / max(sum(key_units(k) for k in row) for row in rows)
    gap = min(unit, kh) * config.ONSCREEN_GAP_FRACTION

    def cell(x: float, y: float, w: float, h: float) -> Rect:
        return Rect(x + gap / 2, y + gap / 2, w - gap, h - gap)

    keys: list[Key] = []
    sh = kh * config.ONSCREEN_SUGGESTION_ROW_UNITS
    sw = inner_w / config.ONSCREEN_SUGGESTIONS
    for i in range(config.ONSCREEN_SUGGESTIONS):
        keys.append(Key(SUGGEST, i, cell(area.x + pad + i * sw, area.y + pad, sw, sh)))
    y = area.y + pad + sh
    for row in rows:
        row_w = sum(key_units(k) for k in row) * unit
        x = area.x + pad + (inner_w - row_w) / 2
        for name in row:
            w = key_units(name) * unit
            kind = name if name in SPECIAL_KEYS else "char"
            keys.append(Key(kind, name, cell(x, y, w, kh)))
            x += w
        y += kh
    return keys


def choose_key(
    point: tuple[float, float], keys: list[Key], area: Rect, current: int | None,
    key_h: float,
) -> int | None:
    """Index of the key the cursor snaps to, or None outside the keyboard.

    The nearest key center wins, but the current key is kept unless another
    center is closer by more than ``ONSCREEN_SNAP_HYSTERESIS`` key heights, so the
    highlight does not flicker between two neighbours.
    """
    px, py = point
    if not keys or not area.grown(config.ONSCREEN_SNAP_MARGIN * key_h).contains(px, py):
        return None

    def dist(i: int) -> float:
        cx, cy = keys[i].rect.center
        return ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

    best = min(range(len(keys)), key=dist)
    if current is not None and 0 <= current < len(keys) and current != best:
        if dist(current) - dist(best) <= config.ONSCREEN_SNAP_HYSTERESIS * key_h:
            return current
    return best


# -- word suggestions -------------------------------------------------------------
def load_words(path: Path) -> list[str]:
    """One word per line, most frequent first; '#' lines are comments."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    seen: set[str] = set()
    words = []
    for line in lines:
        word = line.strip().lower()
        if word and not word.startswith("#") and word not in seen:
            seen.add(word)
            words.append(word)
    return words


class Suggester:
    """Most frequent words that start with the typed prefix."""

    def __init__(self, words: list[str]) -> None:
        self.words = words

    def suggest(self, prefix: str, n: int = config.ONSCREEN_SUGGESTIONS) -> list[str]:
        if not prefix:
            return []
        low = prefix.lower()
        found = [w for w in self.words if len(w) > len(low) and w.startswith(low)][:n]
        if prefix[0].isupper():
            found = [w[0].upper() + w[1:] for w in found]
        return found


# -- what a key press does ----------------------------------------------------------
# Output actions: ("text", "abc") types characters, ("key", BACKSPACE / ENTER) a key.
Action = tuple[str, str]


class KeyboardModel:
    """Layout, Shift, the word being typed and its suggestions. Not thread-safe."""

    def __init__(self, suggesters: dict[str, Suggester], lang: str | None = None) -> None:
        self.suggesters = suggesters
        self.lang = lang or config.ONSCREEN_LAYOUT_DEFAULT
        self.shift = False
        self.word = ""
        self.suggestions: list[str] = []

    def label(self, key: Key) -> str:
        if key.kind == "char":
            return key.value.upper() if self.shift else key.value
        if key.kind == LANG:
            return LAYOUT_NAMES[self.lang]
        if key.kind == SUGGEST:
            return self.suggestions[key.value] if key.value < len(self.suggestions) else ""
        return SPECIAL_LABELS[key.kind]

    def press(self, key: Key) -> list[Action]:
        out: list[Action] = []
        if key.kind == "char":
            ch = key.value.upper() if self.shift else key.value
            self.shift = False
            out.append(("text", ch))
            self.word = self.word + ch if ch.isalpha() or ch in WORD_CHARS_EXTRA else ""
        elif key.kind == SPACE:
            out.append(("text", " "))
            self.word = ""
        elif key.kind == ENTER:
            out.append(("key", ENTER))
            self.word = ""
        elif key.kind == BACKSPACE:
            out.append(("key", BACKSPACE))
            self.word = self.word[:-1]
        elif key.kind == SHIFT:
            self.shift = not self.shift
        elif key.kind == LANG:
            names = list(LAYOUTS)
            self.lang = names[(names.index(self.lang) + 1) % len(names)]
            self.word = ""
        elif key.kind == SUGGEST:
            if key.value < len(self.suggestions):
                word = self.suggestions[key.value]
                out.append(("text", word[len(self.word):] + " "))
                self.word = ""
                self.shift = False
        suggester = self.suggesters.get(self.lang)
        self.suggestions = suggester.suggest(self.word) if suggester else []
        return out


# -- output -------------------------------------------------------------------------
class Sender(Protocol):
    def text(self, s: str) -> None: ...
    def key(self, name: str) -> None: ...


class KeySender:
    """Real key presses through pynput (characters go out as Unicode)."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key as PynputKey

        self._ctl = Controller()
        self._keys = {BACKSPACE: PynputKey.backspace, ENTER: PynputKey.enter}

    def text(self, s: str) -> None:
        self._ctl.type(s)

    def key(self, name: str) -> None:
        k = self._keys[name]
        self._ctl.press(k)
        self._ctl.release(k)


# -- thread-safe facade ---------------------------------------------------------------
@dataclass(frozen=True)
class View:
    """What the window draws, copied under the lock."""

    visible: bool = False
    area: Rect | None = None
    keys: tuple[Key, ...] = ()
    labels: tuple[str, ...] = ()
    hovered: int | None = None
    flash: int | None = None
    shift: bool = False
    version: int = 0


@dataclass
class _Flash:
    index: int | None = None
    until: float = 0.0


class OnscreenKeyboard:
    """Shared by the engine thread (snapping, presses) and the Qt window (drawing)."""

    def __init__(self, sender: Sender | None = None,
                 suggesters: dict[str, Suggester] | None = None) -> None:
        if suggesters is None:
            suggesters = {lang: Suggester(load_words(p)) for lang, p in config.WORDS_PATHS.items()}
        self._sender = sender
        self._lock = threading.RLock()
        self.model = KeyboardModel(suggesters)
        self._visible = False
        self._area: Rect | None = None
        self._keys: list[Key] = []
        self._key_h = 0.0
        self._hovered: int | None = None
        self._flash = _Flash()
        self._version = 0

    # -- geometry / visibility ---------------------------------------------------------
    def set_screen(self, width: int, height: int) -> None:
        with self._lock:
            self._area = keyboard_rect(width, height)
            self._relayout()

    def _relayout(self) -> None:
        if self._area is None:
            return
        self._keys = build_keys(self.model.lang, self._area)
        self._key_h = key_height(self._area, self.model.lang)
        self._hovered = None
        self._changed()

    def _changed(self) -> None:
        self._version += 1

    @property
    def visible(self) -> bool:
        with self._lock:
            return self._visible

    def set_visible(self, value: bool) -> None:
        with self._lock:
            if self._visible != value:
                self._visible = value
                self._hovered = None
                self._changed()

    def toggle(self) -> bool:
        with self._lock:
            self.set_visible(not self._visible)
            return self._visible

    # -- used by the controller -------------------------------------------------------
    def hover(self, x: float, y: float) -> tuple[int, int] | None:
        """Highlights the key the cursor snaps to; returns its center or None."""
        with self._lock:
            if not self._visible or self._area is None:
                return None
            index = choose_key((x, y), self._keys, self._area, self._hovered, self._key_h)
            if index != self._hovered:
                self._hovered = index
                self._changed()
            if index is None:
                return None
            cx, cy = self._keys[index].rect.center
            return round(cx), round(cy)

    def leave(self) -> None:
        with self._lock:
            if self._hovered is not None:
                self._hovered = None
                self._changed()

    def press(self) -> bool:
        """Presses the highlighted key; False if no key is highlighted."""
        with self._lock:
            if not self._visible or self._hovered is None:
                return False
            self._press_index(self._hovered)
            return True

    def press_at(self, x: float, y: float) -> bool:
        """Presses the key under a real mouse click (physical pixels)."""
        with self._lock:
            for i, key in enumerate(self._keys):
                if key.rect.contains(x, y):
                    self._press_index(i)
                    return True
            return False

    def _press_index(self, index: int) -> None:
        key = self._keys[index]
        lang = self.model.lang
        actions = self.model.press(key)
        self._flash = _Flash(index, time.monotonic() + config.ONSCREEN_PRESS_FLASH_S)
        if self._sender is None:
            self._sender = KeySender()
        for kind, value in actions:
            if kind == "text":
                self._sender.text(value)
            else:
                self._sender.key(value)
        if self.model.lang != lang:
            self._relayout()
        self._changed()

    # -- used by the window ---------------------------------------------------------------
    def view(self) -> View:
        with self._lock:
            flash = self._flash.index if time.monotonic() < self._flash.until else None
            return View(
                visible=self._visible, area=self._area, keys=tuple(self._keys),
                labels=tuple(self.model.label(k) for k in self._keys),
                hovered=self._hovered, flash=flash, shift=self.model.shift,
                version=self._version,
            )


def _make_window(keyboard: OnscreenKeyboard):
    from gesture_control.onscreen_window import KeyboardWindow

    return KeyboardWindow(keyboard)


def _demo() -> None:
    import sys

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from gesture_control.mouse import enable_dpi_awareness, screen_size

    enable_dpi_awareness()
    app = QApplication(sys.argv)
    kb = OnscreenKeyboard()
    kb.set_screen(*screen_size())
    kb.set_visible(True)
    window = _make_window(kb)  # noqa: F841 - keeps the window alive
    print("Click keys with the mouse; text goes to the focused window. Ctrl+C or wait to exit.")
    QTimer.singleShot(config.ONSCREEN_DEMO_EXIT_MS, app.quit)
    timer = QTimer()
    timer.timeout.connect(lambda: None)  # lets Python handle Ctrl+C
    timer.start(config.UI_REFRESH_MS)
    import signal

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    app.exec()


if __name__ == "__main__":
    _demo()
