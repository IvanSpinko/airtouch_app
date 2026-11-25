import pytest

from gesture_control import config
from gesture_control.typing_test import Result, edit_distance, summarize


@pytest.mark.parametrize("a,b,d", [
    ("", "", 0), ("abc", "abc", 0), ("abc", "abd", 1), ("abc", "ab", 1),
    ("hello world", "helo wrld", 2), ("kitten", "sitting", 3),
])
def test_edit_distance(a, b, d):
    assert edit_distance(a, b) == d == edit_distance(b, a)


def test_wpm_uses_five_chars_per_word():
    r = Result("x" * 50, "x" * 50, seconds=60.0)  # 50 chars in a minute = 10 words
    assert r.wpm == pytest.approx(50 / config.CHARS_PER_WORD)
    assert r.accuracy == 1.0 and r.net_wpm == r.wpm


def test_errors_reduce_accuracy_and_net_wpm():
    r = Result("hello world", "helo world", seconds=30.0)
    assert r.errors == 1
    assert r.accuracy == pytest.approx(1 - 1 / 11)
    assert r.net_wpm < r.wpm


def test_zero_time_is_safe():
    assert Result("a", "a", 0.0).wpm == 0.0


def test_summary_reports_target():
    fast = [Result("x" * 60, "x" * 60, 30.0)]  # 24 WPM
    slow = [Result("x" * 10, "x" * 10, 60.0)]  # 2 WPM
    assert "[OK]" in summarize(fast)
    assert "[TODO]" in summarize(slow)
