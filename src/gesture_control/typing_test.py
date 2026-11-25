"""Typing speed test for stage 4: words per minute with gestures + osk.exe.

Run: ``python -m gesture_control.typing_test`` in a terminal, start AirTouch, open
the on-screen keyboard with an open palm and click into this terminal. For each
phrase: press Enter to start the timer, type the phrase with pinches and press
Enter again. Words per minute use the standard 5 characters = 1 word.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from gesture_control import config


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insertions, deletions, substitutions)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class Result:
    target: str
    typed: str
    seconds: float

    @property
    def wpm(self) -> float:
        """Gross words per minute: typed characters / 5 per minute."""
        if self.seconds <= 0:
            return 0.0
        return len(self.typed) / config.CHARS_PER_WORD / (self.seconds / 60.0)

    @property
    def errors(self) -> int:
        return edit_distance(self.typed, self.target)

    @property
    def accuracy(self) -> float:
        if not self.target:
            return 1.0
        return max(0.0, 1.0 - self.errors / len(self.target))

    @property
    def net_wpm(self) -> float:
        """Words per minute counting only correct characters."""
        return self.wpm * self.accuracy


def summarize(results: list[Result]) -> str:
    total_chars = sum(len(r.typed) for r in results)
    total_s = sum(r.seconds for r in results)
    gross = total_chars / config.CHARS_PER_WORD / (total_s / 60.0) if total_s else 0.0
    acc = sum(r.accuracy * len(r.target) for r in results) / max(sum(len(r.target) for r in results), 1)
    ok = gross * acc >= config.TARGET_WPM
    return (f"overall: {gross:.1f} WPM, accuracy {acc:.0%}, net {gross * acc:.1f} WPM  "
            f"[{'OK' if ok else 'TODO'}] target >= {config.TARGET_WPM} WPM")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phrase", action="append", help="custom phrase (repeatable)")
    args = parser.parse_args()
    phrases = args.phrase or list(config.TYPING_TEST_PHRASES)

    print(__doc__)
    results = []
    for n, phrase in enumerate(phrases, 1):
        print(f"\n[{n}/{len(phrases)}]  {phrase}")
        input("Press Enter to start...")
        start = time.perf_counter()
        typed = input("> ")
        r = Result(phrase, typed.strip(), time.perf_counter() - start)
        results.append(r)
        print(f"  {r.seconds:.1f} s, {r.wpm:.1f} WPM, accuracy {r.accuracy:.0%}, "
              f"net {r.net_wpm:.1f} WPM")
    print("\n" + summarize(results))


if __name__ == "__main__":
    main()
