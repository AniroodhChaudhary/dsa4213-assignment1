"""Tokeniser that maps raw text (AG News) onto WikiText-2's token conventions.

WikiText-2 was produced with the Moses tokeniser and then post-processed so that
hyphens inside words become ` @-@ `, thousands separators become ` @,@ ` and
decimal points become ` @.@ `. We reproduce that with sacremoses plus two regexes,
and lowercase because the whole project uses a lowercased vocabulary.
"""
from __future__ import annotations

import html
import re

from sacremoses import MosesTokenizer

_BARE_ENTITY = re.compile(r"(?<!&)#(\d+);")  # AG News has ' #39;s' style fragments
_NUM_COMMA = re.compile(r"(?<=\d),(?=\d)")
_NUM_DOT = re.compile(r"(?<=\d)\.(?=\d)")


class WikiTextTokenizer:
    def __init__(self) -> None:
        self._mt = MosesTokenizer(lang="en")

    def clean(self, text: str) -> str:
        text = _BARE_ENTITY.sub(lambda m: chr(int(m.group(1))), text)
        text = html.unescape(text)
        return text.replace("\\", " ")

    def __call__(self, text: str) -> list[str]:
        toks = self._mt.tokenize(self.clean(text), aggressive_dash_splits=True, escape=False)
        out: list[str] = []
        for t in toks:
            t = _NUM_COMMA.sub(" @,@ ", t.lower())
            t = _NUM_DOT.sub(" @.@ ", t)
            out.extend(t.split())
        return out
