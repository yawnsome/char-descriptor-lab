# -*- coding: utf-8 -*-
"""
validator.py
============
Отбрасывает GPT-«метакомментарии» и очевидный мусор, но максимально
сохраняет лексико-семантическое разнообразие для дальнейшего обучения.
Это соответствует принципу *weak supervision* (§ 2.3.2).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import nltk  # для счётчика предложений

# подгружаем токенизатор
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class TextValidator:  # pylint: disable=too-few-public-methods
    """Минимальная, но воспроизводимая проверка качества синтетического текста."""

    _GARBAGE_RE = [
        # три группы: метакомментарии, технические указания, извинения/пояснения
        r"\b(?:я\s+)?(?:создам|опишу|напишу)\b",
        r"\bструктура\s+описания\b",
        r"\bизвините|к\s+сожалению\b",
    ]
    _GARBAGE = [re.compile(p, flags=re.I) for p in _GARBAGE_RE]

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    # --------------------------------------------------------------------- #
    def validate(self, text: str, level: int) -> Tuple[bool, str, Dict]:
        """Возвращает (is_valid, msg, quick_stats)."""
        txt = text.strip()

        # критические отказы
        if len(txt) < 10:
            return False, "too_short", {}
        if any(p.search(txt) for p in self._GARBAGE):
            return False, "meta_garbage", {}

        words = re.findall(r"\b[а-яё]+\b", txt.lower())
        if len(words) < 8:
            return False, "not_enough_russian_words", {}

        # базовые метрики
        sents = [s for s in re.split(r"[.!?]+", txt) if s.strip()]
        stats = {
            "word_count": len(words),
            "sentence_count": len(sents),
            "avg_sent_len": len(words) / max(len(sents), 1),
        }

        # мягкие пороги уровня
        if not self.cfg.min_words[level] <= stats["word_count"] <= self.cfg.max_words[level]:
            return False, "len_out_of_bounds", stats

        return True, "ok", stats
