# -*- coding: utf-8 -*-
"""
generator.py
============
Создаёт батчи запросов к OpenAI API, валидирует, логирует и сохраняет
корпус уровня 1|2|3 в *.csv*.  
Главные акценты:
  •   семантическая диверсификация за счёт случайного жанра (genres.py);
  •   воспроизводимость через Config + логирование в папку *logs/*;
  •   мягкая остановка SIGINT (Ctrl-C) → не теряем незаписанные батчи.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import openai
import pandas as pd

from prompts import build_prompt, TEMPERATURE
from genres import weighted_random_genre, GENRES
from validator import TextValidator


class _GracefulExit:  # pylint: disable=too-few-public-methods
    """Ловим Ctrl-C, завершаем батч корректно."""

    def __init__(self) -> None:
        self.stop = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, *_):
        logging.info("⏹  Получен сигнал остановки")
        self.stop = True


class DatasetGenerator:
    """Высоко-уровневый менеджер генерации."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.exit = _GracefulExit()

        if not cfg.api_key.startswith("sk-"):
            raise ValueError("Некорректный OpenAI API-ключ")

        self.client = openai.OpenAI(api_key=cfg.api_key)
        self.validator = TextValidator(cfg)
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(exist_ok=True)

        logging.info(
            "🟢  Инициализирован генератор (%s, жанров — %d)",
            cfg.model_name, len(GENRES)
        )

    # ------------------------------------------------------------------ #
    def test_run(self) -> None:
        """Генерируем по одному примеру на уровень 1|2|3 и печатаем."""
        for lvl in (1, 2, 3):
            genre = weighted_random_genre()
            rec = self._generate_one(lvl, genre)
            if rec:
                logging.info("✅  Test %d-star (%s): %s…", lvl, genre, rec["description"][:60])

    # ------------------------------------------------------------------ #
    def generate_level(self, level: int) -> None:
        """Основной цикл: пока не сгенерируем нужное число валидных текстов."""
        target = self.cfg.target_count_per_level
        file_csv = self.out_dir / f"level_{level}.csv"
        done = sum(1 for _ in open(file_csv, encoding="utf-8")) - 1 if file_csv.exists() else 0
        logging.info("🚀  Уровень %d: уже %d / %d", level, done, target)

        batch: list[Dict] = []

        while done < target and not self.exit.stop:
            rec = self._generate_one(level, weighted_random_genre())
            if rec:
                batch.append(rec)
                done += 1

            if len(batch) >= self.cfg.batch_size:
                self._save_batch(batch, file_csv)
                batch = []
                logging.info("📊  %d / %d готово", done, target)

        if batch:
            self._save_batch(batch, file_csv)
        logging.info("✅  Уровень %d завершён (%d описаний)", level, done)

    # ------------------------------------------------------------------ #
    def generate_all(self) -> None:
        """Проходим уровни 1→3."""
        for lvl in (1, 2, 3):
            if self.exit.stop:
                break
            self.generate_level(lvl)

    # ========================= helpers ================================== #
    def _generate_one(self, level: int, genre: str) -> Dict | None:  # noqa: C901
        """Запрос → LLM → валидация → dict или None."""
        sys_prompt, user_prompt = build_prompt(level, genre)
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE[level],
                    max_tokens=400,
                )
                text = resp.choices[0].message.content.strip()
                is_ok, msg, stats = self.validator.validate(text, level)
                if is_ok:
                    return {
                        "level": level,
                        "genre": genre,
                        "description": text,
                        "word_count": stats["word_count"],
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                logging.debug("Отбраковано: %s", msg)
                return None
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("Попытка %d | ошибка API: %s", attempt + 1, exc)
                time.sleep(2)
        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _save_batch(batch: List[Dict], path: Path) -> None:
        """Аппендим batch к csv (header — только если новый файл)."""
        df = pd.DataFrame(batch)
        df.to_csv(path, mode="a", header=not path.exists(),
                  index=False, encoding="utf-8")
