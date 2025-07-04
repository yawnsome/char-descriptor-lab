# -*- coding: utf-8 -*-
"""
config.py
=========
Собирает все гиперпараметры, чтобы:
  •   обеспечить репродуцируемость эксперимента (см. требования GEP-11);
  •   позволить быстро переиспользовать код для генерации описаний локаций
      (подглава 2.3.3).
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    # --- Параметры LLM -------------------------------------------------------
    api_key: str = field(repr=False, default="YOUR_OPENAI_KEY")
    model_name: str = "gpt-4o-2024-08-06"

    # --- Объём и контроль генерации -----------------------------------------
    target_count_per_level: int = 3000
    batch_size: int = 10
    max_retries: int = 5

    # Температуры эмпирически подобраны под требования § 2.3.2
    temperature_1: float = 0.2
    temperature_2: float = 0.6
    temperature_3: float = 0.8

    # --- Валидационные пороги (см. validator.py) ----------------------------
    min_words: Dict[int, int] = None
    max_words: Dict[int, int] = None

    # --- Пути вывода ---------------------------------------------------------
    output_dir: str = "generated_dataset"

    # ------------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Инициализируем пороги, создаём директорию вывода."""
        self.min_words = self.min_words or {1: 50, 2: 120, 3: 180}
        self.max_words = self.max_words or {1: 120, 2: 250, 3: 350}
        Path(self.output_dir).mkdir(exist_ok=True)

    # ------------------------------------------------------------------------
    def save(self, path: str | Path = "config.json") -> None:
        """Сохраняем конфигурацию для контроля версий и чекпоинтов."""
        serialisable = {
            "model_name": self.model_name,
            "target_count_per_level": self.target_count_per_level,
            "batch_size": self.batch_size,
            "temperatures": {
                "1": self.temperature_1, "2": self.temperature_2, "3": self.temperature_3
            },
            "validation": {"min_words": self.min_words, "max_words": self.max_words},
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(serialisable, fp, indent=2, ensure_ascii=False)
