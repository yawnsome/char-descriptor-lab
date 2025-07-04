# -*- coding: utf-8 -*-
"""
utils.py
========
Вспомогательные функции: логирование и генерация post-hoc отчёта.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


def setup_logging(debug: bool = False) -> None:
    """Настраиваем ротационный logger."""
    Path("logs").mkdir(exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/generation.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def create_summary(output_dir: str = "generated_dataset") -> dict:
    """Генерируем быстрый json-отчёт: сколько текстов, средняя длина, топ-жанры."""
    out = Path(output_dir)
    report: dict = {"generated_at": datetime.utcnow().isoformat(), "levels": {}}

    for lvl in (1, 2, 3):
        csv = out / f"level_{lvl}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        report["levels"][lvl] = {
            "count": len(df),
            "avg_word_count": df["word_count"].mean(),
            "top_genres": df["genre"].value_counts().head(10).to_dict(),
        }

    with open(out / "summary_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    return report
