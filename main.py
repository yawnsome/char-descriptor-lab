#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
========
Запуск генерации синтетического корпуса описаний **персонажей** для проверки
Гипотезы 1 диссертационной работы «Автоматизированная оценка описаний
в художественной прозе».

Основные сценарии:
    • --level 1|2|3     ─ сгенерировать только указанный уровень «звёздности»;
    • --count N         ─ переопределить количество описаний на уровень
                          (по умолчанию 3 000 для статистической устойчивости);
    • --test            ─ одиночная пробная генерация (1 текст на каждый уровень);
    • --debug           ─ подробные логи, стек вызовов при ошибках.

Логическая связь с работой:
    1. Генерация → валидация (validator.py) → csv-датасеты уровня 1|2|3
    2. Датасеты позже используются в главе 3 для обучения лёгкого ML-классификатора
       и сравниваются с zero-shot LLM (Гипотеза 2).
"""

import argparse
import logging
import sys
from pathlib import Path

from config import Config
from generator import DatasetGenerator
from utils import setup_logging


def main() -> None:
    """CLI-обёртка над DatasetGenerator."""
    parser = argparse.ArgumentParser(
        prog="dataset_generator",
        description="Генерация синтетического корпуса описаний персонажей",
    )
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3],
        help="Генерировать только указанный уровень качества"
    )
    parser.add_argument(
        "--count", type=int, default=3000,
        help="Количество описаний на уровень (по умолчанию 3000)"
    )
    parser.add_argument("--debug", action="store_true", help="Подробная отладка")
    parser.add_argument("--test", action="store_true", help="Тестовый запуск")

    args = parser.parse_args()
    setup_logging(debug=args.debug)

    try:
        cfg = Config()  # ← загружаем все параметры (API-ключ, t-границы и т. д.)
        if args.count != 3000:
            cfg.target_count_per_level = args.count

        generator = DatasetGenerator(cfg)

        if args.test:
            generator.test_run()
        elif args.level:
            generator.generate_level(args.level)
        else:
            generator.generate_all()

    except KeyboardInterrupt:
        logging.info("⏹  Генерация прервана пользователем")
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Необработанная ошибка: %s", exc)
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
