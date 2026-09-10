from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from interpolation import validate_points


FUNCTIONS: Dict[str, Tuple[str, Callable[[float], float]]] = {
    "1": ("sin(x)", math.sin),
    "2": ("cos(x)", math.cos),
    "3": ("exp(x)", math.exp),
}


def read_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            print("Ошибка: введите число.")


def read_int(prompt: str, min_value: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if min_value is not None and value < min_value:
                print(f"Ошибка: значение должно быть не меньше {min_value}.")
                continue
            return value
        except ValueError:
            print("Ошибка: введите целое число.")


def manual_input() -> Tuple[List[float], List[float], None]:
    n = read_int("Количество узлов (>= 2): ", min_value=2)
    x, y = [], []

    print("Введите пары x, y. Узлы x должны идти строго по возрастанию.")
    for i in range(n):
        xi = read_float(f"x[{i}] = ")
        yi = read_float(f"y[{i}] = ")
        x.append(xi)
        y.append(yi)

    validate_points(x, y)
    return x, y, None


def load_from_file(path: str | Path) -> Tuple[List[float], List[float], None]:
    """
    Формат:
    # комментарий
    x y
    x y
    ...
    Допустимы пробел, ; или запятая как разделитель.
    Для десятичной части рекомендуется точка.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")

    x, y = [], []

    for line_no, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Сначала поддерживаем ';', затем обычный whitespace.
        if ";" in line:
            parts = [s.strip() for s in line.split(";")]
        else:
            parts = line.split()

        if len(parts) != 2:
            raise ValueError(
                f"Строка {line_no}: ожидаются ровно два значения x и y."
            )

        try:
            xi = float(parts[0].replace(",", "."))
            yi = float(parts[1].replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"Строка {line_no}: некорректное число.") from exc

        x.append(xi)
        y.append(yi)

    validate_points(x, y)
    return x, y, None


def generate_from_function():
    print("\nДоступные функции:")
    for key, (name, _) in FUNCTIONS.items():
        print(f"  {key}. {name}")

    choice = input("Выберите функцию: ").strip()
    if choice not in FUNCTIONS:
        raise ValueError("Неизвестный номер функции.")

    name, func = FUNCTIONS[choice]
    a = read_float("Левая граница a = ")
    b = read_float("Правая граница b = ")
    if a >= b:
        raise ValueError("Должно выполняться a < b.")

    n = read_int("Количество точек на интервале (>= 2): ", min_value=2)
    h = (b - a) / (n - 1)

    x = [a + i * h for i in range(n)]
    y = [func(xi) for xi in x]

    return x, y, (name, func, a, b)
