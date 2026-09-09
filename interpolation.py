from __future__ import annotations

from math import factorial
from typing import List, Sequence, Tuple


EPS = 1e-12


def validate_points(x: Sequence[float], y: Sequence[float]) -> None:
    if len(x) != len(y):
        raise ValueError("Количество x и y должно совпадать.")
    if len(x) < 2:
        raise ValueError("Необходимо минимум две точки.")
    if any(not isinstance(v, (int, float)) for v in list(x) + list(y)):
        raise ValueError("Все значения должны быть числами.")
    if len(set(x)) != len(x):
        raise ValueError("Узлы x не должны повторяться.")
    if any(x[i] >= x[i + 1] for i in range(len(x) - 1)):
        raise ValueError("Узлы x должны быть заданы строго по возрастанию.")


def is_equally_spaced(x: Sequence[float], tol: float = 1e-9) -> Tuple[bool, float | None]:
    if len(x) < 2:
        return False, None
    h = x[1] - x[0]
    if abs(h) < EPS:
        return False, None
    ok = all(abs((x[i] - x[i - 1]) - h) <= tol * max(1.0, abs(h))
             for i in range(2, len(x)))
    return ok, h if ok else None


# ---------------------------------------------------------------------------
# 1. Многочлен Лагранжа
# ---------------------------------------------------------------------------

def lagrange_value(x_nodes: Sequence[float], y_nodes: Sequence[float], x: float) -> float:
    """
    L_n(x) = sum(y_i * l_i(x)),
    l_i(x) = product_{j != i} (x - x_j) / (x_i - x_j)
    """
    validate_points(x_nodes, y_nodes)
    n = len(x_nodes)
    result = 0.0

    for i in range(n):
        li = 1.0
        for j in range(n):
            if i != j:
                li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * li

    return result


# ---------------------------------------------------------------------------
# 2. Разделенные разности и многочлен Ньютона
# ---------------------------------------------------------------------------

def divided_difference_table(
    x_nodes: Sequence[float], y_nodes: Sequence[float]
) -> List[List[float | None]]:
    """
    table[i][0] = y_i
    table[i][k] = f[x_i, ..., x_{i+k}]
    """
    validate_points(x_nodes, y_nodes)
    n = len(x_nodes)
    table: List[List[float | None]] = [[None] * n for _ in range(n)]

    for i in range(n):
        table[i][0] = float(y_nodes[i])

    for order in range(1, n):
        for i in range(n - order):
            numerator = table[i + 1][order - 1] - table[i][order - 1]
            denominator = x_nodes[i + order] - x_nodes[i]
            table[i][order] = numerator / denominator

    return table


def newton_divided_forward(
    x_nodes: Sequence[float], y_nodes: Sequence[float], x: float
) -> float:
    """
    Первая форма Ньютона с разделенными разностями (вперед):
    N_n(x) = f(x0) + f[x0,x1](x-x0) + ...
    """
    table = divided_difference_table(x_nodes, y_nodes)
    n = len(x_nodes)

    result = float(table[0][0])
    product = 1.0

    for order in range(1, n):
        product *= x - x_nodes[order - 1]
        result += float(table[0][order]) * product

    return result


def newton_divided_backward(
    x_nodes: Sequence[float], y_nodes: Sequence[float], x: float
) -> float:
    """
    Вторая форма Ньютона с разделенными разностями (назад):
    N_n(x) = f(xn) + f[x_{n-1},xn](x-xn) + ...
    """
    table = divided_difference_table(x_nodes, y_nodes)
    n = len(x_nodes)

    result = float(table[n - 1][0])
    product = 1.0

    for order in range(1, n):
        product *= x - x_nodes[n - order]
        coefficient = float(table[n - 1 - order][order])
        result += coefficient * product

    return result


# ---------------------------------------------------------------------------
# 3. Конечные разности и формулы Ньютона для равноотстоящих узлов
# ---------------------------------------------------------------------------

def finite_difference_table(y_nodes: Sequence[float]) -> List[List[float | None]]:
    """
    table[i][0] = y_i
    table[i][1] = Δy_i
    table[i][2] = Δ²y_i
    ...
    """
    n = len(y_nodes)
    if n < 2:
        raise ValueError("Необходимо минимум два значения y.")

    table: List[List[float | None]] = [[None] * n for _ in range(n)]

    for i, value in enumerate(y_nodes):
        table[i][0] = float(value)

    for order in range(1, n):
        for i in range(n - order):
            table[i][order] = table[i + 1][order - 1] - table[i][order - 1]

    return table


def newton_finite_forward(
    x_nodes: Sequence[float], y_nodes: Sequence[float], x: float
) -> float:
    """
    Первая интерполяционная формула Ньютона для равноотстоящих узлов:
    t = (x-x0)/h
    N = y0 + t Δy0 + t(t-1)/2! Δ²y0 + ...
    """
    validate_points(x_nodes, y_nodes)
    equal, h = is_equally_spaced(x_nodes)
    if not equal or h is None:
        raise ValueError(
            "Формула Ньютона с конечными разностями требует равноотстоящих узлов."
        )

    table = finite_difference_table(y_nodes)
    t = (x - x_nodes[0]) / h

    result = float(y_nodes[0])
    t_product = 1.0

    for order in range(1, len(x_nodes)):
        t_product *= t - (order - 1)
        result += (t_product / factorial(order)) * float(table[0][order])

    return result


def newton_finite_backward(
    x_nodes: Sequence[float], y_nodes: Sequence[float], x: float
) -> float:
    """
    Вторая интерполяционная формула Ньютона для равноотстоящих узлов:
    t = (x-xn)/h
    N = yn + t Δy_{n-1} + t(t+1)/2! Δ²y_{n-2} + ...
    """
    validate_points(x_nodes, y_nodes)
    equal, h = is_equally_spaced(x_nodes)
    if not equal or h is None:
        raise ValueError(
            "Формула Ньютона с конечными разностями требует равноотстоящих узлов."
        )

    table = finite_difference_table(y_nodes)
    n = len(x_nodes)
    t = (x - x_nodes[-1]) / h

    result = float(y_nodes[-1])
    t_product = 1.0

    for order in range(1, n):
        t_product *= t + (order - 1)
        delta = float(table[n - 1 - order][order])
        result += (t_product / factorial(order)) * delta

    return result


def recommended_finite_newton_formula(x_nodes: Sequence[float], x: float) -> str:
    """
    По лекции:
    - для левой половины отрезка удобна первая формула Ньютона;
    - для правой половины — вторая.
    Для экстраполяции:
    - x < x0 -> первая;
    - x > xn -> вторая.
    """
    midpoint = (x_nodes[0] + x_nodes[-1]) / 2.0
    return "forward" if x <= midpoint else "backward"


# ---------------------------------------------------------------------------
# Представление таблиц
# ---------------------------------------------------------------------------

def format_divided_table(
    x_nodes: Sequence[float], table: Sequence[Sequence[float | None]]
) -> str:
    n = len(x_nodes)
    headers = ["x", "f(x)"] + [f"Δр.{k}" for k in range(1, n)]
    rows = []

    for i in range(n):
        row = [f"{x_nodes[i]:.8g}"]
        for order in range(n):
            value = table[i][order]
            row.append("" if value is None else f"{value:.10g}")
        rows.append(row)

    return _ascii_table(headers, rows)


def format_finite_table(
    x_nodes: Sequence[float], table: Sequence[Sequence[float | None]]
) -> str:
    n = len(x_nodes)
    headers = ["x", "y"] + [f"Δ^{k}y" for k in range(1, n)]
    rows = []

    for i in range(n):
        row = [f"{x_nodes[i]:.8g}"]
        for order in range(n):
            value = table[i][order]
            row.append("" if value is None else f"{value:.10g}")
        rows.append(row)

    return _ascii_table(headers, rows)


def _ascii_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    def render_row(row):
        return " | ".join(str(cell).ljust(widths[j]) for j, cell in enumerate(row))

    line = "-+-".join("-" * w for w in widths)
    return "\n".join([render_row(headers), line] + [render_row(r) for r in rows])
