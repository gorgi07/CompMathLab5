from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from interpolation import newton_divided_forward


def plot_interpolation(
    x_nodes: Sequence[float],
    y_nodes: Sequence[float],
    target_x: float,
    source_function: Optional[Tuple[str, Callable[[float], float], float, float]] = None,
) -> None:
    left, right = min(x_nodes[0], target_x), max(x_nodes[-1], target_x)
    margin = max((right - left) * 0.08, 0.1)
    grid = np.linspace(left - margin, right + margin, 600)

    interp_y = [
        newton_divided_forward(x_nodes, y_nodes, float(x))
        for x in grid
    ]
    target_y = newton_divided_forward(x_nodes, y_nodes, target_x)

    plt.figure(figsize=(10, 6))

    if source_function is not None:
        name, func, a, b = source_function
        exact_grid = np.linspace(a, b, 600)
        exact_y = [func(float(x)) for x in exact_grid]
        plt.plot(exact_grid, exact_y, color="blue", label=f"Исходная функция {name}")

    plt.plot(grid, interp_y, color="red", label="Интерполяционный многочлен Ньютона")
    plt.scatter(x_nodes, y_nodes, color="black", marker="o", zorder=5, label="Узлы интерполяции")
    plt.scatter(
        [target_x],
        [target_y],
        color="orange",
        edgecolor="black",
        marker="*",
        s=180,
        zorder=6,
        label=f"Точка интерполяции x = {target_x:.6g}",
    )
    plt.axvline(target_x, color="orange", linestyle=":", alpha=0.6)

    # Для табличного ввода неизвестна исходная функция между узлами,
    # поэтому соединяем узлы только как визуальный ориентир.
    if source_function is None:
        plt.plot(
            x_nodes,
            y_nodes,
            color="green",
            linestyle="--",
            alpha=0.65,
            label="Табличные данные",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Интерполяция")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
