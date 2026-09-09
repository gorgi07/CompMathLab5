import math
import pytest

from interpolation import (
    divided_difference_table,
    finite_difference_table,
    is_equally_spaced,
    lagrange_value,
    newton_divided_backward,
    newton_divided_forward,
    newton_finite_backward,
    newton_finite_forward,
)


def test_all_methods_equal_on_uniform_grid():
    x = [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65]
    y = [0.1213, 1.1316, 2.1459, 3.1565, 4.1571, 5.1819, 6.1969]
    point = 1.277

    values = [
        lagrange_value(x, y, point),
        newton_divided_forward(x, y, point),
        newton_divided_backward(x, y, point),
        newton_finite_forward(x, y, point),
        newton_finite_backward(x, y, point),
    ]

    assert max(values) - min(values) < 1e-9


def test_quadratic_exact():
    x = [-2, -1, 0, 1, 2]
    y = [(v + 1) ** 2 for v in x]
    point = 0.37
    exact = (point + 1) ** 2

    assert lagrange_value(x, y, point) == pytest.approx(exact)
    assert newton_divided_forward(x, y, point) == pytest.approx(exact)


def test_unequal_grid_rejects_finite_newton():
    x = [0.15, 0.20, 0.33, 0.47]
    y = [1.25, 2.38, 3.79, 5.44]

    assert is_equally_spaced(x)[0] is False

    with pytest.raises(ValueError):
        newton_finite_forward(x, y, 0.22)


def test_duplicate_x_rejected():
    x = [0, 1, 1, 2]
    y = [1, 2, 3, 4]

    with pytest.raises(ValueError):
        lagrange_value(x, y, 0.5)


def test_sin_interpolation_reasonable():
    x = [0.0, 0.25, 0.5, 0.75, 1.0]
    y = [math.sin(v) for v in x]
    point = 0.33

    value = newton_finite_forward(x, y, point)
    assert value == pytest.approx(math.sin(point), abs=1e-4)
