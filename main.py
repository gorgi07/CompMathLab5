from __future__ import annotations


from data_sources import generate_from_function, load_from_file, manual_input, read_float
from interpolation import (
    divided_difference_table,
    finite_difference_table,
    format_divided_table,
    format_finite_table,
    is_equally_spaced,
    lagrange_value,
    newton_divided_backward,
    newton_divided_forward,
    newton_finite_backward,
    newton_finite_forward,
    recommended_finite_newton_formula,
)
from plotting import plot_interpolation


def choose_source():
    print(
        """
Способ задания исходных данных:
  1. Ввести таблицу x, y с клавиатуры
  2. Прочитать таблицу из файла
  3. Сформировать таблицу по выбранной функции
"""
    )
    choice = input("Ваш выбор: ").strip()

    if choice == "1":
        return manual_input()
    if choice == "2":
        path = input("Путь к файлу: ").strip()
        return load_from_file(path)
    if choice == "3":
        return generate_from_function()

    raise ValueError("Неизвестный способ ввода.")


def print_method_results(x_nodes, y_nodes, target_x):
    print("\n" + "=" * 76)
    print(f"Интерполяция в точке x = {target_x}")
    print("=" * 76)

    results = {}

    # 1. Лагранж
    lagrange = lagrange_value(x_nodes, y_nodes, target_x)
    results["Лагранж"] = lagrange
    print(f"Многочлен Лагранжа:                         {lagrange:.12g}")

    # 2. Ньютон с разделенными разностями — обе формы
    ndf = newton_divided_forward(x_nodes, y_nodes, target_x)
    ndb = newton_divided_backward(x_nodes, y_nodes, target_x)
    results["Ньютон (разд. разности, вперед)"] = ndf
    results["Ньютон (разд. разности, назад)"] = ndb

    print(f"Ньютон, разделенные разности, 1-я форма:    {ndf:.12g}")
    print(f"Ньютон, разделенные разности, 2-я форма:    {ndb:.12g}")

    # 3. Ньютон с конечными разностями — только равноотстоящая сетка
    equal, h = is_equally_spaced(x_nodes)
    if equal:
        nff = newton_finite_forward(x_nodes, y_nodes, target_x)
        nfb = newton_finite_backward(x_nodes, y_nodes, target_x)
        results["Ньютон (кон. разности, вперед)"] = nff
        results["Ньютон (кон. разности, назад)"] = nfb

        recommended = recommended_finite_newton_formula(x_nodes, target_x)
        recommended_name = "первая (вперед)" if recommended == "forward" else "вторая (назад)"

        print(f"Ньютон, конечные разности, 1-я формула:     {nff:.12g}")
        print(f"Ньютон, конечные разности, 2-я формула:     {nfb:.12g}")
        print(f"Рекомендуемая по положению x формула:       {recommended_name}")
    else:
        print(
            "Ньютон с конечными разностями: НЕ применяется, "
            "так как узлы не равноотстоящие."
        )

    values = list(results.values())
    spread = max(values) - min(values)
    print(f"\nМаксимальное расхождение результатов: {spread:.6e}")

    return results


def main():
    print("Лабораторная работа: интерполяция функций")
    print("Методы: Лагранж, Ньютон с разделенными и конечными разностями.")

    try:
        x_nodes, y_nodes, source_function = choose_source()

        print("\nИсходные данные:")
        for i, (x, y) in enumerate(zip(x_nodes, y_nodes)):
            print(f"{i:>3}: x = {x:>14.8g}, y = {y:>14.8g}")

        # Таблица разделенных разностей
        print("\nТаблица разделенных разностей:")
        div_table = divided_difference_table(x_nodes, y_nodes)
        print(format_divided_table(x_nodes, div_table))

        # Таблица конечных разностей — имеет смысл для равноотстоящих узлов
        equal, h = is_equally_spaced(x_nodes)
        if equal:
            print(f"\nУзлы равноотстоящие, h = {h:.10g}")
            print("Таблица конечных разностей:")
            fin_table = finite_difference_table(y_nodes)
            print(format_finite_table(x_nodes, fin_table))
        else:
            print(
                "\nУзлы не являются равноотстоящими. "
                "Таблица конечных разностей не используется для формул "
                "Ньютона с равномерной сеткой."
            )

        target_x = read_float("\nВведите аргумент x, для которого требуется интерполяция: ")
        results = print_method_results(x_nodes, y_nodes, target_x)

        if source_function is not None:
            name, func, _, _ = source_function
            exact = func(target_x)
            print(f"\nТочное значение {name}: {exact:.12g}")
            print("Абсолютные погрешности:")
            for method, value in results.items():
                print(f"  {method:<38} {abs(value - exact):.6e}")

        answer = input("\nПостроить график? [y/n]: ").strip().lower()
        if answer in {"y", "yes", "д", "да"}:
            plot_interpolation(x_nodes, y_nodes, target_x, source_function)

    except (ValueError, FileNotFoundError) as exc:
        print(f"\nОшибка входных данных: {exc}")
    except KeyboardInterrupt:
        print("\nРабота программы прервана пользователем.")


if __name__ == "__main__":
    main()
    print("З.Ы. Приходите ещё :)")
