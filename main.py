"""Модуль запуска всего расчета."""

from sys import stdout
from numpy import linspace, isclose
from tqdm import tqdm

from paraphin.solver import Solver
from paraphin.utils.constants import Time_end, dT, sol_time_step


def solve():
    """Запуск расчета."""
    iter = 0
    sol = Solver()
    sol.initialize()  # Задание начальных условий

    times = linspace(0, Time_end, int(Time_end / dT + 1))
    for t in tqdm(iterable=times, ncols=100, desc='Парафин считается', file=stdout):
        sol.upd_time_step()

        if t >= iter * sol_time_step or isclose(t, Time_end):
            sol.save_results(iter)
            iter += 1

    sol.start_visualize()


if __name__ == '__main__':
    solve()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns
