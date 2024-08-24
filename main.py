"""Модуль запуска всего расчета."""

from sys import stdout
from numpy import linspace
from tqdm import tqdm

from paraphin.solver import Solver
from paraphin.utils.constants import Time_end, dT


def solve():
    """Запуск расчета."""
    sol = Solver()
    sol.initialize()    # Задать начальные условия

    times = linspace(0, 1, int(Time_end/dT))
    for t in tqdm(iterable=times, ncols=100, desc='Парафин считается', file=stdout):
        sol.upd_time_step()

    breakpoint()


if __name__ == '__main__':
    solve()

# Идея использовать либу taichi пришла благодаря репозиторию: https://github.com/hejob/taichi-fvm2d-fluid-ns

