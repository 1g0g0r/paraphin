"""Модуль запуска всего расчета."""
from numpy import linspace
from tqdm import tqdm

from paraphin.solver import Solver
from paraphin.utils.constants import Time_end, dT


def solve():
    """Запуск расчета."""
    sol = Solver()
    sol.initialize()    # Задать начальные условия

    def run_iteration():
        sol.update_p()  # Обновление давления
        sol.update_s()  # Обновление насыщенности
        sol.update_wps()  # Обновлнние концентрации взвешенного парафина
        sol.update_t()  # Обновление температуры

    times = linspace(0, 1, int(Time_end/dT))
    list(tqdm(map(run_iteration, times), total=len(times), ncols=70,
              desc="Парафин считается", colour="#009FBD"))
    breakpoint()


if __name__ == '__main__':
    solve()
    #     https://github.com/hejob/taichi-fvm2d-fluid-ns
