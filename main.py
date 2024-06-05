import numpy as np

from constants import dT
from solver import Solver


def solve():
    sol = Solver()
    sol.initialize()
    # Задать начальные условия

    time = np.linspace(0, 1, 1/dT)
    for t in time:
        # Решение уравнения давления
        sol.update_p()

        # Решение уравнения насыщенности
        print("все будет хорошо")


if __name__ == '__main__':
    solve()
