import numpy as np

from my_consatnts import *
from my_mesh import create_mesh
from calc_pressure import calc_pressure
from calc_saturation import calc_saturation


def solve():
    # Создали сетку
    points, cells = create_mesh()

    # Задать начальные условия

    time = np.linspace(0, 1, 1/dt)
    for t in time:
        # Решение уравнения давления
        calc_pressure(cells)

        # Решение уравнения насыщенности
        calc_saturation(cells)


if __name__ == '__main__':
    solve()
