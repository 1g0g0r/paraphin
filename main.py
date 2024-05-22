from my_consatnts import *
from my_mesh import create_mesh
from calc_p import calc_p


def solve():
    # Создали сетку
    points, cells = create_mesh()

    # Задать начальные условия

    # Решение уравнения давления
    calc_p(cells)


