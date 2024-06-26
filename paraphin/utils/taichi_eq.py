import taichi as ti
import numpy as np
from time import time
import pickle

ti.init(arch=ti.cpu)

# Размеры сетки
n = 512
t_max = 7000
corr_t_max = int((t_max + 1) / 100)

# Создаем поля для хранения значений
u = ti.field(ti.f32, shape=(n, n))
b = ti.field(ti.f32, shape=(n, n))
times = np.empty(corr_t_max + 1, dtype=object)


@ti.kernel
def set_boundary():
    """Задаем граничные условия"""
    u[int(n/2), int(n/2)] = 10.0
    # for i, j in b:
    #     if i == 0:
    #         u[0, j] = u[1, j] + 10.0
    #     elif i == n - 1:
    #         u[n - 1, j] = 10
    #     elif j == 0 or j == n - 1:
    #         u[i, j] = 0.0


@ti.kernel
def poisson():
    """Решаем уравнение Пуассона"""
    for i, j in u:
        if 0 < i < n-1 and 0 < j < n - 1:
            u[i, j] = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] + b[i, j]) * 0.25


def start_calc():
    t1 = time()

    for i in range(t_max):
        set_boundary()
        poisson()
        if i % 100 == 0 or i == t_max - 1:
            times[int(i/100)] = u.to_numpy()
        if i == t_max - 1:
            times[int(t_max/100)] = u.to_numpy()
    print(time() - t1)

    with open('data.pkl', 'wb') as f:
        pickle.dump(times, f)


def start_visualize():
    with open('data.pkl', 'rb') as f:
        new_times = pickle.load(f)
    # Визуализируем решение
    gui = ti.GUI("Poisson Equation", res=(n, n))
    time_slider = gui.slider("Time", 0, t_max)

    while gui.running:
        gui.set_image(new_times[int(time_slider.value / 100)])
        gui.show()


if __name__ == '__main__':
    start_calc()
    start_visualize()
