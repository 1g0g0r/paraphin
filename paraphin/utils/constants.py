import taichi as ti
import numpy as np

# Инициализация Taichi
ti.init(arch=ti.amdgpu)  # amdgpu


# Параметры сетки
Nx, Ny = 150, 150  # Число узлов сетки по x и y
X_min, X_max = 0., 1.
Y_min, Y_max = 0., 1.
hx = (X_max - X_min) / Nx
hy = (Y_max - Y_min) / Ny
area = 1.  # hx * hy
volume = area * 1

# Параметры решения
Time_end = 1.
dT = 0.01
sol_time_step = 0.1

# Физические параметры задачи
S_min = 0.18
S_max = 0.7

ro_w = 1000  # плотность воды
ro_o = 860  # плотность нефти
ro_p = 900  # плотность парафина
ro_f = 2500  # плотность парафина
qw = 50 / ro_w  # дебит воды [кг/сут]
qo = -100 / ro_o  # дебит нефти [кг/сут]

# TODO проверить!!
K_p = 0.2  # Вт/(м·К)
K_w = 0.6  # Вт/(м·К)
K_o = 0.12  # Вт/(м·К)
K_f = 2  # Вт/(м·К)

# данные инициализации
init_p = 1e7  # [Па]
init_S = 0.0
init_Wo = 1.0
init_Wp = 0.0
init_Wps = 0.0
init_k = 3 * 10 ** -14  # [м^2]
init_m = 0.3
init_R = 0.0
init_T = 40  # [C]
