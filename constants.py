import taichi as ti

# Инициализация Taichi
ti.init(arch=ti.cpu)
# TODO сравнить с gpu


# Параметры сетки
Nx, Ny = 3, 3  # Число узлов сетки по x и y
X_min, X_max = 0., 1.
Y_min, Y_max = 0., 1.
hx = (X_max - X_min) / Nx
hy = (Y_max - Y_min) / Ny
area = hx * hy
volume = area * 1

# Параметры решения
T_end = 1.
dT = 0.001

# Физические параметры задачи
S_min = 0.18
S_max = 0.7
mu_o = 0.006  # вязкость нефти при 20 гр [Pa*c]
mu_w = 0.001  # вязкость воды при 20 гр [Pa*c]
ro_w = 1000  # плотность воды
ro_o = 860  # плотность нефти
ro_p = 900  # плотность парафина

qw = 50 / ro_w  # дебит воды [кг/сут]
qo = -100 / ro_o  # дебит нефти [кг/сут]

# данные инициализации
init_p = 1e7
init_S = 0.0
init_Wo = 1.0
init_Wp = 0.0
init_Wps = 0.0
init_k = 3 * 10 ** -14
init_m = 0.3
init_R = 0.0
