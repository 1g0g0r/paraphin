import numpy as np

# Параметры сетки
Nx, Ny = 5, 6
X_min, X_max = 0., 1.
Y_min, Y_max = 0., 1.
N_elements = (Nx - 1) * (Ny - 1)
hx = (X_max - X_min) / Nx
hy = (Y_max - Y_min) / Ny
area = hx * hy

# Параметры решения
T_end = 1.
dt = 0.001

# Физические параметры задачи
S_min = 0.18
S_max = 0.7
mu_o = 0.006  # вязкость нефти при 20 гр [Pa*c]
mu_w = 0.001  # вязкость воды при 20 гр [Pa*c]

qw = 50 / 1000  # дебит воды [кг/сут]
qo = -1000 / 860  # дебит нефти [кг/сут]
p = np.zeros(N_elements)  # давление
S = np.zeros(N_elements)  # Водонасыщенность
Wo = np.ones(N_elements)  # Массовая доля маслянного компонента в нефти
Wo_0 = np.ones(N_elements)
Wp = np.zeros(N_elements)  # Массовая доля растворенного парафина в нефти
Wps = np.zeros(N_elements)  # Массовая доля взвешенного парафина в нефти
k = np.ones(N_elements) * 3 * 10**-14  # проницаемость [m^2]
m = np.ones(N_elements) * 0.3  # пористость
m_0 = np.ones(N_elements) * 0.3
volume = np.ones(N_elements) * area  # объем элемента h=1


# Граничные условия пласта
p_bound = 1e7
