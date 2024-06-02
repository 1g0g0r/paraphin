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
ro_w = 1000  # плотность воды
ro_o = 860  # плотность нефти
ro_p = 900  # плотность парафина

qw = 50 / ro_w  # дебит воды [кг/сут]
qo = -100 / ro_o  # дебит нефти [кг/сут]
p = np.zeros(N_elements, dtype=np.float64)  # давление
S = np.zeros(N_elements, dtype=np.float64)  # Водонасыщенность
Wo = np.ones(N_elements, dtype=np.float64)  # Массовая доля маслянного компонента в нефти
Wo_0 = np.ones(N_elements, dtype=np.float64)
Wp = np.zeros(N_elements, dtype=np.float64)  # Массовая доля растворенного парафина в нефти
Wps = np.zeros(N_elements, dtype=np.float64)  # Массовая доля взвешенного парафина в нефти
k = np.ones(N_elements, dtype=np.float64) * 3 * 10**-14  # проницаемость [m^2]
#  from decimal import Decimal
m = np.ones(N_elements, dtype=np.float64) * 0.3  # пористость
m_0 = np.ones(N_elements, dtype=np.float64) * 0.3
volume = np.ones(N_elements, dtype=np.float64) * area  # объем элемента h=1
R = np.zeros(N_elements, dtype=np.float64)  # концентрация взвешенного парафина

# Граничные условия пласта
p_bound = 1e7
