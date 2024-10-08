from lib2to3.refactor import RefactoringTool

from numpy import array, float32

output_file_name = 'data.pkl'

# Параметры сетки
Nx, Ny = 128, 128  # Число узлов сетки по x и y
X_min, X_max = 0., 1.
Y_min, Y_max = 0., 1.
hx = (X_max - X_min) / Nx
hy = (Y_max - Y_min) / Ny
area: float = hx * hy
volume = area * 1

# Параметры решения
Time_end = 1.
dt = 0.0001
sol_time_step = 0.1

# Физические параметры задачи
S_min = 0.18
S_max = 0.7

Tm = 55  # температура кристализации парафина, [C]
ro_w = 1000  # плотность воды, [кг/м^3]
ro_o = 860  # плотность нефти, [кг/м^3]
ro_p = 900  # плотность парафина, [кг/м^3]
ro_f = 2500  # плотность пласта, [кг/м^3]
qw = 50 / ro_w  # дебит воды [кг/сут]  # вопрос
qo = -100 / ro_o  # дебит нефти [кг/сут]

# TODO проверить!!
K_p = 0.2  # Вт/(м·К)
K_w = 0.6  # Вт/(м·К)
K_o = 0.12  # Вт/(м·К)
K_f = 2  # Вт/(м·К)

# данные моделирования кольматации\суффозии
Diff = 2.0 * 10**-16  # Коэффициент диффузионного осаждения частиц
D = 4.0 * 10**-6  # Размер частицы
betta = 0.01  # Доля блокируемых каналов
gamma = 0.4  # Отношение радиуса горла к радиусу канала
Lk =  3.0 * 10**-5 # Длина капиляра
eta = 1.0  # Коэффициент извилистости

# данные инициализации
init_p = 1e7  # [Па]
init_S = 0.0
init_Wo = 1.0
init_Wp = 0.0
init_Wps = 0.0
init_k = 3 * 10**-12  # [м^2]
init_m = 0.6
init_T = 40  # [C]
r = array([0, 0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000007, 0.000008,
           0.000009, 0.00001, 0.000011, 0.000012, 0.000013, 0.000014, 0.000015, 0.000016, 0.000017,
           0.000018, 0.000019, 0.00002, 0.000021, 0.000022, 0.000023, 0.000024, 0.000025], dtype=float32)

fi_0 = array([0, 0.013, 0.023, 0.031, 0.035, 0.034, 0.027, 0.021, 0.016, 0.018, 0.025, 0.032, 0.041,
              0.052, 0.061, 0.073, 0.082, 0.086, 0.081, 0.07, 0.059, 0.048, 0.035, 0.024, 0.013, 0], dtype=float32)
Nr: int = len(r)
