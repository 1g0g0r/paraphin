import pickle
import numpy as np

from paraphin.utils.constants import *
from paraphin.equations.calc_pressure import calc_pressure
from paraphin.equations.calc_saturation import calc_saturation
from paraphin.equations.calc_wps import calc_wps
from paraphin.equations.calc_temperature import calc_temperature
from paraphin.fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p


@ti.data_oriented
class Solver:
    def __init__(self, nx=Nx, ny=Ny, dt=dT):
        # константы
        self.nx = nx
        self.ny = ny
        self.dt = dt

        # свойства флюидов
        self.mu_o = ti.field(dtype=ti.f32, shape=(nx, ny))  # вязкость нефти
        self.mu_w = ti.field(dtype=ti.f32, shape=(nx, ny))  # вязкость воды
        self.C_w = ti.field(dtype=ti.f32, shape=(nx, ny))  # теплоемкость воды
        self.C_o = ti.field(dtype=ti.f32, shape=(nx, ny))  # теплоемкость нефти
        self.C_f = ti.field(dtype=ti.f32, shape=(nx, ny))  # теплоемкость пласта
        self.C_p = ti.field(dtype=ti.f32, shape=(nx, ny))  # теплоемкость парафина

        # поля данных
        self.p = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # давление
        self.S = ti.field(dtype=ti.f32, shape=(nx, ny))  # Водонасыщенность
        self.S_0 = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.Wo = ti.field(dtype=ti.f32, shape=(nx, ny))  # Массовая доля маслянного компонента в нефти
        self.Wo_0 = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.Wp = ti.field(dtype=ti.f32, shape=(nx, ny))  # Массовая доля растворенного парафина в нефти
        self.Wp_0 = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.Wps = ti.field(dtype=ti.f32, shape=(nx, ny))  # Массовая доля взвешенного парафина в нефти
        self.k = ti.field(dtype=ti.f32, shape=(nx, ny))  # проницаемость [m^2]
        self.m = ti.field(dtype=ti.f32, shape=(nx, ny))  # пористость
        self.m_0 = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.T = ti.field(dtype=ti.f32, shape=(nx, ny))  # температура [C]

        # динамика образования парафина
        self.qp: float = 0.0  # скорость отложения парафиновых отложений в общем объеме пористой породы

        # Массив результатов
        self.pressure = np.empty(np.ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)
        self.saturation = np.empty(np.ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)
        self.temperature = np.empty(np.ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)

    @ti.kernel
    def initialize(self):
        for i, j in ti.ndrange(self.nx + 2, self.ny + 2):
            self.p[i, j] = init_p

        for i, j in ti.ndrange(self.nx, self.ny):
            # параметры пласта
            self.S[i, j] = init_S
            self.S_0[i, j] = init_S
            self.Wo[i, j] = init_Wo
            self.Wo_0[i, j] = init_Wo
            self.Wp[i, j] = init_Wp
            self.Wp_0[i, j] = init_Wp
            self.Wps[i, j] = init_Wps
            self.k[i, j] = init_k
            self.m[i, j] = init_m
            self.m_0[i, j] = init_m
            self.T[i, j] = init_T

            # свойства флюидов
            self.mu_o[i, j] = calc_mu_o(init_T)
            self.mu_w[i, j] = calc_mu_w(init_T)
            self.C_w[i, j] = calc_c_w(init_T)
            self.C_o[i, j] = calc_c_o(init_T)
            self.C_f[i, j] = calc_c_f(init_T)
            self.C_p[i, j] = calc_c_p(init_T)

    @ti.kernel
    def update_mu_and_c(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            self.mu_o[i, j] = calc_mu_o(self.T[i, j])
            self.mu_w[i, j] = calc_mu_w(self.T[i, j])
            self.C_w[i, j] = calc_c_w(self.T[i, j])
            self.C_o[i, j] = calc_c_o(self.T[i, j])
            self.C_f[i, j] = calc_c_f(self.T[i, j])
            self.C_p[i, j] = calc_c_p(self.T[i, j])

    def update_p(self):
        self.p = calc_pressure(self.Wo, self.Wo_0, self.m, self.m_0, self.k,
                               self.S, self.p, self.mu_o, self.mu_w)

    def update_s(self):
        self.S, self.S_0 = calc_saturation(self.S, self.S_0, self.p, self.k, self.m,
                                           self.m_0, self.mu_o, self.mu_w)

    def update_wps(self):
        self.Wps = calc_wps(self.qp, self.m, self.m_0, self.S, self.S_0, self.Wp, self.Wp_0,
                            self.Wps, self.p, self.k, self.mu_o, self.mu_w)

    def update_t(self):
        self.T = calc_temperature(self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p,
                                  self.Wp, self.Wps, self.p, self.k, self.mu_o, self.mu_w)

    def upd_time_step(self):
        self.update_p()  # Обновление давления
        self.update_s()  # Обновление насыщенности
        self.update_wps()  # Обновлнние концентрации взвешенного парафина
        self.update_t()    # Обновление температуры

    def save_results(self, idx):
        self.pressure[idx] = self.p.to_numpy()[1:-1, 1:-1]
        self.saturation[idx] = self.S.to_numpy()
        self.temperature[idx] = self.T.to_numpy()

        if np.isclose(idx, len(self.pressure) - 1):
            with open('data.pkl', 'wb') as f:
                pickle.dump([self.pressure, self.saturation, self.temperature], f)

    def start_visualize(self):
        with open('data.pkl', 'rb') as f:
            pres, sat, temp = pickle.load(f)

        # Визуализируем решение
        gui = ti.GUI("Поля данных", res=(self.nx, self.ny))
        time_slider = gui.slider("Время", 0, Time_end)

        # Максимальные/минимальные значения полей данных
        min_pres = min([np.min(q) for q in pres])
        max_pres = max([np.max(q) for q in pres])
        min_sat = min([np.min(q) for q in sat])
        max_sat = max([np.max(q) for q in sat])
        min_temp = min([np.min(q) for q in temp])
        max_temp = max([np.max(q) for q in temp])

        # Начальная инициализация GUI
        data = pres[0]
        min_val = min_pres
        max_val = max_pres

        while gui.running:
            # Получаем индекс времени
            time_index = int(time_slider.value / sol_time_step)

            # Выбор поля данных
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key == ti.GUI.SPACE:
                    gui.running = False
                elif e.key == '1':
                    data = pres[time_index]
                    min_val = min_pres
                    max_val = max_pres
                elif e.key == '2':
                    data = sat[time_index]
                    min_val = min_sat
                    max_val = max_sat
                elif e.key == '3':
                    data = temp[time_index]
                    min_val = min_temp
                    max_val = max_temp

            # Нормализуем данные для преобразования в цвет
            normalized_data = (data - min_val) / (max_val - min_val)  # Нормализация
            color_data = np.zeros((self.nx, self.ny, 3))  # Создаем массив для цвета

            # Преобразуем нормализованные данные в цвет (RGB) # Пример: градиент от красного к синему
            color_data[:, :, 0] = normalized_data
            color_data[:, :, 1] = np.ones_like(normalized_data) * 0.5
            color_data[:, :, 2] = 1.0 - normalized_data

            # Устанавливаем цветное изображение
            gui.set_image(color_data)

            # Отображаем GUI
            gui.show()
