from pickle import dump

from numpy import empty, ceil, isclose
from taichi import f32, field, ndrange, data_oriented, kernel

from paraphin.equations.calc_pressure import calc_pressure
from paraphin.equations.calc_saturation import calc_saturation
from paraphin.equations.calc_temperature import calc_temperature
from paraphin.equations.calc_wps import calc_wps
from paraphin.fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p
from paraphin.utils.constants import (Nx, Ny, Time_end, sol_time_step, output_file_name, init_T,
                                      init_k, init_S, init_m, init_Wp, init_Wps, init_Wo, init_p)


@data_oriented
class Solver:
    def __init__(self, nx=Nx, ny=Ny):
        # константы
        self.nx = nx
        self.ny = ny

        # свойства флюидов
        self.mu_o = field(dtype=f32, shape=(nx, ny))  # вязкость нефти
        self.mu_w = field(dtype=f32, shape=(nx, ny))  # вязкость воды
        self.C_w = field(dtype=f32, shape=(nx, ny))  # теплоемкость воды
        self.C_o = field(dtype=f32, shape=(nx, ny))  # теплоемкость нефти
        self.C_f = field(dtype=f32, shape=(nx, ny))  # теплоемкость пласта
        self.C_p = field(dtype=f32, shape=(nx, ny))  # теплоемкость парафина

        # поля данных
        self.p = field(dtype=f32, shape=(nx + 2, ny + 2))  # давление
        self.S = field(dtype=f32, shape=(nx, ny))  # Водонасыщенность
        self.S_0 = field(dtype=f32, shape=(nx, ny))
        self.Wo = field(dtype=f32, shape=(nx, ny))  # Массовая доля маслянного компонента в нефти
        self.Wo_0 = field(dtype=f32, shape=(nx, ny))
        self.Wp = field(dtype=f32, shape=(nx, ny))  # Массовая доля растворенного парафина в нефти
        self.Wp_0 = field(dtype=f32, shape=(nx, ny))
        self.Wps = field(dtype=f32, shape=(nx, ny))  # Массовая доля взвешенного парафина в нефти
        self.k = field(dtype=f32, shape=(nx, ny))  # проницаемость [m^2]
        self.m = field(dtype=f32, shape=(nx, ny))  # пористость
        self.m_0 = field(dtype=f32, shape=(nx, ny))
        self.T = field(dtype=f32, shape=(nx, ny))  # температура [C]

        # динамика образования парафина
        self.qp: float = 0.0  # скорость отложения парафиновых отложений в общем объеме пористой породы

        # Массив результатов
        self.pressure = empty(ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)
        self.saturation = empty(ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)
        self.temperature = empty(ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)

    @kernel
    def initialize(self):
        for i, j in ndrange(self.nx + 2, self.ny + 2):
            self.p[i, j] = init_p

        for i, j in ndrange(self.nx, self.ny):
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

    @kernel
    def _update_mu_and_c(self):
        for i, j in ndrange(self.nx, self.ny):
            self.mu_o[i, j] = calc_mu_o(self.T[i, j])
            self.mu_w[i, j] = calc_mu_w(self.T[i, j])
            self.C_w[i, j] = calc_c_w(self.T[i, j])
            self.C_o[i, j] = calc_c_o(self.T[i, j])
            self.C_f[i, j] = calc_c_f(self.T[i, j])
            self.C_p[i, j] = calc_c_p(self.T[i, j])

    def _update_p(self) -> None:
        self.p = calc_pressure(self.Wo, self.Wo_0, self.m, self.m_0, self.k,
                               self.S, self.p, self.mu_o, self.mu_w)

    def _update_s(self) -> None:
        self.S, self.S_0 = calc_saturation(self.S, self.S_0, self.p, self.k, self.m,
                                           self.m_0, self.mu_o, self.mu_w)

    def _update_wps(self) -> None:
        self.Wps = calc_wps(self.qp, self.m, self.m_0, self.S, self.S_0, self.Wp, self.Wp_0,
                            self.Wps, self.p, self.k, self.mu_o, self.mu_w)

    def _update_t(self) -> None:
        self.T = calc_temperature(self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p,
                                  self.Wp, self.Wps, self.p, self.k, self.mu_o, self.mu_w)

    def upd_time_step(self) -> None:
        """Метод IMPES: явный по насыщенности и неявный по давления"""
        self._update_p()  # Обновление давления
        self._update_s()  # Обновление насыщенности
        self._update_wps()  # Обновлнние концентрации взвешенного парафина
        self._update_t()    # Обновление температуры

        self._update_mu_and_c()

    def save_results(self, idx) -> None:
        self.pressure[idx] = self.p.to_numpy()[1:-1, 1:-1]
        self.saturation[idx] = self.S.to_numpy()
        self.temperature[idx] = self.T.to_numpy()

        if isclose(idx, len(self.pressure) - 1):
            with open(output_file_name, 'wb') as f:
                dump([self.pressure, self.saturation, self.temperature], f)


