from pickle import dump

from numpy import empty, ceil, isclose
from taichi import f32, field, ndrange, data_oriented, kernel, types

from paraphin.equations import (calc_qp, calc_pressure, calc_saturation, calc_temperature, calc_wps_wp)
from paraphin.fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p
from paraphin.utils.constants import (Nx, Ny, Nr, Time_end, sol_time_step, output_file_name, init_T, r, fi_0,
                                      init_k, init_S, init_m, init_Wp, init_Wps, init_Wo, init_p, init_qp)


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
        self.p = field(dtype=f32, shape=(nx, ny))  # давление
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

        # динамика образования парафина (кольматация\суффозия)
        self.integr_r2_fi0 = field(dtype=f32, shape=())
        self.integr_r4_fi0 = field(dtype=f32, shape=())
        self.r = field(dtype=f32, shape=Nr)
        self.fi = field(dtype=f32, shape=(nx, ny, fi_0.shape[0]))
        self.qp = field(dtype=f32, shape=(nx, ny))  # скорость отложения парафиновых отложений в общем объеме пористой породы

        # Массив результатов
        self.pres_arr = empty(ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)
        self.sat_arr = empty(ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)
        self.temp_arr = empty(ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)

    def initialize(self):
        @kernel
        def calc_integrals(rr: types.ndarray(), fi_o: types.ndarray()):
            self.integr_r2_fi0[None] = 0.0
            self.integr_r4_fi0[None] = 0.0

            self.r[0] = rr[0]
            r3_old = rr[0] ** 3
            r4_old = rr[0] * r3_old
            r5_old = rr[0] * r4_old
            r6_old = rr[0] * r5_old
            for i in ndrange((1, rr.shape[0])):
                self.r[i] = rr[i]
                dr = rr[i] - rr[i - 1]
                r3_new = rr[i] ** 3
                r4_new = rr[i] * r3_new
                r5_new = rr[i] * r4_new
                r6_new = rr[i] * r5_new
                a = (fi_o[i - 1] * rr[i] - fi_o[i] * rr[i - 1]) / dr
                b = (fi_o[i] - fi_o[i - 1]) / dr
                self.integr_r2_fi0[None] += (r3_new - r3_old) * a / 3 + (r4_new - r4_old) * b / 4  # r^2 * fi
                self.integr_r4_fi0[None] += (r5_new - r5_old) * a / 5 + (r6_new - r6_old) * b / 6  # r^4 * fi
                r3_old = r3_new
                r4_old = r4_new
                r5_old = r5_new
                r6_old = r6_new

        @kernel
        def initialize_params_loop(fi_o: types.ndarray()):
            for i, j in ndrange(self.nx, self.ny):
                # параметры пласта
                self.p[i, j] = init_p
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
                self.qp[i, j] = init_qp

                # свойства флюидов
                self.mu_o[i, j] = calc_mu_o(init_T)
                self.mu_w[i, j] = calc_mu_w(init_T)
                self.C_w[i, j] = calc_c_w(init_T)
                self.C_o[i, j] = calc_c_o(init_T)
                self.C_f[i, j] = calc_c_f(init_T)
                self.C_p[i, j] = calc_c_p(init_T)

                for ij in ndrange(fi_o.shape[0]):
                    self.fi[i, j, ij] = fi_o[ij]

        calc_integrals(rr=r, fi_o=fi_0)
        initialize_params_loop(fi_o=fi_0)

    @kernel
    def _update_mu_and_c_temp(self):
        for i, j in ndrange(self.nx, self.ny):
            self.mu_o[i, j] = calc_mu_o(self.T[i, j])
            self.mu_w[i, j] = calc_mu_w(self.T[i, j])
            self.C_w[i, j] = calc_c_w(self.T[i, j])
            self.C_o[i, j] = calc_c_o(self.T[i, j])
            self.C_f[i, j] = calc_c_f(self.T[i, j])
            self.C_p[i, j] = calc_c_p(self.T[i, j])

    def _update_p(self) -> None:
        self.p = calc_pressure(self.Wo, self.Wo_0, self.m, self.m_0,
                               self.k, self.S, self.mu_o, self.mu_w)

    def _update_s(self) -> None:
        self.S, self.S_0 = calc_saturation(self.S, self.S_0, self.p, self.k, self.m,
                                           self.m_0, self.mu_o, self.mu_w)

    def _update_wps_wp(self) -> None:
        self.Wps, self.Wp = calc_wps_wp(self.qp, self.m, self.m_0, self.S, self.S_0, self.Wp, self.Wp_0,
                                        self.Wps, self.p, self.k, self.mu_o, self.mu_w, self.T)

    def _update_t(self) -> None:
        self.T = calc_temperature(self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p,
                                  self.Wp, self.Wps, self.p, self.k, self.mu_o, self.mu_w)

    def _update_qp_m_k(self):
        self.qp, self.m, self.k = calc_qp(self.p, self.Wps, self.m, self.fi, self.r,
                                          self.integr_r2_fi0, self.integr_r4_fi0)

    def upd_time_step(self) -> None:
        """Метод IMPES: явный по насыщенности и неявный по давления"""
        self._update_p()  # Обновление давления
        self._update_s()  # Обновление насыщенности
        self._update_wps_wp()  # Обновлнние концентрации взвешенного парафина
        self._update_t()    # Обновление температуры
        self._update_qp_m_k()  # Обновление объема выделяемого парафина, пористости, проницаемости

        # self._update_mu_and_c_temp()  # Обновление свойств веществ ввиду изменения температуры

    def save_results(self, idx) -> None:
        self.pres_arr[idx] = self.p.to_numpy()
        self.sat_arr[idx] = self.S.to_numpy()
        self.temp_arr[idx] = self.T.to_numpy()

        if isclose(idx, len(self.pres_arr) - 1):
            with open(output_file_name, 'wb') as f:
                dump([self.pres_arr, self.sat_arr, self.temp_arr], f)
