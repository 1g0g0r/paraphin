from pickle import dump

from numpy import empty, ceil, isclose
from taichi import f32, field, ndrange, data_oriented, kernel, types

from paraphin.equations import (calc_qp, calc_pressure, calc_saturation, calc_temperature, calc_wps_wp)
from paraphin.utils.fluids_correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w, calc_c_p
from paraphin.constants import (Nx, Ny, Nr, Time_end, sol_time_step, output_file_name, init_T, r, fi_0,
                                      init_k, init_S, init_m, init_Wp, init_Wps, init_Wo, init_p, init_qp, init_h_sloy)


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
        self.fi = field(dtype=f32, shape=(nx, ny, Nr))
        self.h_sloy = field(dtype=f32, shape=(nx, ny, Nr))
        self.qp = field(dtype=f32, shape=(nx, ny))  # скорость отложения парафина в общем объеме

        # Массив результатов
        self.pres_arr = empty(ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)
        self.sat_arr = empty(ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)
        self.temp_arr = empty(ceil(Time_end / sol_time_step + 1).astype(int), dtype=object)


    def initialize(self):
        @kernel
        def calc_integrals(rr: types.ndarray(), fi_o: types.ndarray()):
            """Вычисление интегралов от функций r^4*fi_o(r) и r^2*fi_o(r)"""
            self.integr_r2_fi0[None] = 0.0
            self.integr_r4_fi0[None] = 0.0
            self.r.from_numpy(rr)

            r3 = rr ** 3
            r4 = r3 * rr
            r5 = r4 * rr
            r6 = r5 * rr
            for i in ndrange((1, rr.shape[0])):
                dr = rr[i] - rr[i - 1]
                a = (fi_o[i - 1] * rr[i] - fi_o[i] * rr[i - 1]) / dr
                b = (fi_o[i] - fi_o[i - 1]) / dr
                self.integr_r2_fi0[None] += (r3[i] - r3[i-1]) * a / 3 + (r4[i] - r4[i-1]) * b / 4  # r^2 * fi
                self.integr_r4_fi0[None] += (r5[i] - r5[i-1]) * a / 5 + (r6[i] - r6[i-1]) * b / 6  # r^4 * fi


        @kernel
        def initialize_params_loop(fi_o: types.ndarray()):
            for i in ndrange(self.nx):
                for j in ndrange(self.ny):
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
                        self.h_sloy[i, j, ij] = init_h_sloy

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
        """Обновление давления."""
        self.p = calc_pressure(self.p, self.Wo, self.Wo_0, self.m, self.m_0,
                               self.k, self.S, self.mu_o, self.mu_w)


    def _update_s(self) -> field(dtype=f32, shape=(Nx, Ny)):
        """Обновление насыщенности."""
        return calc_saturation(self.S, self.p, self.k, self.m, self.m_0, self.mu_o, self.mu_w)


    def _update_wps_wp(self) -> (field(dtype=f32, shape=(Nx, Ny)), field(dtype=f32, shape=(Nx, Ny))):
        """Обновлнние концентрации взвешенного и растворенного парафина."""
        new_Wps, new_Wp = calc_wps_wp(self.qp, self.m, self.m_0, self.S, self.S_0, self.Wp, self.Wp_0,
                                      self.Wps, self.p, self.k, self.mu_o, self.mu_w, self.T)
        return new_Wps, new_Wp


    def _update_t(self) -> field(dtype=f32, shape=(Nx, Ny)):
        """Обновление температуры."""
        return calc_temperature(self.T, self.m, self.S, self.C_o, self.C_w, self.C_f, self.C_p,
                                self.Wp, self.Wps, self.p, self.k, self.mu_o, self.mu_w)


    def _update_qp_m_k(self) -> (field(dtype=f32, shape=(Nx, Ny)), field(dtype=f32, shape=(Nx, Ny)),
                                 field(dtype=f32, shape=(Nx, Ny))):
        """Обновление объема выделяемого парафина, пористости и проницаемости."""
        new_qp, m_mult, k_mult = calc_qp(self.p, self.Wps, self.mu_o, self.m, self.qp, self.fi, self.h_sloy,
                                       self.r, self.integr_r2_fi0, self.integr_r4_fi0)
        return new_qp, m_mult, k_mult


    def upd_time_step(self) -> None:
        """Метод IMPES: явный по насыщенности неявный по давлению"""
        self._update_p()

        with ProcessPoolExecutor() as executor:
            # Запускаем задачи параллельно
            sat    = executor.submit(self._update_s)       # Обновление насыщенности
            wps_wp = executor.submit(self._update_wps_wp)  # Обновление концентрации взвешенного парафина
            temp   = executor.submit(self._update_t)       # Обновление температуры
            qp_m_k = executor.submit(self._update_qp_m_k)  # Обновление объема выделяемого парафина, пористости, проницаемости
    
            # Получаем результаты вычислений
            new_s = sat.result()
            new_wps, new_wp = wps_wp.result()
            new_t = temp.result()
            new_qp, m_mult, k_mult = qp_m_k.result()

        # self._update_mu_and_c_temp()  # Обновление свойств веществ ввиду изменения температуры
        self.swap_time_steps(new_s, new_wps, new_wp, new_t, new_qp, m_mult, k_mult)


    def swap_time_steps(self, new_s, new_wps, new_wp, new_t, new_qp, m_mult, k_mult):
        """Обновление полей данных на новом временном слое."""
        self.S_0, self.S = self.S, new_s
        self.Wo_0 = self.Wo
        self.Wo.from_numpy(1.0 - self.Wp.to_array() - self.Wps.to_array())
        self.Wp_0, self.Wp = self.Wp, new_wp
        self.Wps = new_wps
        self.k.from_numpy(self.k.to_array() * k_mult)
        self.m_0 = self.m
        self.m.from_numpy(self.m.to_array() * m_mult)
        self.T = new_t
        self.qp = new_qp


    def save_results(self, idx) -> None:
        """Сохранение полей данных в файл формата pkl."""
        self.pres_arr[idx] = self.p.to_numpy()
        self.sat_arr[idx] = self.S.to_numpy()
        self.temp_arr[idx] = self.T.to_numpy()

        if isclose(idx, len(self.pres_arr) - 1):
            with open(output_file_name, 'wb') as f:
                dump([self.pres_arr, self.sat_arr, self.temp_arr], f)
