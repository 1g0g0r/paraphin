from paraphin.utils.constants import *
from paraphin.equations.calc_pressure import calc_pressure
from paraphin.equations.calc_saturation import calc_saturation
from paraphin.equations.calc_wps import calc_wps
from paraphin.equations.calc_temperature import calc_temperature
from paraphin.correlations import calc_mu_o, calc_mu_w, calc_c_f, calc_c_o, calc_c_w


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
        self.C_o = ti.field(dtype=ti.f32, shape=(nx, ny))  # теплоемкость воды
        self.C_f = ti.field(dtype=ti.f32, shape=(nx, ny))  # теплоемкость воды

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
        self.T = ti.field(dtype=ti.f32, shape=(nx, ny))  # температура

        # динамика образования парафина
        self.qp: float = 0.0  # скорость отложения парафиновых отложений в общем объеме пористой породы

    @ti.kernel
    def initialize(self):
        for i in range(self.nx + 2):
            for j in range(self.ny + 2):
                self.p[i, j] = init_p

        for i in range(self.nx):
            for j in range(self.ny):
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

    @ti.kernel
    def update_mu_and_c(self):
        for i in range(self.nx):
            for j in range(self.ny):
                self.mu_o[i, j] = calc_mu_o(self.T[i, j])
                self.mu_w[i, j] = calc_mu_w(self.T[i, j])
                self.C_w[i, j] = calc_c_w(self.T[i, j])
                self.C_o[i, j] = calc_c_o(self.T[i, j])
                self.C_f[i, j] = calc_c_f(self.T[i, j])

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
        self.T = calc_temperature(self.T, self.m, self.S, self.C_o, self.C_w, self.C_f,
                                  self.C_p, self.Wp, self.Wps, self.p, self.k)
