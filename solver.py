import taichi as ti

from constants import *
from calc_pressure import calc_pressure
from calc_saturation import calc_saturation

# Инициализация Taichi
ti.init(arch=ti.cpu)
# TODO сравнить с gpu


@ti.data_oriented
class Solver:
    def __init__(self, nx=Nx, ny=Ny, dt=dT):
        # константы
        self.nx = nx
        self.ny = ny
        self.dt = dt

        # поля данных
        self.p = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # давление
        self.S = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # Водонасыщенность
        self.Wo = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # Массовая доля маслянного компонента в нефти
        self.Wo_0 = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))
        self.Wp = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # Массовая доля растворенного парафина в нефти
        self.Wps = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # Массовая доля взвешенного парафина в нефти
        self.k = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # проницаемость [m^2]
        self.m = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # пористость
        self.m_0 = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))
        self.R = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # концентрация взвешенного парафина

    @ti.kernel
    def initialize(self):
        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.p[i, j] = init_p
                self.S[i, j] = init_S
                self.Wo[i, j] = init_Wo
                self.Wo_0[i, j] = init_Wo
                self.Wp[i, j] = init_Wp
                self.Wps[i, j] = init_Wps
                self.k[i, j] = init_k
                self.m[i, j] = init_m
                self.m_0[i, j] = init_m
                self.R[i, j] = init_R

    def update_p(self):
        self.p = calc_pressure(self.nx, self.ny, self.Wo, self.Wo_0,
                               self.m, self.m_0, self.k, self.S, self.p)

    def update_s(self):
        self.S = calc_saturation
