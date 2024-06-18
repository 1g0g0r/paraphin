from paraphin.utils.constants import *
from paraphin.equations.calc_pressure import calc_pressure
from paraphin.equations.calc_saturation import calc_saturation


@ti.data_oriented
class Solver:
    def __init__(self, nx=Nx, ny=Ny, dt=dT):
        # константы
        self.nx = nx
        self.ny = ny
        self.dt = dt

        # поля данных
        self.p = ti.field(dtype=ti.f32, shape=(nx + 2, ny + 2))  # давление
        self.S = ti.field(dtype=ti.f32, shape=(nx, ny))  # Водонасыщенность
        self.Wo = ti.field(dtype=ti.f32, shape=(nx, ny))  # Массовая доля маслянного компонента в нефти
        self.Wo_0 = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.Wp = ti.field(dtype=ti.f32, shape=(nx, ny))  # Массовая доля растворенного парафина в нефти
        self.Wps = ti.field(dtype=ti.f32, shape=(nx, ny))  # Массовая доля взвешенного парафина в нефти
        self.k = ti.field(dtype=ti.f32, shape=(nx, ny))  # проницаемость [m^2]
        self.m = ti.field(dtype=ti.f32, shape=(nx, ny))  # пористость
        self.m_0 = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.R = ti.field(dtype=ti.f32, shape=(nx, ny))  # концентрация взвешенного парафина

    @ti.kernel
    def initialize(self):
        for i in range(self.nx + 2):
            for j in range(self.ny + 2):
                self.p[i, j] = init_p

        for i in range(self.nx):
            for j in range(self.ny):
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
        self.p = calc_pressure(self.Wo, self.Wo_0, self.m, self.m_0, self.k, self.S, self.p)

    def update_s(self):
        self.S = calc_saturation(self.S, self.p, self.k, self.m, self.m_0)
