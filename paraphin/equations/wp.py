from taichi import f32, field, ndrange, kernel, static, exp

from paraphin.utils.constants import Nx, Ny, hx, hy, dt, ro_p, ro_o, volume, area, Tm
from paraphin.utils.utils import up_ko, mid


def calc_wp(Wp, Wps, T) -> field(dtype=f32, shape=(Nx, Ny)):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме
    """
    alpha = ...  # молярные фракции парафина, растворенные в нефти, в зависимости от температуры
    R = 8.31446261815324   #газовая постоянная [J⋅K^−1⋅mol^−1]
    @kernel
    def calc_wp_loop():
        temp = 1.0 / (1.8 * Tm + 32.0)
        for i, j in ndrange(Nx, Ny):
            Wp[i,j] = Wps[i,j] * exp(alpha / R * (1.0 / (1.8 * T[i,j] + 32.0) + temp))

    calc_wp_loop()

    return Wp
