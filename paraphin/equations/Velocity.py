from taichi import f32, field, ndrange, kernel
from numpy import gradient

from paraphin.utils.constants import Nx, Ny, hx, hy, dt, qw, volume, area
from paraphin.utils.utils import K_o, K_w


def calc_velocity(p, k, S, mu_o, mu_w) -> (field(dtype=f32, shape=(Nx, Ny)),
                                                          field(dtype=f32, shape=(Nx, Ny))):
    """
    Вычисление водонасыщенности по явной схеме.
    """
    Ko = field(dtype=f32, shape=(Nx, Ny))
    Kw = field(dtype=f32, shape=(Nx, Ny))
    @kernel
    def calc_velocity_loop():
        for i in ndrange(Nx):
            for j in ndrange(Ny):
                Ko[i, j] = K_o(k[i, j], S[i, j], mu_o[i, j])
                Kw[i, j] = K_w(k[i, j], S[i, j], mu_w[i, j])

    calc_velocity_loop()

    dp_dx, dp_dy = gradient(p)

    # Вычисление поля скоростей по формуле Дарси
    Vx_o = -Ko.to_numpy() * dp_dx
    Vy_o = -Ko.to_numpy() * dp_dy

    Vx_w = -Kw.to_numpy() * dp_dx
    Vy_w = -Kw.to_numpy() * dp_dy

    return S, S_0  # вопрос как получить сред скорость
