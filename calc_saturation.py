from constants import *
from utils import up, mid


def calc_saturation(nx, ny, S, p, k, m, m_0) -> ti.field(dtype=ti.f32, shape=(Nx, Ny)):
    """
    Вычисление водонасыщенности по явной схеме
    """
    @ti.kernel
    def calc_saturation_loop():
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                s1 = up(k[i, j], S[i, j], p[i, j], k[i + 1, j], S[i + 1, j], p[i + 1, j]) * \
                     mid(k[i, j], S[i, j], k[i + 1, j], S[i + 1, j]) * area * (p[i, j] - p[i + 1, j]) / hx
                s2 = up(k[i, j], S[i, j], p[i, j], k[i - 1, j], S[i - 1, j], p[i - 1, j]) * \
                     mid(k[i, j], S[i, j], k[i - 1, j], S[i - 1, j]) * area * (p[i, j] - p[i - 1, j]) / hx
                s3 = up(k[i, j], S[i, j], p[i, j], k[i, j + 1], S[i, j + 1], p[i, j + 1]) * \
                     mid(k[i, j], S[i, j], k[i, j + 1], S[i, j + 1]) * area * (p[i, j] - p[i, j + 1]) / hy
                s4 = up(k[i, j], S[i, j], p[i, j], k[i, j - 1], S[i, j - 1], p[i, j - 1]) * \
                     mid(k[i, j], S[i, j], k[i, j - 1], S[i, j + 1]) * area * (p[i, j] - p[i, j - 1]) / hy
                S[i, j] += -S[i, j] * (m[i, j] - m_0[i, j]) / m[i, j] - (s1 + s2 + s3 + s4) / m[i, j] / volume

    calc_saturation_loop()
    S[0, 0] += dT * qw / m[0, 0]

    return S
