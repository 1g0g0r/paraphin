from constants import *
from utils import up, mid, K_o, K_w


def calc_saturation(cells):
    """
    Вычисление водонасыщенности по явной схеме
    """
    for i, elem in enumerate(cells):
        if (i < Nx-1) or (i % (Nx-1) == 0) or ((i+1) % (Nx-1) == 0) or (i > (Ny-2) * (Nx-1)):
            S[i] = 1  # TODO ГУ !!!!!
        else:
            s1 = up(i, i + 1) * mid(i, i + 1) * area * (p[i] - p[i + 1]) / hx
            s2 = up(i, i - 1) * mid(i, i - 1) * area * (p[i] - p[i - 1]) / hx
            s3 = up(i, i + Nx - 1) * mid(i, i + Nx - 1) * area * (p[i] - p[i + Nx - 1]) / hy
            s4 = up(i, i - Nx + 1) * mid(i, i - Nx + 1) * area * (p[i] - p[i - Nx + 1]) / hy
            S[i] += -S[i] * (m[i] - m_0[i]) / m[i] - (s1 + s2 + s3 + s4) / m[i] / volume[i]

    S[0] += dt * qw / m[0]
