from paraphin.utils.constants import *


def calc_particles(cells):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме
    """

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            R[i] += dt / (m[i] * (1-S[i]) * ro_p * volume[i]) * (-R[i])
