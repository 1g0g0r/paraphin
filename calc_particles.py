from constants import *


def calc_particles(cells):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме
    """

    for i in range(len(cells)):
        R[i] += dt / (m[i] * (1-S[i]) * ro_p * volume[i]) * (-R[i])
