import numpy as np

from constants import *
from phase_f import pf_o, pf_w


def mid(i1: int, j1: int, i2: int, j2: int, k: np.ndarray, S: np.ndarray):
    """mid(Ko + Kw)_ij"""
    x = K_o(k[i1, j1], S[i1, j1]) + K_w(k[i1, j1], S[i1, j1])
    y = K_o(k[i2, j2], S[i2, j2]) + K_w(k[i2, j2], S[i2, j2])
    return 2 * x * y / (x + y)


def up(i: int, j: int, p: np.ndarray):
    if p[i] >= p[j]:
        return K_w(i) / (K_w(i) + K_o(i))
    else:
        return K_w(j) / (K_w(j) + K_o(j))


def K_o(k: np.ndarray, S: np.ndarray):
    try:
        return k * pf_o(S) / mu_o
    except IndexError:
        print("Ошибка с индексом в функции K_o")


def K_w(k: np.ndarray, S: np.ndarray):
    try:
        return k * pf_w(S) / mu_w
    except IndexError:
        print("Ошибка с индексом в функции K_w")
