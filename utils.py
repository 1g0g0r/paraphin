import numpy as np
import taichi as ti

from constants import *
from phase_f import pf_o, pf_w


@ti.func
def mid(k1: ti.types.f32, s1: ti.types.f32, k2: ti.types.f32, s2: ti.types.f32):
    """mid(Ko + Kw)_ij"""
    x = K_o(k1, s1) + K_w(k1, s1)
    y = K_o(k2, s2) + K_w(k2, s2)
    return 2 * x * y / (x + y)


def up(i: int, j: int, p: np.ndarray):
    if p[i] >= p[j]:
        return K_w(i) / (K_w(i) + K_o(i))
    else:
        return K_w(j) / (K_w(j) + K_o(j))


@ti.func
def K_o(k: ti.types.f32, s: ti.types.f32):
    return k * pf_o(s) / mu_o


@ti.func
def K_w(k: ti.types.f32, s: ti.types.f32):
    return k * pf_w(s) / mu_w
