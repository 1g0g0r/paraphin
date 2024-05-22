from my_consatnts import *
from phase_f import pf_o, pf_w


def mid(i, j):
    """mid(Ko + Kw)_ij"""
    x = K_o(i) + K_w(i)
    y = K_o(j) + K_w(j)
    return 2 * x * y / (x + y)


def up(i, j):
    if p[i] >= p[j]:
        return K_w(i) / (K_w(i) + K_o(i))
    else:
        return K_w(j) / (K_w(j) + K_o(j))


def K_o(idx: int):
    try:
        return k[idx] * pf_o(S[idx]) / mu_o
    except IndexError:
        print("Ошибка с индексом в функции K_o")


def K_w(idx: int):
    try:
        return k[idx] * pf_w(S[idx]) / mu_w
    except IndexError:
        print("Ошибка с индексом в функции K_w")
