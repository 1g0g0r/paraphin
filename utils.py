from my_consatnts import *
from phase_f import pf_o, pf_w


def mid(x, y):
    return 2 * x * y / (x + y)


def K_o(idx: int):
    return k[idx] * pf_o(S[idx]) / mu_o


def K_w(idx: int):
    return k[idx] * pf_w(S[idx]) / mu_w
