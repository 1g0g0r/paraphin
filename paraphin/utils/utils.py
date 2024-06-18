from paraphin.utils.constants import *
from paraphin.utils.phase_f import pf_o, pf_w


@ti.func
def mid(k1: ti.types.f32, s1: ti.types.f32, mu_o1: ti.types.f32, mu_w1: ti.types.f32,
        k2: ti.types.f32, s2: ti.types.f32, mu_o2: ti.types.f32, mu_w2: ti.types.f32) -> ti.f32:
    """mid(Ko + Kw)_ij"""
    x = K_o(k1, s1, mu_o1) + K_w(k1, s1, mu_w1)
    y = K_o(k2, s2, mu_o2) + K_w(k2, s2, mu_w2)
    return 2 * x * y / (x + y)


@ti.func
def up_kw(k1: ti.types.f32, s1: ti.types.f32, p1: ti.types.f32, mu_o1: ti.types.f32, mu_w1: ti.types.f32,
          k2: ti.types.f32, s2: ti.types.f32, p2: ti.types.f32, mu_o2: ti.types.f32, mu_w2: ti.types.f32) -> ti.f32:
    """up(kw / (ko + kw)"""
    ret = 0.0

    if p1 >= p2:
        ret = K_w(k1, s1, mu_w1) / (K_w(k1, s1, mu_w1) + K_o(k1, s1, mu_o1))
    else:
        ret = K_w(k2, s2, mu_w2) / (K_w(k2, s2, mu_w2) + K_o(k2, s2, mu_o2))

    return ret


@ti.func
def up_ko(k1: ti.types.f32, s1: ti.types.f32, p1: ti.types.f32, mu_o1: ti.types.f32, mu_w1: ti.types.f32,
          k2: ti.types.f32, s2: ti.types.f32, p2: ti.types.f32, mu_o2: ti.types.f32, mu_w2: ti.types.f32) -> ti.f32:
    """up(ko / (ko + kw)"""
    ret = 0.0

    if p1 >= p2:
        ret = K_o(k1, s1, mu_o1) / (K_w(k1, s1, mu_w1) + K_o(k1, s1, mu_o1))
    else:
        ret = K_o(k2, s2, mu_o2) / (K_w(k2, s2, mu_w2) + K_o(k2, s2, mu_o2))

    return ret


@ti.func
def K_o(k: ti.types.f32, s: ti.types.f32, mu_o: ti.types.f32) -> ti.f32:
    return k * pf_o(s, mu_o) / mu_o


@ti.func
def K_w(k: ti.types.f32, s: ti.types.f32, mu_w: ti.types.f32) -> ti.f32:
    return k * pf_w(s, mu_w) / mu_w
