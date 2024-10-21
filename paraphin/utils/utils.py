from taichi import func, types, f32

from paraphin.utils.phase_f import pf_o, pf_w


@func
def mid(k1: types.f32, s1: types.f32, mu_o1: types.f32, mu_w1: types.f32,
        k2: types.f32, s2: types.f32, mu_o2: types.f32, mu_w2: types.f32) -> f32:
    """mid(Ko + Kw)_ij"""
    x = K_o(k1, s1, mu_o1) + K_w(k1, s1, mu_w1)
    y = K_o(k2, s2, mu_o2) + K_w(k2, s2, mu_w2)
    return 2.0 * x * y / (x + y)


@func
def up_kw(k1: types.f32, s1: types.f32, p1: types.f32, mu_o1: types.f32, mu_w1: types.f32,
          k2: types.f32, s2: types.f32, p2: types.f32, mu_o2: types.f32, mu_w2: types.f32) -> f32:
    """up(kw / (ko + kw)"""
    ret = 0.0

    if p1 >= p2:
        ret = K_w(k1, s1, mu_w1) / (K_w(k1, s1, mu_w1) + K_o(k1, s1, mu_o1))
    else:
        ret = K_w(k2, s2, mu_w2) / (K_w(k2, s2, mu_w2) + K_o(k2, s2, mu_o2))

    return ret


@func
def up_ko(k1: types.f32, s1: types.f32, p1: types.f32, mu_o1: types.f32, mu_w1: types.f32,
          k2: types.f32, s2: types.f32, p2: types.f32, mu_o2: types.f32, mu_w2: types.f32) -> f32:
    """up(ko / (ko + kw)"""
    ret = 0.0

    if p1 >= p2:
        ret = K_o(k1, s1, mu_o1) / (K_w(k1, s1, mu_w1) + K_o(k1, s1, mu_o1))
    else:
        ret = K_o(k2, s2, mu_o2) / (K_w(k2, s2, mu_w2) + K_o(k2, s2, mu_o2))

    return ret


@func
def K_o(k: types.f32, s: types.f32, mu_o: types.f32) -> f32:
    return k * pf_o(s) / mu_o


@func
def K_w(k: types.f32, s: types.f32, mu_w: types.f32) -> f32:
    return k * pf_w(s) / mu_w
