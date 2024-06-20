from paraphin.utils.constants import *
from paraphin.utils.utils import up_ko, mid


def calc_temperature(T, m, m_0, S, C_o, C_w, C_f, p, k):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме
    """
    @ti.kernel
    def calc_temperature_loop():
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                # записать цикл по граням
                for i1, j1, h in [(i+1, j, hx), (i-1, j, hx), (i, j+1, hy), (i, j-1, hy)]:
                    r1 = up_ko(k[i, j], S[i, j], p[i, j], k[i + 1, j], S[i + 1, j], p[i + 1, j]) * \
                         mid(k[i, j], S[i, j], k[i + 1, j], S[i + 1, j]) * area * (p[i, j] - p[i + 1, j]) / hx

                T[i, j] += dT / (m[i, j] * S[i, j] * ro_w * C_w[i, j] + () * ro_p) / volume * (-(R[i, j] * ro_p + ro_o * Wp[i, j]) *
                            (m[i, j] * (1 - S[i, j]) - m_0[i, j] * (1 - S_0[i, j])) / dT - ro_o * Wp[i, j] *
                            (1 - S[i, j]) * (Wp[i, j] - Wp_0[i, j]) / dT - ro_p * qp * volume - ro_o *
                            (Wp[i, j] + Wps[i, j]) * (r1 + r2 + r3 + r4))

    calc_temperature_loop()

    return R
