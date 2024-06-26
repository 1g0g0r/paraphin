from paraphin.utils.constants import *
from paraphin.utils.utils import up_ko, mid


def calc_particles(qp, R, m, m_0, S, S_0, Wp, Wp_0, Wps, p, k, mu_o, mu_w):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме
    """
    @ti.kernel
    def calc_particles_loop():
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                temp_val = 0.0
                for i1, j1, hij in [(i + 1, j, hx), (i - 1, j, hx), (i, j + 1, hy), (i, j - 1, hy)]:
                    temp_val += up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                      k[i1, j1], S[i1, j1], p[i1+1, j1+1], mu_o[i1, j1], mu_w[i1, j1]) * \
                                mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                    k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])*area*(p[i+1, j+1] - p[i1+1, j+1])/hij

                R[i, j] += dT / (m[i, j] * (1-S[i, j]) * ro_p * volume) * (-(R[i, j] * ro_p + ro_o * Wp[i, j]) *
                            (m[i, j] * (1 - S[i, j]) - m_0[i, j] * (1 - S_0[i, j])) / dT - ro_o * Wp[i, j] *
                            (1 - S[i, j]) * (Wp[i, j] - Wp_0[i, j]) / dT - ro_p * qp * volume - ro_o *
                            (Wp[i, j] + Wps[i, j]) * temp_val)

    calc_particles_loop()

    return R
