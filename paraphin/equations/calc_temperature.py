from paraphin.utils.constants import *
from paraphin.utils.utils import up_kw, up_ko, mid


def calc_temperature(T, m, m_0, S, C_o, C_w, C_f, C_p, R, Wps, p, k):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме
    """
    @ti.kernel
    def calc_temperature_loop():
        # TODO for i, j in ti.ndrange(nx, ny):
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                multiplier = dT / (m[i, j] * S[i, j] * ro_w * C_w[i, j] + m[i, j] * (1 - S[i, j]) * ro_o * C_o[i, j] +
                                   (m * (1 - S[i, j]) * R + E_p) * ro_p * C_p + (1 - m - E_p) * ro_f * C_f) / volume
                t1, t2, t3 = 0, 0, 0
                # цикл по граням
                for i1, j1, hij in [(i+1, j, hx), (i-1, j, hx), (i, j+1, hy), (i, j-1, hy)]:
                    temp_val = mid(k[i, j], S[i, j], k[i1, j1], S[i1, j1]) * (p[i, j] - p[i1, j1]) / hij

                    t1 += area * (T[i, j] - T[i1, j1]) / hij
                    t2 += up_kw(k[i, j], S[i, j], p[i, j], k[i1, j1], S[i1, j1], p[i1, j1]) * temp_val

                    t3 += up_ko(k[i, j], S[i, j], p[i, j], k[i1, j1], S[i1, j1], p[i1, j1]) * temp_val

                t1 *= (m[i, j] * (S[i, j] * K_w + (1-S[i, j]) * K_o) +
                       (m[i, j] * (1-S[i, j]) * R[i, j] + E_p) * K_p + (1 - m[i, j] - E_p) * K_f)

                t2 *= T[i, j] * ro_w * C_w

                t3 *= T[i, j] * (ro_o * C_o + ro_o * Wps * C_p) * area

    calc_temperature_loop()

    return T
