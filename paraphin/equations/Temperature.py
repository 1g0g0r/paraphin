from taichi import f32, field, ndrange, kernel, static

from paraphin.utils.constants import (Nx, Ny, hx, hy, dt, volume, area, ro_w, ro_f,
                                      ro_o, ro_p, K_o, K_f, K_w, K_p)
from paraphin.utils.utils import up_kw, up_ko, mid


def calc_temperature(T, m, S, C_o, C_w, C_f, C_p, Wp, Wps, p, k, mu_o, mu_w) -> field(dtype=f32, shape=(Nx, Ny)):
    """
    Вычисление температуры по явной схеме
    """
    @kernel
    def calc_temperature_loop():
        for i, j in ndrange((1, Nx - 1), (1, Ny - 1)):
            multiplier = dt / (m[i, j] * S[i, j] * ro_w * C_w[i, j] + m[i, j] * (1.0 - S[i, j]) * ro_o * C_o[i, j] +
                               (m[i, j] * (1.0 - S[i, j]) * Wps[i, j] + Wp[i, j]) * ro_p * C_p[i, j] + (1.0 - m[i, j] - Wp[i, j]) * ro_f * C_f[i, j]) / volume
            t1, t2, t3 = 0.0, 0.0, 0.0

            # цикл по граням
            arr = [[i + 1, j, hx], [i - 1, j, hx], [i, j + 1, hy], [i, j - 1, hy]]
            for idx in static(ndrange(4)):
                i1, j1, hij = arr[idx]
                temp_val = mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                               k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * (p[i, j] - p[i1, j1]) / hij

                t1 += (T[i, j] - T[i1, j1]) / hij
                t2 += up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                            k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * temp_val

                t3 += up_ko(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                            k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * temp_val

            t1 *= area * (m[i, j] * (S[i, j] * K_w + (1.0 - S[i, j]) * K_o) +
                          (m[i, j] * (1.0 - S[i, j]) * Wps[i, j] + Wp[i, j]) * K_p + (1.0 - m[i, j] - Wp[i, j]) * K_f)

            t2 *= area * T[i, j] * ro_w * C_w[i, j]

            t3 *= area * T[i, j] * (ro_o * C_o[i, j] * (1.0 - Wps[i, j]) + ro_p * Wps[i, j] * C_p[i, j])

            T[i, j] += multiplier * (t1 + t2 + t3)

    calc_temperature_loop()

    return T
