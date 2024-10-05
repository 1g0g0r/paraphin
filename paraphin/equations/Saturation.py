from taichi import f32, field, ndrange, kernel, static

from paraphin.utils.constants import Nx, Ny, hx, hy, dt, qw, volume, area
from paraphin.utils.utils import up_kw, mid


def calc_saturation(S, S_0, p, k, m, m_0, mu_o, mu_w) -> (field(dtype=f32, shape=(Nx, Ny)),
                                                          field(dtype=f32, shape=(Nx, Ny))):
    """
    Вычисление водонасыщенности по явной схеме.
    """
    @kernel
    def calc_saturation_loop():
        for i, j in ndrange((1, Nx - 1), (1, Ny - 1)):
            temp_val = 0.0
            S_0[i, j] = S[i, j]
            arr = [[i+1, j, hx], [i-1, j, hx], [i, j+1, hy], [i, j-1, hy]]

            # расчет средней скорости воды
            # вопрос Vx Vy
            for idx in static(ndrange(4)):
                i1, j1, hij = arr[idx]
                temp_val += up_kw(k[i, j], S[i, j], p[i, j], mu_o[i, j], mu_w[i, j],
                                  k[i1, j1], S[i1, j1], p[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * \
                            mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])*area*(p[i, j] - p[i1, j1])/hij

            S[i, j] += -S[i, j] * (m[i, j] - m_0[i, j]) / m[i, j] - temp_val / m[i, j] / volume

    calc_saturation_loop()
    S[0, 0] += dt * qw / m[0, 0]

    return S, S_0
