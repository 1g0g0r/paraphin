from paraphin.utils.constants import *
from paraphin.utils.utils import up_kw, mid


def calc_saturation(S, S_0, p, k, m, m_0, mu_o, mu_w) -> (ti.field(dtype=ti.f32, shape=(Nx, Ny)),
                                                          ti.field(dtype=ti.f32, shape=(Nx, Ny))):
    """
    Вычисление водонасыщенности по явной схеме
    """
    @ti.kernel
    def calc_saturation_loop():
        for i, j in ti.ndrange((1, Nx - 1), (1, Ny - 1)):
            temp_val = 0.0
            S_0[i, j] = S[i, j]
            # для давления i,j = i+1, j+1
            arr = [[i+1, j, hx], [i-1, j, hx], [i, j+1, hy], [i, j-1, hy]]
            for idx in ti.static(ti.ndrange(4)):
                i1, j1, hij = arr[idx]
                temp_val += up_kw(k[i, j], S[i, j], p[i+1, j+1], mu_o[i, j], mu_w[i, j],
                                  k[i1, j1], S[i1, j1], p[i1+1, j1+1], mu_o[i1, j1], mu_w[i1, j1]) * \
                            mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1])*area*(p[i+1, j+1] - p[i1+1, j1+1])/hij

            S[i, j] += -S[i, j] * (m[i, j] - m_0[i, j]) / m[i, j] - temp_val / m[i, j] / volume

    calc_saturation_loop()
    S[0, 0] += dT * qw / m[0, 0]

    return S, S_0
