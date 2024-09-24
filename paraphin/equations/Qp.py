from taichi import f32, field, ndrange, kernel, static, exp

from paraphin.utils.constants import Nx, Ny, hx, hy, dt, ro_p, ro_o, volume, area, Tm
from paraphin.utils.utils import up_ko, mid


def calc_qp(Wps, m, fi, r) -> field(dtype=f32, shape=(Nx, Ny)):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме
    """

    @kernel
    def calc_qp_loop():

        for i, j in ndrange(Nx, Ny):


    calc_qp_loop()

    return qp