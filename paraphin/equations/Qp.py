from taichi import f32, field, ndrange, func, kernel, static, exp

from paraphin.utils.constants import (Nx, Ny, hx, hy, dt, ro_p, ro_o, volume, area, Tm,
                                      D, gamma, betta, Diff)
from paraphin.utils.utils import up_ko, mid


def calc_qp(Wps, m, fi, r) -> field(dtype=f32, shape=(Nx, Ny)):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме
    """

    @kernel
    def calc_qp_loop():

        for i, j in ndrange(Nx, Ny):
            ...

    calc_qp_loop()

    return qp


@func
def u_r(wps, Um, r, L) -> float:
    """Скорость изменения радиуса капилляра.

    Parameters
    ----------
    wps: float
        Объемная концентрация взвешенных частиц парафина, [м3/м3]
    Um: float
        Cреднее значение скорости жидкости в канале, [м/сут]
    D: float
        Размер частицы (диаметр частицы), [м]
    r: float
        Радиус капилляра, [м]
    h: float
        толщина осадочного слоя, [м]
    Lk: float
        средняя длина капилляра, [м]
    gamma: float
        Отношение радиуса горла к радиусу канала
    delta: float
        Кинетическая константа суффозии, [1/м]
    diff: float
        Коэффициент диффузионного осаждения частиц, [м2/сек]
    Returns
    -------
    Ur: float
        Скорость изменения радиуса капилляра, [м/сут]
    """
    if 2 * r * gamma < D:
        Ur = 0.0
    else:
        # Сужение(кольматация)
        Ur = -wps * (2 * Um * (86400 * DIFFo) ** 2 / (R * Lk)) ** (1/3)

        # Расширение(суффозия) каналов
        if Um > Uk and h > 0:
            Ur += DELTA * (Um - Uk) * h * (r + h * 0.5) / r

    return Ur


@func
def u_b(Um, wps, fi, r) -> float:
    """Интенсивность блокирования капилляров.

    Parameters
    ----------
    Um: float
        Средняя скорость жидкости в капилляре, [м/сут]
    wps: float
        Концентрация частиц в потоке, [м3/м3]
    r: float
        Радиус капилляра, [м]
    fi: float
        Значение функции распределения пор по размерам
    D: float
        Размер частицы (диаметр частицы), [м]
    betta: float
         Коэффициент формы частицы (beta<=1)
    gamma: float
        Отношение радиуса горла к радиусу канала
    Returns
    -------
    Ub: float
        Интенсивность блокирования капилляров, [м/сут]
    """
    if 2.0 * gamma * r <= D:
        return 6.0 * betta * wps * r * r * fi * Um / D**3
    else:
        return 0.0


def u_c() -> float:
    """Критическая скорость

    Parameters
    ----------
    nu: float
        Объем частицы
    Return
    ------
    uc: float
        Критическая скорость
    """
    if r <= r_max:
        return - beta * wps * fi * um / nu
    else:
        return 0.0