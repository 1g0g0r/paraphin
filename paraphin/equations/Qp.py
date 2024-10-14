from taichi import f32, field, ndrange, func, kernel, static, exp
from numpy import zeros, float32

from paraphin.utils.constants import (Nx, Ny, Nr, hx, hy, dt, ro_p, ro_o, volume, area, Tm,
                                      D, gamma, betta, Diff)
from paraphin.utils.utils import up_ko, mid


def calc_qp(Wps, Um, m, fi, r) -> field(dtype=f32, shape=(Nx, Ny)):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме

    Parameters
    ----------
    Wps: taichi.field(Nx, Ny)
        Концентрации взвешенных частиц парафина
    Um: taichi.field(Nx, Ny)
        Концентрация растворенного парафина
    m: taichi.field(Nx, Ny)
        Пористость
    fi: taichi.field(Nx, Ny, Nr)
        Функция распределения пор по размеру
    r: taichi.field(Nr)
        Радиусы пор

    Returns
    -------
    qp: taichi.field(Nx, Ny)
         Скорость отложения парафиновых отложений в общем объеме пористой породы
    fi: taichi.field(Nx, Ny, Nr)
        Обновленная функция распределения пор по размеру
    """
    u_r = zeros(Nr, dtype=float32)
    u_b = zeros(Nr, dtype=float32)
    u_c = zeros(Nr, dtype=float32)

    @kernel
    def calc_qp_loop():

        for i in ndrange(Nx):
            for i, j in ndrange(Ny):

                # Расчет скоростей Ur, Ub, Uc
                for ij in ndrange(Nr):
                    u_r[ij] = u_r(Wps[i,j], Um[i,j], r[ij])
                    u_b[ij] = u_b(Um[i,j], Wps[i,j], fi[i,j,ij], r[ij])
                    u_c[ij] = u_c(Wps[i,j], fi[i,j,ij], Um[i,j])

                # Расчет интегралов qp1, qp2, kf, mf
                qp1 = 0.0
                qp2 = 0.0
                kf = 0.0
                mf = 0.0
                for ij in ndrange(1, Nr):
                    dr = r[ij] - r[ij-1]
                    a = (fi[i,j,ij-1] * r[ij] - fi[i,j,ij] * r[ij-1]) / dr
                    b = (fi[i,j,ij] - fi[i,j,ij-1]) / dr
                    qp1 = ...
                    qp2 = ...
                    kf = ...
                    mf = ...

                for ij in ndrange(1, Nr):
                    fi[i, j] = upd_fi(fi[i, j, ij], u_r[i, j], fi[i, j, ij - 1], u_r[ij - 1], r[ij] - r[ij - 1], u_b[ij])

    calc_qp_loop()

    return qp


@func
def u_r(wps: float, Um: float, r: flaot) -> float:
    """Скорость изменения радиуса капилляра.

    Parameters
    ----------
    wps: float
        Объемная концентрация взвешенных частиц парафина, [м3/м3]
    Um: float
        Среднее значение скорости жидкости в канале, [м/сут]
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
        Ur = -wps * (2 * Um * (86400 * diff) ** 2 / (r * Lk)) ** (1/3)

        # Расширение(суффозия) каналов
        if Um > Uk and h > 0:
            Ur += delta * (Um - Uk) * h * (r + h * 0.5) / r

    return Ur


@func
def u_b(Um: float, wps: float, fi: float, r: float) -> float:
    """Скорость блокирования капилляров.

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
        Скорость блокирования капилляров, [м/сут]
    """
    if 2.0 * gamma * r <= D:
        return 6.0 * betta * wps * r * r * fi * Um / D**3
    else:
        return 0.0


@func
def u_c(wps: float, fi: float, Um: float) -> float:
    """Критическая скорость

    Parameters
    ----------
    wps: float
        Концентрация частиц в потоке, [м3/м3]
    fi: float
        Значение функции распределения пор по размерам
    Um: float
        Средняя скорость жидкости в капилляре, [м/сут]
    nu: float
        Объем частицы

    Return
    ------
    uc: float
        Критическая скорость
    """
    if r <= r_max:
        return - beta * wps * fi * Um / nu
    else:
        return 0.0


@func
def upd_fi(fi: float, Ur: float, fi1: float, Ur1: float, dr: float, Ub: float) -> float:
    """
    Обновление функции пор по размерам.

    Parameters
    ----------
    fi: float
        fi[i] - функции распределения пор по размерам
    Ur: float
        ur[i] - Скорость изменения радиуса капилляра
    fi1: float
        fi[i-1] - функции распределения пор по размерам
    Ur1: float
        ur[i-1] - Скорость изменения радиуса капилляра
    dr: float
        r[i]-r[i-1] - шаг дискретизации
    Ub: float
        ub[i] - Скорость блокирования капилляров, [м/сут]

    Returns
    -------
    fi: float
        Обновленная функции пор по размерам
    """

    fi -= dt * ((fi * Ur - fi1 * Ur1) / dr + Ub)
    return fi
