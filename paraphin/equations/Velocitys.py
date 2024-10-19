from taichi import func

from paraphin.utils.constants import (D, gamma, betta, Diff, Lk)

# Lk: float
#     средняя длина капилляра, [м]
# gamma: float
#     Отношение радиуса горла к радиусу канала
# delta: float
#     Кинетическая константа суффозии, [1/м]
# Diff: float
#     Коэффициент диффузионного осаждения частиц, [м2/сек]
# D: float
#     Размер частицы(диаметр частицы), [м]


@func
def u_r(wps: float, Um: float, r: float, h: float) -> float:
    """Скорость изменения радиуса капилляра.

    Parameters
    ----------
    wps: float
        Объемная концентрация взвешенных частиц парафина, [м3/м3]
    Um: float
        Среднее значение скорости жидкости в канале, [м/сут]
    r: float
        Радиус капилляра, [м]
    h: float
        толщина осадочного слоя, [м]

    Returns
    -------
    Ur: float
        Скорость изменения радиуса капилляра, [м/сут]
    """
    if 2.0 * r * gamma < D:
        Ur = 0.0
    else:
        # Сужение(кольматация)
        Ur = -wps * (2 * Um * Diff ** 2 / (r * Lk)) ** (1/3)

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
    fi: float
        Значение функции распределения пор по размерам
    r: float
        Радиус капилляра, [м]

    Returns
    -------
    Ub: float
        Скорость блокирования капилляров, [м/сут]
    """
    if 2.0 * r * gamma <= D:
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
    TODO узнать, как вычисляется объем частицы

    Return
    ------
    uc: float
        Критическая скорость
    """
    if r <= r_max:
        return - beta * wps * fi * Um / nu
    else:
        return 0.0
