from taichi import func

from paraphin.constants import S_min, S_max


@func
def pf_o(s, mu_o):
    """
    Функция отностельной фазовой проницаемости нефти
        [(Smax-S)/(Smax-Smin)]**2/mu_o
    """
    return ((s - S_min) / (S_max-S_min)) ** 2 / mu_o


@func
def pf_w(s, mu_w):
    """
    Функция отностельной фазовой проницаемости воды
        [(S-Smin)/(Smax-Smin)]**Pw/Vw
    """
    return ((1 - s - S_min) / (S_max-S_min)) ** 2 / mu_w
