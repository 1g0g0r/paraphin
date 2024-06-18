from paraphin.utils.constants import *


@ti.func
def pf_o(s):
    """
    Функция отностельной фазовой проницаемости нефти
        [(Smax-S)/(Smax-Smin)]**2/mu_o
    """
    return ((s - S_min)/(S_max-S_min)) ** 2/mu_o


@ti.func
def pf_w(s):
    """
    Функция отностельной фазовой проницаемости воды
        [(S-Smin)/(Smax-Smin)]**Pw/Vw
    """
    return ((1 - s - S_min)/(S_max-S_min)) ** 2/mu_w
