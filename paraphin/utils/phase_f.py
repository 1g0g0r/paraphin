from taichi import func

from paraphin.constants import S_min, S_max


@func
def pf_o(s):
    """
    Функция отностельной фазовой проницаемости нефти
        [(Smax-S)/(Smax-Smin)]**2
    """
    # TODO спросить про значения, выходяящие за границы [Smin, Smax]
    return ((S_max - s) / (S_max-S_min)) ** 2


@func
def pf_w(s):
    """
    Функция отностельной фазовой проницаемости воды
        [(S-Smin)/(Smax-Smin)]**2
    """
    # TODO спросить про значения, выходяящие за границы [Smin, Smax]
    return ((s - S_min) / (S_max-S_min)) ** 2
