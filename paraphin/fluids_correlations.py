from paraphin.utils.constants import *


@ti.func
def calc_mu_o(t: ti.types.f32):
    """Вязкость нефти [Pa*c] Уравнение Аррениуса"""
    return 0.001 * ti.exp(5000 / 8.314 / (t + 273.15))


@ti.func
def calc_mu_w(t: ti.types.f32):
    """Вязкость воды [Pa*c]  уравнение Андраде"""
    return 2.414 * 10 ** -5 * 10 ** (247.8 / (t + 133.15))


@ti.func
def calc_c_w(t: ti.types.f32):
    """"Теплоемкость воды [Дж/K]"""
    return 4217 - 2.15 * t + 0.002 * t ** 2


@ti.func
def calc_c_o(t: ti.types.f32):
    """"Теплоемкость нефти [Дж/K]"""
    return 1800 + 4 * t + 0.01 * t ** 2


@ti.func
def calc_c_f(t: ti.types.f32):
    """"Теплоемкость пласта [Дж/K]"""
    return 800 + 0.75 * t


# TODO добавить теплопроводности
# oil 0.13 + 0.0005(t-20)
# water 0.58 + 0.001(t-20)
# paraphin 0.25 + 0.0007(t-20)
# formation 2.5 + 0.0008(t-20)
