from numpy import zeros, float32, gradient
from numpy.linalg import norm
from taichi import f32, field, ndrange, func, kernel

from paraphin.constants import (Nx, Ny, Nr, dt, ro_o, eta)
from .Velocitys import u_c, u_b, u_r


def calc_qp(p, Wps, mu_o, m, fi, h_sloy, r, integr_r2_fi0, integr_r4_fi0) -> (field(dtype=f32, shape=(Nx, Ny)),
                                                                        field(dtype=f32, shape=(Nx, Ny)),
                                                                        field(dtype=f32, shape=(Nx, Ny))):
    """
    Вычисление концентрации взвешенных частиц парафина по явной схеме

    Parameters
    ----------
    p: taichi.field(Nx, Ny)
        Давление
    Wps: taichi.field(Nx, Ny)
        Концентрации взвешенных частиц парафина
    m: taichi.field(Nx, Ny)
        Пористость
    fi: taichi.field(Nx, Ny, Nr)
        Функция распределения пор по размеру
    h_sloy: taichi.field(Nx, Ny, Nr)
        Толщина осадочного слоя
    r: taichi.field(Nr)
        Радиусы пор
    integr_r2_fi0: float
        Интеграл r^2 * fi_o(r)
    integr_r4_fi0: float
        Интеграл r^4 * fi_o(r)

    Returns
    -------
    qp: taichi.field(Nx, Ny)
         Скорость отложения парафиновых отложений в общем объеме пористой породы
    fi: taichi.field(Nx, Ny, Nr)
        Обновленная функция распределения пор по размеру
    """
    ur = zeros(Nr, dtype=float32)
    ub = zeros(Nr, dtype=float32)
    uc = zeros(Nr, dtype=float32)


    Um_r2 = norm(gradient(p.to_numpy()), axis=0) * 0.125 / eta / mu_o.to_numpy()

    @kernel
    def calc_qp_loop():

        for i in ndrange(Nx):
            for j in ndrange(Ny):

                # Расчет скоростей Ur, Ub, Uc
                for ij in ndrange(Nr):
                    u_m = Um_r2[i, j] * r[ij] * r[ij]
                    # TODO спросить про осадочный слой
                    ur[ij] = u_r(Wps[i,j], Um[i,j], r[ij], h_sloy)
                    ub[ij] = u_b(Um[i,j], Wps[i,j], fi[i,j,ij], r[ij])
                    uc[ij] = u_c(r[ij], mu_o[i,j], ro_o[i, j])
                    h_sloy[i, j, ij] = sed_h(h_sloy[i, j, ij], ur, r)

                # Расчет интегралов qp1, qp2, kf, mf
                qp1 = 0.0
                qp2 = 0.0
                kf = 0.0
                mf = 0.0
                r3_old = rr[0] ** 3
                r4_old = rr[0] * r3_old
                for ij in ndrange(1, Nr):
                    dr = r[ij] - r[ij-1]
                    a = (fi[i,j,ij-1] * r[ij] - fi[i,j,ij] * r[ij-1]) / dr
                    b = (fi[i,j,ij] - fi[i,j,ij-1]) / dr
                    qp1 = ...  # вопрос нужно ли использовать u_r(r) интегрировании
                    qp2 = ... # вопрос нужно ли использовать u_b(r) интегрировании
                    kf = ...
                    mf = ...

                for ij in ndrange(1, Nr):
                    fi[i, j] = upd_fi(fi[i, j, ij], u_r[i, j], fi[i, j, ij - 1], u_r[ij - 1], r[ij] - r[ij - 1], u_b[ij])

    calc_qp_loop()

    return qp


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


def sed_h(h0, ur, r):
    hr = h0 - dt * ur
    hr = max(0.0, min(hr, r - 1e-7))
    return hr