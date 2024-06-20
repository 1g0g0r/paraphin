import numpy as np
from time import time

from paraphin.utils.constants import dT, Time_end
from paraphin.solver import Solver


def solve():
    sol = Solver()
    sol.initialize()
    # Задать начальные условия

    tim = np.linspace(0, 1, int(Time_end/dT))
    for t in tim:
        tt = time()
        sol.update_p()  # Обновление давления

        sol.update_s()  # Обновление насыщенности

        sol.update_r()  # Обновлнние концентрации взвешенного парафина

        print(time()-tt)
        if t > 2 * dT:
            break


if __name__ == '__main__':
    solve()
    #     https://github.com/hejob/taichi-fvm2d-fluid-ns

