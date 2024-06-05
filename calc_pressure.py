import numpy as np
import taichi as ti

from utils import mid
from constants import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def calc_pressure(nx, ny, Wo, Wo_0, m, m_0, k, S, p):
    """
    Сборка матрицы и решение СЛАУ уравнения давления (МКЭ)
    """

    @ti.kernel
    def build_matrix():
        # Создаем разреженную матрицу
        N = (nx + 2) * (ny + 2)
        A = sp.lil_matrix((N, N))

        for i in range(nx + 2):
            for j in range(ny + 2):
                idx = i * (ny + 2) + j
                if 1 <= i <= nx and 1 <= j <= ny:
                    A[idx, idx + 1] = - Wo[i + 1, j] * mid(i, j, i + 1, j, k, S) * area / hx
                    A[idx, idx - 1] = -Wo[i - 1, j] * mid(i, j, i - 1, j, k, S) * area / hx
                    A[idx, idx + Nx - 2] = -Wo[i, j - 1] * mid(i, j, i, j + 1, k, S) * area / hy
                    A[idx, idx - Nx + 2] = -Wo[i + 1] * mid(i, j, i, j - 1, k, S) * area / hy
                    A[idx, idx] = - A[i, i + 1] - A[i, i - 1] - A[i, i + Nx - 1] - A[i, i - Nx + 1]
                else:
                    A[idx, idx] = 1  # For boundary points, set diagonal to 1 (Neumann BC)

        return A.tocsr()

    @ti.kernel
    def build_rhs() -> np.ndarray:
        # Создаем разреженную матрицу
        N = (nx + 2) * (ny + 2)
        b = np.zeros(N)

        for i in range(nx + 2):
            for j in range(ny + 2):
                idx = i * (ny + 2) + j
                if 1 <= i <= nx and 1 <= j <= ny:
                    b[idx] = Wo[i, j] * (m[i, j] - m_0[i, j]) / dT * volume + (1 - S[i, j]) * m[i, j] * volume * (
                                Wo[i, j] - Wo_0[i, j]) / dT
                else:
                    if j == 0:
                        b[idx] = p[i, j + 1]
                    if j == ny + 1:
                        b[idx] = p[i, j - 1]
                    if i == 0:
                        b[idx] = p[i + 1, j]
                    if i == nx + 1:
                        b[idx] = p[i - 1, j]

        # Добавили скважины в точки (1,1) (nx+1, ny+1)
        b[ny + 2] += Wo[0] * qw * volume
        b[(nx + 1) * (ny + 2) + (ny + 1)] += qo * volume

        return b

    A = build_matrix()
    b = build_rhs()
    p_new = spla.spsolve(A, b).reshape((nx + 2, ny + 2))

    return p_new[1:-1, 1:-1]
