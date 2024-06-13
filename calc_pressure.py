import numpy as np
import taichi as ti

from utils import mid
from constants import *


def calc_pressure(nx, ny, Wo, Wo_0, m, m_0, k, S, p):
    """
    Сборка матрицы и решение СЛАУ уравнения давления (МКЭ)
    """
    N = (nx + 2) * (ny + 2)
    mat = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=5 * N)
    b = ti.field(ti.f32, shape=N)

    @ti.kernel
    def fill_matrix_and_rhs(A: ti.types.sparse_matrix_builder()):
        for i in range(nx + 2):
            for j in range(ny + 2):
                idx = i * (ny + 2) + j
                if 1 <= i <= nx and 1 <= j <= ny:
                    a1 = Wo[i + 1, j] * mid(k[i, j], S[i, j], k[i + 1, j], S[i + 1, j]) * area / hx
                    a2 = Wo[i - 1, j] * mid(k[i, j], S[i, j], k[i - 1, j], S[i - 1, j]) * area / hx
                    a3 = Wo[i, j - 1] * mid(k[i, j], S[i, j], k[i, j + 1], S[i, j + 1]) * area / hy
                    a4 = Wo[i, j + 1] * mid(k[i, j], S[i, j], k[i, j - 1], S[i, j - 1]) * area / hy
                    A[idx, idx + 1] -= a1
                    A[idx, idx - 1] -= a2
                    A[idx, idx + Nx - 2] -= a3
                    A[idx, idx - Nx + 2] -= a4
                    A[idx, idx] += (a1 + a2 + a3 + a4)

                    # rhs
                    b[idx] = Wo[i, j] * (m[i, j] - m_0[i, j]) / dT * volume + (1 - S[i, j]) * m[i, j] * volume * (Wo[i, j] - Wo_0[i, j]) / dT
                else:
                    A[idx, idx] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)

                    if j == 0:
                        b[idx] = p[i, j + 1]
                    if j == ny + 1:
                        b[idx] = p[i, j - 1]
                    if i == 0:
                        b[idx] = p[i + 1, j]
                    if i == nx + 1:
                        b[idx] = p[i - 1, j]

        # Добавили скважины в точки (1,1) (nx+1, ny+1)
        b[ny + 2] += Wo[0, 0] * qw * volume
        b[(nx + 1) * (ny + 2) + (ny + 1)] += qo * volume

    fill_matrix_and_rhs(mat)
    sparse_matrix = mat.build()
    print('matrix build')

    solver = ti.linalg.SparseSolver(solver_type="LU")  # ti.linalg.SolverType.LLT, ti.linalg.Ordering.AMD
    solver.analyze_pattern(sparse_matrix)
    solver.factorize(sparse_matrix)

    p_new = solver.solve(b.to_numpy())

    return p_new
