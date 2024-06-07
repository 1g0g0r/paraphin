import numpy as np
import taichi as ti

from utils import mid
from constants import *


def calc_pressure(nx, ny, Wo, Wo_0, m, m_0, k, S, p):
    """
    Сборка матрицы и решение СЛАУ уравнения давления (МКЭ)
    """
    N = (nx + 2) * (ny + 2)

    Wo_np, Wo_0_np, m_np, m_0_np, k_np, S_np, p_np = Wo.to_numpy(), Wo_0.to_numpy(), m.to_numpy(), \
        m_0.to_numpy(), k.to_numpy(), S.to_numpy(), p.to_numpy()

    # @ti.kernel
    def fill_matrix_and_rhs(A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray(), Wo: ti.types.ndarray()):
        for i in range(nx + 2):
            for j in range(ny + 2):
                idx = i * (ny + 2) + j
                if 1 <= i <= nx and 1 <= j <= ny:
                    A[idx, idx + 1] -= Wo[i + 1, j] * mid(i, j, i + 1, j, k, S) * area / hx
                    A[idx, idx - 1] -= Wo_np[i - 1, j] * mid(i, j, i - 1, j, k, S) * area / hx
                    A[idx, idx + Nx - 2] -= Wo_np[i, j - 1] * mid(i, j, i, j + 1, k, S) * area / hy
                    A[idx, idx - Nx + 2] -= Wo_np[i, j + 1] * mid(i, j, i, j - 1, k, S) * area / hy
                    A[idx, idx] -= (A[idx, idx + 1] + A[idx, idx - 1] + A[idx, idx + Nx - 1] + A[idx, idx - Nx + 1])

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
        b[ny + 2] += Wo[0] * qw * volume
        b[(nx + 1) * (ny + 2) + (ny + 1)] += qo * volume

    mat = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=5*N)
    b = ti.ndarray(ti.f32, shape=N)
    fill_matrix_and_rhs(mat, b, Wo)
    sparse_matrix = mat.build()

    solver = ti.linalg.SparseSolver(ti.linalg.SolverType.LLT, ti.linalg.Ordering.AMD)
    solver.analyze_pattern(sparse_matrix)
    solver.factorize(sparse_matrix)

    p_new = solver.solve(b)

    return p_new
