from taichi import f32, linalg, field, types, ndrange, kernel

from paraphin.utils.utils import mid
from paraphin.utils.constants import Nx, Ny, area, hx, hy, dt, volume, qw, qo


def calc_pressure(Wo, Wo_0, m, m_0, k, S, p, mu_o, mu_w) -> field(dtype=f32, shape=(Nx, Ny)):
    """
    Сборка матрицы и решение СЛАУ уравнения давления (МКО)
    """
    N = (Nx + 2) * (Ny + 2)
    mat = linalg.SparseMatrixBuilder(N, N, max_num_triplets=5 * N)
    b = field(f32, shape=N)

    @kernel
    def fill_matrix_and_rhs(A: types.sparse_matrix_builder()):
        for i, j in ndrange(Nx + 2, Ny + 2):
            idx = i * (Ny + 2) + j
            if 1 <= i <= Nx and 1 <= j <= Ny:
                # i, j -> i-1, j-1
                p1 = Wo[i, j - 1] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
                                        k[i, j - 1], S[i, j - 1], mu_o[i, j - 1], mu_w[i, j - 1]) * area / hx
                p2 = Wo[i - 2, j - 1] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
                                            k[i - 2, j - 1], S[i - 2, j - 1], mu_o[i - 2, j - 1], mu_w[i - 2, j - 1]) * area / hx
                p3 = Wo[i - 1, j - 2] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
                                            k[i - 1, j], S[i - 1, j], mu_o[i - 1, j], mu_w[i - 1, j]) * area / hy
                p4 = Wo[i - 1, j] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
                                        k[i - 1, j - 2], S[i - 1, j - 2], mu_o[i - 1, j - 2], mu_w[i - 1, j - 2]) * area / hy
                A[idx, idx + 1] -= p1
                A[idx, idx - 1] -= p2
                A[idx, idx + Nx + 2] -= p3
                A[idx, idx - Nx - 2] -= p4
                A[idx, idx] += (p1 + p2 + p3 + p4)

                # rhs
                b[idx] = Wo[i, j] * (m[i, j] - m_0[i, j]) / dt * volume + (1 - S[i, j]) * m[i, j] * volume * (
                            Wo[i, j] - Wo_0[i, j]) / dt
            else:
                A[idx, idx] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)

                if j == 0:
                    b[idx] = p[i, j + 1]
                if j == Ny + 1:
                    b[idx] = p[i, j - 1]
                if i == 0:
                    b[idx] = p[i + 1, j]
                if i == Nx + 1:
                    b[idx] = p[i - 1, j]

        # Добавили скважины в точки (1,1) (nx+1, ny+1)
        b[Ny + 2] += Wo[0, 0] * qw * volume
        b[(Nx + 1) * (Ny + 2) + (Ny + 1)] += qo * volume

    fill_matrix_and_rhs(mat)
    sparse_matrix = mat.build()

    solver = linalg.SparseSolver(solver_type="LLT", ordering="AMD")
    solver.analyze_pattern(sparse_matrix)
    solver.factorize(sparse_matrix)

    p_new = solver.solve(b.to_numpy())
    p.from_numpy(p_new.reshape((Nx + 2, Nx + 2)))

    return p
