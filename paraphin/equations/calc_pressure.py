from taichi import f32, linalg, field, types, ndrange, kernel, static

from paraphin.utils.constants import Nx, Ny, area, hx, hy, dt, volume, qw, qo
from paraphin.utils.utils import mid


def calc_pressure(Wo, Wo_0, m, m_0, k, S, p, mu_o, mu_w) -> field(dtype=f32, shape=(Nx, Ny)):
    """
    Сборка матрицы и решение СЛАУ уравнения давления (МКО)
    """
    N = Nx * Ny
    mat = linalg.SparseMatrixBuilder(N, N, max_num_triplets=5 * N)
    b = field(f32, shape=N)

    @kernel
    def fill_matrix_and_rhs(A: types.sparse_matrix_builder()):
        p1, p2, p3, p4 = 0, 0, 0, 0
        bc = 17.0  # TODO проверить можно ли поставить ноль

        for i, j in ndrange((1, Nx-1), (1, Ny-1)):
            idx = j * Nx + i

            # matrix
            arr = [[i + 1, j, hx], [i - 1, j, hx], [i, j + 1, hy], [i, j - 1, hy]]
            for qq in static(ndrange(4)):
                i1, j1, hij = arr[qq]
                p = Wo[i1, j1] * mid(k[i, j], S[i, j], mu_o[i, j], mu_w[i, j],
                                     k[i1, j1], S[i1, j1], mu_o[i1, j1], mu_w[i1, j1]) * area / hij
                A[idx, idx + (i1-i) + Nx * (j1-j)] -= p
                A[idx, idx] += p

            # rhs
            b[idx] = (Wo[i, j] * (m[i, j] - m_0[i, j]) / dt * volume + (1 - S[i, j]) *
                      m[i, j] * volume * (Wo[i, j] - Wo_0[i, j]) / dt)

        for i in ndrange(Nx):
            idx_bottom = i
            idx_top = (Nx - 1) * (Ny - 2) + i

            b[idx_bottom] = p[i, 1]
            b[idx_top] = p[i, Ny]
            A[idx_bottom, idx_bottom] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)
            A[idx_top, idx_top] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)

        for j in ndrange(Ny):
            idx_left = j * Nx
            idx_right = (j + 1) * Nx - 1

            b[idx_left] = p[1, j]
            b[idx_right] = p[Nx, j]
            A[idx_left, idx_right] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)
            A[idx_right, idx_right] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)

        # Добавили скважины в точки (0,0) (nx, ny)
        b[0] += Wo[0, 0] * qw * volume
        b[N-1] += qo * volume

    fill_matrix_and_rhs(mat)
    sparse_matrix = mat.build()

    solver = linalg.SparseSolver(solver_type="LLT", ordering="AMD")
    solver.analyze_pattern(sparse_matrix)
    solver.factorize(sparse_matrix)

    p_new = solver.solve(b.to_numpy())
    p.from_numpy(p_new.reshape((Nx + 2, Nx + 2)))

    return p


# @kernel
# def fill_matrix_and_rhs(A: types.sparse_matrix_builder()):
#     p1, p2, p3, p4 = 10.0, 10.0, 10.0, 10.0
#     for i, j in ndrange((1, Nx + 1), (1, Ny + 1)):
#         idx = j * (Nx + 2) + i
#
#         # i, j -> i-1, j-1
#         p1 = Wo[i, j - 1] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
#                                 k[i, j - 1], S[i, j - 1], mu_o[i, j - 1], mu_w[i, j - 1]) * area / hx
#         if i == 1:
#             p2 = Wo[i - 1, j - 1] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
#                                         k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1]) * area / hx
#         else:
#             p2 = Wo[i - 2, j - 1] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
#                                         k[i - 2, j - 1], S[i - 2, j - 1], mu_o[i - 2, j - 1],
#                                         mu_w[i - 2, j - 1]) * area / hx
#         p3 = Wo[i - 1, j - 2] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
#                                     k[i - 1, j], S[i - 1, j], mu_o[i - 1, j], mu_w[i - 1, j]) * area / hy
#         if j == 1:
#             p4 = Wo[i - 1, j] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
#                                     k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1]) * area / hy
#         else:
#             p4 = Wo[i - 1, j] * mid(k[i - 1, j - 1], S[i - 1, j - 1], mu_o[i - 1, j - 1], mu_w[i - 1, j - 1],
#                                     k[i - 1, j - 2], S[i - 1, j - 2], mu_o[i - 1, j - 2], mu_w[i - 1, j - 2]) * area / hy
#         A[idx, idx + 1] -= p1
#         A[idx, idx - 1] -= p2
#         A[idx, idx + Nx + 2] -= p3
#         A[idx, idx - Nx - 2] -= p4
#         A[idx, idx] += (p1 + p2 + p3 + p4)
#
#         # rhs
#         b[idx] = Wo[i, j] * (m[i, j] - m_0[i, j]) / dt * volume + (1 - S[i, j]) * m[i, j] * volume * (
#                     Wo[i, j] - Wo_0[i, j]) / dt
#         print(i, j, p1, p2, p3, p4)
#
#     for i in ndrange((Nx + 2)):
#         idx_bottom = i
#         idx_top = (Ny + 1) * (Ny + 2) + i
#         b[idx_bottom] = p[i, 1]
#         b[idx_top] = p[i, Ny]
#         A[idx_bottom, idx_bottom] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)
#         A[idx_top, idx_top] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)
#
#     for j in ndrange((Ny + 2)):
#         idx_left = j * (Ny + 2)
#         idx_right = j * (Ny + 2) + (Nx + 2)
#         b[idx_left] = p[1, j]
#         b[idx_right] = p[Nx, j]
#         A[idx_left, idx_right] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)
#         A[idx_right, idx_right] += 1.0  # For boundary points, set diagonal to 1 (Neumann BC)
#
#     # Добавили скважины в точки (1,1) (nx+1, ny+1)
#     b[Ny + 2] += Wo[0, 0] * qw * volume
#     b[(Nx + 1) * (Ny + 2) + (Ny + 1)] += qo * volume