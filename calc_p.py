from my_consatnts import *
from utils import mid, K_o, K_w
from scipy.sparse import dok_matrix


def calc_p(cells):
    """
    Сборка матрицы и решение СЛАУ уравнения давления (МКЭ)
    """
    # Создаем разреженную матрицу
    A = dok_matrix((N_elements, N_elements))
    b = np.zeros(N_elements)

    for i, elem in enumerate(cells):
        if (i < Nx-1) or (i % (Nx-1) == 0) or ((i+1) % (Nx-1) == 0) or (i > (Ny-2) * (Nx-1)):
            A[i, i] = 1
            p[i] = p_bound

        else:
            # TODO попробовать переписать векторно
            KwKo_i = K_o(i) + K_w(i)
            A[i, i + 1] = - Wo[i] * mid(KwKo_i, K_o(i + 1) + K_w(i + 1)) * area / hx
            A[i, i - 1] = -Wo[i] * mid(KwKo_i, K_o(i - 1) + K_w(i - 1)) * area / hx
            A[i, i + Nx - 1] = -Wo[i] * mid(KwKo_i, K_o(i + Nx - 1) + K_w(i + Nx - 1)) * area / hy
            A[i, i - Nx + 1] = -Wo[i] * mid(KwKo_i, K_o(i - Nx + 1) + K_w(i - Nx + 1)) * area / hy
            A[i, i] = - A[i, i + 1] - A[i, i - 1] - A[i, i + Nx - 1] - A[i, i - Nx + 1]

            # Create a rhs
            b[i] = Wo[i] * (m[i] - m_0[i]) / dt * volume[i] + (1 - S[i]) * m[i] * volume[i] * (Wo[i] - Wo_0[i]) / dt

    # Добавили скважины
    b[0] += Wo[0] * qw * volume[0]
    b[-1] -= qo * volume[-1]

    # Solve the system of linear equations
    p[:] = np.linalg.solve(A.A, b)
