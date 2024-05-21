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
        if ((i+1) % Nx == 0) or (i % Nx == 0) or (i < Nx) or (i > Nx * (Ny - 1)):
            continue

        KwKo_i = K_o(i) + K_w(i)
        A[i, i + 1] = - Wo[i] * mid(KwKo_i, K_o(i+1) + K_w(i+1)) * area / hx
        A[i, i - 1] = -Wo[i] * mid(KwKo_i, K_o(i-1) + K_w(i-1)) * area / hx
        A[i, i + Nx - 1] = -Wo[i] * mid(KwKo_i, K_o(i+Nx-1) + K_w(i+Nx-1)) * area / hy
        A[i, i - Nx + 1] = -Wo[i] * mid(KwKo_i, K_o(i-Nx+1) + K_w(i-Nx+1)) * area / hy
        A[i, i] = - A[i, i + 1] - A[i, i - 1] - A[i, i + Nx - 1] - A[i, i - Nx + 1]

        # Create a vector
        b[i] =

    # Solve the system of linear equations
    x = np.linalg.solve(A.A, b)
    print(x)

