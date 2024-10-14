from imp import new_module

import pickle
from numpy import dot, zeros_like, pi
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, spilu, LinearOperator, bicgstab, gmres
from taichi import f32, f64, i32, field, types, ndrange, kernel, static, func, init, vulkan, cpu, sin, cos

init(arch=cpu)


Nx, Ny = 110, 110
X_min, X_max = 0., 1.
Y_min, Y_max = 0., 1.
hx = (X_max - X_min) / (Nx - 1)
hy = (Y_max - Y_min) / (Ny - 1)
area = hx * hy

def calc_pressure():
    """
    Сборка матрицы и решение СЛАУ уравнения диффузии
    """
    N = Nx * Ny  # размер матрицы
    NN = (Nx - 2) * (Ny - 2) * 5 + (Nx-1) * 2 + (Ny-1) * 2  # количество ненулевых элементов
    data = field(f32, shape=NN)
    row_indices = field(i32, shape=NN)
    col_indices = field(i32, shape=NN)
    b = field(f32, shape=N)

    @kernel
    def fill_matrix_and_rhs():
        num = 0
        for i in ndrange(Nx):
            for j in ndrange(Ny):
                idx = i + j * Nx
                x = i * hx

                if (i == 0) | (i == Nx-1) | (j == 0) | (j == Ny-1):
                    row_indices[num] = idx
                    col_indices[num] = idx
                    data[num] = 1.0
                    num += 1
                    # rhs
                    b[idx] = sin(2 * pi * x) + 0.5 * sin(10 * pi * x)
                else:
                    p_sum = 0.0
                    # matrix
                    arr = [[i + 1, j, hx], [i - 1, j, hx], [i, j + 1, hy], [i, j - 1, hy]]
                    for qq in static(ndrange(4)):
                        i1, j1, hij = arr[qq[0]]
                        p = 1.0 / hij / hij
                        row_indices[num] = idx
                        col_indices[num] = idx + (i1-i) + Nx * (j1-j)
                        data[num] = -p
                        p_sum += p
                        num += 1
                    row_indices[num] = idx
                    col_indices[num] = idx
                    data[num] = p_sum
                    num += 1
                    # rhs
                    b[idx] = 4.0 * pi * pi * sin(2 * pi * x) + 50 * pi * pi * sin(10 * pi * x)

        # # Добавили скважины в точки (0,0) (nx, ny)
        # b[0] += Wo[0, 0] * qw * volume * 0.25
        # b[N-1] += qo * volume * 0.25

    fill_matrix_and_rhs()
    A_csr = csr_matrix((data.to_numpy(), (row_indices.to_numpy(), col_indices.to_numpy())), shape=(N, N))

    # print(A_csr.toarray())

    # x = spsolve(A_csr, b.to_numpy())
    # x = gmres(A_csr, b.to_numpy())[0]
    x = bicgstab(A_csr, b.to_numpy())[0]
    # x = my_bicgstab(A_csr, b.to_numpy())

    p = x.reshape((Nx, Ny))

    show_plot(p)
    # plotly_graph(p)
    return  p


def show_plot(data):
    import matplotlib.pyplot as plt
    # Отображаем массив с помощью imshow
    plt.imshow(data, cmap='viridis')

    # Добавляем цветовую шкалу с дополнительными параметрами
    cbar = plt.colorbar(orientation='horizontal', shrink=0.8)
    cbar.set_label('Значения данных')

    # Отображаем график
    plt.show()


def plotly_graph(data):
    from numpy import linspace
    import plotly.graph_objects as go

    x = linspace(X_min, X_max, Nx)
    y = linspace(Y_min, Y_max, Ny)

    # Создаем тепловую карту
    fig = go.Figure(data=go.Heatmap(
        x=x,
        y=y,
        z=data,
        colorscale='Viridis'
    ))

    # Настраиваем отображение графика
    fig.update_layout(
        title='Двумерное поле данных',
        xaxis_title='X',
        yaxis_title='Y',
        width=800,
        height=600
    )

    # Отображаем график
    fig.show()


def my_bicgstab(A, b, x0=None, tol=1e-5, max_iter=1000):
    """
    Реализация метода бисопряженных градиентов со стабилизацией (BICGSTAB) для решения системы Ax = b.

    Параметры:
    A: scipy.sparse.csr_matrix - разреженная матрица системы.
    b: numpy.ndarray - вектор правой части.
    x0: numpy.ndarray, опционально - начальное приближение решения.
    tol: float, опционально - допустимая погрешность.
    max_iter: int, опционально - максимальное количество итераций.

    Возвращает:
    x: numpy.ndarray - найденное приближение решения.
    """

    if x0 is None:
        x = zeros_like(b)
    else:
        x = x0.copy()

    r = b - A.dot(x)
    r_hat = r.copy()
    rho, alpha, omega = 1.0, 1.0, 1.0
    v = zeros_like(b)
    p = zeros_like(b)

    for i in range(max_iter):
        rho_prev = rho
        rho = dot(r_hat, r)
        beta = (rho / rho_prev) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = A.dot(p)
        alpha = rho / dot(r_hat, v)
        s = r - alpha * v
        t = A.dot(s)
        omega = dot(t, s) / dot(t, t)
        x = x + alpha * p + omega * s
        r = s - omega * t
        err = norm(r)/norm(b)

        if err < tol:
            print(f"BICGSTAB converged in {i+1} iterations.")
            return x

    print("BICGSTAB did not converge within the maximum number of iterations.")
    return x


if __name__ == '__main__':
    calc_pressure()
