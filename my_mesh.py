import pyvista as pv
from meshpy.tet import MeshInfo, build
from my_consatnts import *


def create_mesh():
    # Создание регулярной сетки
    x = np.linspace(X_min, X_max, Nx)
    y = np.linspace(Y_min, Y_max, Ny)
    xx, yy, zz = np.meshgrid(x, y, 1)

    # Преобразование координат сетки в массив вершин
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Создание сетки с использованием pyvista
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (Nx, Ny, 1)

    # Визуализация сетки
    # plotter = pv.Plotter()
    # plotter.add_mesh(grid, show_edges=True)
    # plotter.show()

    return grid.points, grid.cell


def new_mesh_creation():
    mesh_info = MeshInfo()
    mesh_info.set_points([
        (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
        (0, 0, 12), (2, 0, 12), (2, 2, 12), (0, 2, 12),
    ])
    mesh_info.set_facets([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ])
    mesh = build(mesh_info)
    mesh.write_vtk("mesh.vtk")

    return mesh.points, mesh.elements
