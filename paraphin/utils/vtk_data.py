import vtk
import numpy as np

# Параметры сетки
nx, ny = 10, 10  # Количество точек по каждой оси
xmin, xmax = 0, 1
ymin, ymax = 0, 1

# Создание регулярной сетки
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

# Создание пустого объекта vtkStructuredPoints
structured_points = vtk.vtkStructuredPoints()
structured_points.SetDimensions(nx, ny, 1)
structured_points.SetOrigin(xmin, ymin, 0)
structured_points.SetSpacing((xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1), 1)

# Создание данных для записи (например, два скалярных поля)
data1 = np.random.rand(nx, ny)
data2 = np.random.rand(nx, ny)

# Преобразование первого скалярного поля в формат VTK
vtk_data_array1 = vtk.vtkDoubleArray()
vtk_data_array1.SetName("ScalarField1")
vtk_data_array1.SetNumberOfComponents(1)
vtk_data_array1.SetNumberOfTuples(nx * ny)

for i in range(nx):
    for j in range(ny):
        vtk_data_array1.SetTuple1(i + j * nx, data1[i, j])

# Преобразование второго скалярного поля в формат VTK
vtk_data_array2 = vtk.vtkDoubleArray()
vtk_data_array2.SetName("ScalarField2")
vtk_data_array2.SetNumberOfComponents(1)
vtk_data_array2.SetNumberOfTuples(nx * ny)

for i in range(nx):
    for j in range(ny):
        vtk_data_array2.SetTuple1(i + j * nx, data2[i, j])

# Добавление данных в объект vtkStructuredPoints
structured_points.GetPointData().AddArray(vtk_data_array1)
structured_points.GetPointData().AddArray(vtk_data_array2)

# Запись данных в файл VTK
writer = vtk.vtkStructuredPointsWriter()
writer.SetFileName("output.vtk")
writer.SetInputData(structured_points)
writer.Write()