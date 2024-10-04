import taichi as ti
import numpy as np

# Инициализация Taichi
ti.init(arch=ti.cpu)

# Определение Taichi-ядра
@ti.kernel
def taichi_kernel(data: ti.types.ndarray()):
    for i, j in ti.ndrange(data.shape[0], data.shape[1]):
        data[i, j] += 1  # Увеличение каждого элемента на 1

# Создание двумерного numpy массива
data = np.array([[1, 2], [3, 4]], dtype=np.int32)

# Вызов Taichi-ядра с передачей numpy массива
taichi_kernel(data)

# Вывод результата
print(data)  # Вывод: [[2 3] [4 5]]