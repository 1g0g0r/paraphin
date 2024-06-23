import taichi as ti
import numpy as np

# Инициализация Taichi
ti.init(arch=ti.gpu)

# Размер поля
n = 512

# Создание нескольких двумерных полей данных
field1 = ti.field(dtype=ti.f32, shape=(n, n))
field2 = ti.field(dtype=ti.f32, shape=(n, n))
field3 = ti.field(dtype=ti.f32, shape=(n, n))

# Инициализация полей данными
@ti.kernel
def initialize_fields():
    for i, j in field1:
        field1[i, j] = ti.sin(i * 0.1) * ti.cos(j * 0.1)
        field2[i, j] = ti.sin(i * 0.2) * ti.cos(j * 0.2)
        field3[i, j] = ti.sin(i * 0.3) * ti.cos(j * 0.3)

initialize_fields()

# Создание GUI
gui = ti.GUI("Taichi Fields", res=(n, n))

okay = gui.button('pres')
radius = gui.slider('Radius', 1, 50, step=1)
xcoor = gui.label('Press 1, 2, or 3 to switch fields')

# Индекс текущего поля
current_field_index = 0
fields = [field1, field2, field3]

# Основной цикл
while gui.running:
    # Обработка событий GUI
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == '1':
            current_field_index = 0
        elif e.key == '2':
            current_field_index = 1
        elif e.key == '3':
            current_field_index = 2
        elif e.key == okay:
            current_field_index = 0

    # Отображение текущего поля
    gui.set_image(fields[current_field_index])
    gui.show()