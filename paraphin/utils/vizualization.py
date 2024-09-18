from pickle import load

from numpy import min, max, zeros, ones_like
from taichi import GUI, tools

from paraphin.utils.constants import output_file_name, Time_end, sol_time_step

x_pixels = 600
y_pixels = 600


def visualize_solution() -> None:
    with open(output_file_name, 'rb') as f:
        pres, sat, temp = load(f)

    # Визуализируем решение
    gui = GUI("Поля данных", res=(x_pixels, y_pixels))
    time_slider = gui.slider("Время", 0, Time_end)

    # Максимальные/минимальные значения полей данных
    min_pres = min([min(q) for q in pres])
    max_pres = max([max(q) for q in pres])
    min_sat = min([min(q) for q in sat])
    max_sat = max([max(q) for q in sat])
    min_temp = min([min(q) for q in temp])
    max_temp = max([max(q) for q in temp])

    # Начальная инициализация GUI
    data = pres[0]
    min_val = min_pres
    max_val = max_pres

    while gui.running:
        # Получаем индекс времени
        time_index = int(time_slider.value / sol_time_step)

        # Выбор поля данных
        for e in gui.get_events(GUI.PRESS):
            if e.key == GUI.SPACE:
                gui.running = False
            elif e.key == 'p':
                data = pres[time_index]
                min_val = min_pres
                max_val = max_pres
            elif e.key == 's':
                data = sat[time_index]
                min_val = min_sat
                max_val = max_sat
            elif e.key == 't':
                data = temp[time_index]
                min_val = min_temp
                max_val = max_temp

        # Нормализуем данные для преобразования в цвет
        normalized_data = (data - min_val) / (max_val - min_val)  # Нормализация
        color_data = zeros((len(data), len(data), 3))  # Создаем массив для цвета

        # Преобразуем нормализованные данные в цвет (RGB) # Пример: градиент от красного к синему
        color_data[:, :, 0] = normalized_data
        color_data[:, :, 1] = ones_like(normalized_data) * 0.5
        color_data[:, :, 2] = 1.0 - normalized_data

        # Масштабируем изображение под размер GUI
        resized_image = tools.imresize(color_data, x_pixels, y_pixels)

        # Устанавливаем цветное изображение
        gui.set_image(resized_image)

        # Отображаем GUI
        gui.show()
