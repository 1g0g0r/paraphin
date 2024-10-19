import plotly.graph_objects as go
import numpy as np
#
# # Пример данных для двумерных полей
# def generate_data(n, m=1,noise=0.1):
#     x = np.linspace(-2, 2, n)
#     y = np.linspace(-2, 2, n)
#     X, Y = np.meshgrid(x, y)
#     Z = m*np.sin(X**2 + Y**2) + noise * np.random.randn(n, n)
#     return X, Y, Z
#
# # Генерация данных
# n = 50
# X1, Y1, Z1 = generate_data(n, noise=0.001)
# X2, Y2, Z2 = generate_data(n, noise=0.002)
# X3, Y3, Z3 = generate_data(n, noise=0.003)
#
# # Создаем графики
# trace1 = go.Heatmap(z=Z1, x=X1[0], y=Y1[:, 0], colorscale='Viridis')  # , name='Поле 1'
# trace2 = go.Heatmap(z=Z2, x=X2[0], y=Y2[:, 0], colorscale='Viridis')  # , name='Поле 2'
# trace3 = go.Heatmap(z=Z3, x=X3[0], y=Y3[:, 0], colorscale='Viridis')  # , name='Поле 3'
#
# # Создаем фигуру
# fig = go.Figure(data=[trace1, trace2, trace3])
#
# # Добавляем слайдеры для изменения данных
# steps = []
# for i in range(n):
#     step = dict(
#         method="update",
#         args=[{"z": [generate_data(n, 1), generate_data(n, 2), generate_data(n, 3)]}],
#         label=str(i)
#     )
#     steps.append(step)
#
# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Время: "},
#     pad={"t": 50},
#     steps=steps
# )]
#
# # Добавляем кнопки для выбора разных наборов данных
# fig.update_layout(
#     updatemenus=[
#         dict(
#             type="buttons",
#             direction="left",
#             buttons=list([
#                 dict(
#                     args=[{"visible": [True, False, False]}],
#                     label="Pressure",
#                     method="update"
#                 ),
#                 dict(
#                     args=[{"visible": [False, True, False]}],
#                     label="Saturation",
#                     method="update"
#                 ),
#                 dict(
#                     args=[{"visible": [False, False, True]}],
#                     label="Temperature",
#                     method="update"
#                 )
#             ]),
#             pad={"r": 10, "t": 10},
#             showactive=True,
#             x=0.1,
#             xanchor="left",
#             y=1.1,
#             yanchor="top"
#         ),
#     ],
#     sliders=sliders,
#     width = 1000,  # Устанавливаем ширину фигуры
#     height = 800  # Устанавливаем высоту фигуры
# )
#
# # По умолчанию показываем только первое поле
# fig.update_traces(visible=False)
# fig.data[0].visible = True
#
# # Отображаем график
# fig.write_html('results.html', include_plotlyjs='plotly_script.js')


def _vector_field_with_arrows(x, y, z):
    grad = np.array(np.gradient(z)) * 0.7
    # fig = go.Figure(data=go.Heatmap(z=z, colorscale='Viridis'))

    # Создаем списки для хранения данных
    arrows_x = []
    arrows_y = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            arrows_x.append(x[i, j])
            arrows_y.append(y[i, j])
            arrows_x.append(x[i, j] + grad[0][i, j])
            arrows_y.append(y[i, j] + grad[1][i, j])
            arrows_x.append(None)  # Для разрыва между векторами
            arrows_y.append(None)

    # Создаем график
    fig = go.Figure()

    # Добавляем тепловую карту
    fig.add_trace(go.Heatmap(
        x=np.unique(x),
        y=np.unique(y),
        z=z,
        colorscale='Viridis',
        opacity=0.5,
        showscale=True
    ))

    # Добавляем векторы на график
    fig.add_trace(go.Scatter(
        x=arrows_x,
        y=arrows_y,
        mode='lines',
        line=dict(color='blue', width=1),
        hoverinfo='none'
    ))

    # Настройка макета
    fig.update_layout(
        title='Векторное поле',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        showlegend=False,
        width=800,
        height=800,
    )

    # Отображение графика
    fig.write_html('results.html', include_plotlyjs='plotly_script.js')


if __name__ == '__main__':
    n = 100
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X**2 + Y**2)

    _vector_field_with_arrows(x, y, Z)
