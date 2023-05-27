# Алгоритм оцінки параметрів рівняння Лотки-Вольтерра
import deepxde as dde
import numpy as np

# Вичитка точок з файлу
def gen_traindata():
    data_t = []
    data_y = []
    file = open('Lotka-Voltera2years.txt', 'r')
    for line in file:
        values = line.split(';')
        data_tItem = []
        data_yItem = []
        data_tItem.append(float(values[0]))
        data_yItem.append(float(values[1]))
        data_yItem.append(float(values[2]))
        data_t.append(data_tItem)
        data_y.append(data_yItem)
    data_t = np.array(data_t)
    data_y = np.array(data_y)
    print(type(data_t))
    print(data_t.shape)
    print(data_t)
    print(type(data_y))
    print(data_y.shape)
    print(data_y)
    return data_t, data_y


# Невідомі параметри
C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
C3 = dde.Variable(1.0)
C4 = dde.Variable(1.0)

# Рівняння Лотки-Вольтерра
def LV_system(x, y):
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    return [
        dy1_x - C1 * y1 + C2 * y1 * y2,
        dy2_x - C4 * y1 * y2 + C3 * y2,
    ]

# Функція перевірки умов
def boundary(_, on_initial):
    return on_initial

# Часовий інтервал
geom = dde.geometry.TimeDomain(0, 24)

# Початкові умови
ic1 = dde.icbc.IC(geom, lambda X: 20, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda X: 5, boundary, component=1)

# Формування даних для тренування
observe_t, ob_y = gen_traindata()
observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)

# Задання проблеми
data = dde.data.PDE(
    geom,
    LV_system,
    [ic1, ic2, observe_y0, observe_y1],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
    num_test=400
)

# Нейронна мережа та модель
net = dde.nn.FNN([1] + [64] * 6 + [2], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Цільові параметри
external_trainable_variables = [C1, C2, C3, C4]
variable = dde.callbacks.VariableValue(
    external_trainable_variables, period=600, filename="variablesLotka2years.dat" # файл з історією оптимізації параметрів
)

# Тренування моделі
model.compile(
    "adam", lr=0.001, external_trainable_variables=external_trainable_variables
)
losshistory, train_state = model.train(iterations=20000, callbacks=[variable])

# Дотреновування з іншим оптимізатором
model.compile("L-BFGS", external_trainable_variables=external_trainable_variables)
losshistory, train_state = model.train(callbacks=[variable])

dde.saveplot(losshistory, train_state, issave=True, isplot=True) # Графік