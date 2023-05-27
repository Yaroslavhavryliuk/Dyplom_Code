# Алгоритм розв'язку рівняння Лотки-Вольтерра методом Рунге-Кутти
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import tensorflow as tf

geom = dde.geometry.TimeDomain(0.0, 24.0) # задання часового проміжку

# Рівняння Лотки-Вольтерра
def func(t, r):
    x, y = r
    dx_t = 1.1 * x - 0.4 * x * y
    dy_t = 0.1 * x * y - 0.4 * y
    return dx_t, dy_t

# Генерація правильних даних для перевірки методом Рунге-Кутти
def gen_truedata():
    t = np.linspace(0, 24, 100)

    sol = integrate.solve_ivp(func, (0, 240), (20, 5), t_eval=t)
    x_true, y_true = sol.y
    x_true = x_true.reshape(100, 1)
    y_true = y_true.reshape(100, 1)

    return x_true, y_true

# Задання системи ODE
def ode_system(x, y):
    r = y[:, 0:1]
    p = y[:, 1:2]
    dr_t = dde.grad.jacobian(y, x, i=0)
    dp_t = dde.grad.jacobian(y, x, i=1)
    return [
        dr_t - 1.1 * r + 0.4 * r * p,
        dp_t - 0.1 * r * p + 0.4 * p,
    ]

# Початкова умова хижаків
def func1(x):
    return 20

# Початкова умова жертв
def func2(x):
    return 5

# Функція перевірки умов
def boundary(x, on_initial):
    return on_initial

# Задання початкових умов
bc1 = dde.icbc.IC(geom, func1, boundary, component = 0)
bc2 = dde.icbc.IC(geom, func2, boundary, component = 1)

data = dde.data.PDE(geom, ode_system,[bc1,bc2], 3000, 2, num_test = 3000) # Формування задачі

# Нейронна мережа
layer_size = [1] + [64] * 6 + [2]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

# Для задання періодичності
def input_transform(t):
    return tf.concat(
        (
            t,
            tf.sin(t),
            tf.sin(2 * t),
            tf.sin(3 * t),
            tf.sin(4 * t),
            tf.sin(5 * t),
            tf.sin(6 * t),
        ),
        axis=1,
    )


net.apply_feature_transform(input_transform)

# Об'єднання даних і нейромережі у модель, компіляція, тренування
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=50000)

# Дотреновування з іншим оптимізатором
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True) # результати нейромережі

# Оцінка точності
plt.xlabel("t")
plt.ylabel("population")

t = np.linspace(0, 24, 100)
x_true, y_true = gen_truedata()
plt.plot(t, x_true, color="black", label="x_true")
plt.plot(t, y_true, color="blue", label="y_true")

t = t.reshape(100, 1)
sol_pred = model.predict(t)
x_pred = sol_pred[:, 0:1]
y_pred = sol_pred[:, 1:2]

plt.plot(t, x_pred, color="red", linestyle="dashed", label="x_pred")
plt.plot(t, y_pred, color="orange", linestyle="dashed", label="y_pred")
plt.legend()
plt.show()