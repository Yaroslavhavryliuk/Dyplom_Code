# Тестування оцінених параметрів
import numpy as np
import matplotlib.pyplot as plt

# Задання вигляду рівняння
def Lotka_Volter(x, params):
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']

    equation = np.array([alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]])

    return equation

# Алгоритм Рунге-Кутти
def Runge_Kutta(f, x0, t0, tn, h):
    t = np.arange(t0, tn, h)
    nt = t.size

    nx = x0.size
    x = np.zeros((nx, nt))

    x[:, 0] = x0

    for k in range(nt - 1):
        k1 = h * f(t[k], x[:, k]);
        k2 = h * f(t[k] + h / 2, x[:, k] + k1 / 2)
        k3 = h * f(t[k] + h / 2, x[:, k] + k2 / 2)
        k4 = h * f(t[k] + h, x[:, k] + k3)

        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[:, k + 1] = x[:, k] + dx;

    return x, t

params = {"alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1} # параметри початкового рівняння
params_pred = {"alpha": 1.1, "beta": 0.399, "gamma": 0.401, "delta": 0.0995} # Параметри, оцінені нейромережею

func= lambda t,x : Lotka_Volter(x, params)
func_pred= lambda t,x : Lotka_Volter(x, params_pred)
x0 = np.array([20, 5])  # початкові умови


# часовий проміжок та крок
t0 = 0
tn = 24
h = 0.25

# розв'язання оригінального рівняння та рівняння з оціненими параметрами
x, t = Runge_Kutta(func, x0, t0, tn, h)
x_pred,t_pred = Runge_Kutta(func_pred, x0, t0, tn, h)


# Перевірка якості оцінки параметрів
plt.subplot(1, 1, 1)
plt.plot(t, x[0,:], "g", label="Preys")
plt.plot(t, x[1,:], "r", label="Predators")
plt.plot(t, x_pred[0,:], "b", label="Preys predicted")
plt.plot(t, x_pred[1,:], "y", label="Predators predicted")
plt.xlabel("Time (t)")
plt.grid()
plt.legend()


plt.show()