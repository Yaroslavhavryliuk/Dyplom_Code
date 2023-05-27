# Алгоритм розв'язку рівняння Лотки-Вольтерра методом Рунге-Кутти
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
    file = open("Lotka-Voltera2years.txt", "w") # Файл з точками для оцінки параметрів

    for k in range(nt - 1):
        file.write(str(t[k]) + ';' + str(x[0, k]) + ';' + str(x[1, k]) + '\n') # запис точки у файл
        k1 = h * f(t[k], x[:, k]);
        k2 = h * f(t[k] + h / 2, x[:, k] + k1 / 2)
        k3 = h * f(t[k] + h / 2, x[:, k] + k2 / 2)
        k4 = h * f(t[k] + h, x[:, k] + k3)

        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[:, k + 1] = x[:, k] + dx;

    file.close()
    return x, t

params = {"alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1} # параметри рівняння

func= lambda t,x : Lotka_Volter(x, params)
x0 = np.array([20, 5])                 # початкові умови


# часовий проміжок та крок
t0 = 0
tn = 26
h = 1

x, t = Runge_Kutta(func, x0, t0, tn, h) # розв'язання рівняння

# Вивід результатів

plt.subplot(1, 1, 1)
plt.plot(t, x[0,:], "g", label="Preys")
plt.plot(t, x[1,:], "r", label="Predators")
plt.xlabel("Time (t)")
plt.grid()
plt.legend()


plt.show()