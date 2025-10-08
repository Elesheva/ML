import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#№1 моментум
def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)

def g(x):
    return -0.5 + 0.4 * x - 0.03 * x ** 2 - 1.2 * np.cos(4 * x)

eta = 0.1
N = 200
x = -3.5
gamma = 0.8
v = 0

for i in range(N):
    v = (gamma * v) + ((1 - gamma) * eta * g(x))
    x = x - v


#№2 Нестеров
def func(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)
def g(x):
    return 0.4 + 0.2 * np.cos(2*x) - (0.6 * np.sin(3*x))

eta = 1.0
N = 500
x = 4.0
gamma = 0.7
v = 0

for i in range(N):
    v = (gamma * v) + (1 - gamma) * eta * g(x - (gamma * v))
    x = x - v


#№3 RMSProp
def func(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3*x)

def g(x):
    return (2 + 0.3 * x ** 2) - (6 * np.sin(3*x))

eta = 0.5
x = 4
N = 200
a = 0.8
G = 0
epsilon = 0.01

for i in range(N):
    G = a * G + ((1 - a) * (g(x) ** 2))
    x = x - eta * (g(x)/ (G ** 0.5 + epsilon))
    


#№4
def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2

def Q(w, X, y):
    return np.mean(np.square(X @ w - y))

def dQdw(w, X, y):
    return X.T @ (X @ w - y) * (2 / X.shape[0])

coord_x = np.arange(-4.0, 6.0, 0.1)  # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x)  # значения функции по оси ординат

sz = len(coord_x)  # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.])  # начальные значения параметров модели
N = 500  # число итераций алгоритма SGD
lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20  # размер мини-батча (величина K = 20)
gamma = 0.8  # коэффициент гамма для вычисления импульсов Нестерова
v = np.zeros(len(w))  # начальное значение [0, 0, 0, 0]

# создание матрицы X и вектора y
X = np.array([[1, x, x ** 2, x ** 3] for x in coord_x])
y = np.array(coord_y)

# сохраняем историю весов
weights = []
losses = []
# начальное значение Qe
Qe = Q(w, X, y)
np.random.seed(0)  # фиксация случайного генератора

for _ in range(N):
    k = np.random.randint(0, sz - batch_size - 1)
    batch_interval = np.arange(k, k + batch_size)
    Xk = X[batch_interval]
    yk = y[batch_interval]

    # пересчет Qe
    Qe = lm * Q(w, Xk, yk) + (1 - lm) * Qe

    # пересчет импульса и весов
    v = gamma * v + (1 - gamma) * eta * dQdw(w - gamma * v, Xk, yk)
    w -= v
    weights.append(w.copy())
    losses.append(Qe)

# АНИМАЦИЯ
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(coord_x, coord_y, label="Исходная функция", color="blue")
line, = ax.plot([], [], color="red", label="Аппроксимация модели")
ax.legend()
ax.set_ylim(min(coord_y)-1, max(coord_y)+1)

def animate(i):
    w = weights[i]
    y_pred = X @ w
    line.set_data(coord_x, y_pred)
    ax.set_title(f"Итерация {i+1}/{N}, Qe = {losses[i]:.4f}")
    return line,

ani = FuncAnimation(fig, animate, frames=len(weights), interval=100)
plt.show()
# финальный расчет Q для всей выборки
Q = Q(w, X, y)



#№5
# логарифмическая функция потерь
def loss(w, x, y):
    M = np.dot(w, x) * y
    return np.log2(1 + np.exp(-M))

# производная логарифмической функции потерь по вектору w
def df(w, x, y):
    M = np.dot(w, x) * y
    return -(np.exp(-M) * x.T * y) / ((1 + np.exp(-M)) * np.log(2))

data_x = [(5.3, 2.3), (5.7, 2.5), (4.0, 1.0), (5.6, 2.4), (4.5, 1.5), (5.4, 2.3), (4.8, 1.8), (4.5, 1.5), (5.1, 1.5), (6.1, 2.3), (5.1, 1.9), (4.0, 1.2), (5.2, 2.0), (3.9, 1.4), (4.2, 1.2), (4.7, 1.5), (4.8, 1.8), (3.6, 1.3), (4.6, 1.4), (4.5, 1.7), (3.0, 1.1), (4.3, 1.3), (4.5, 1.3), (5.5, 2.1), (3.5, 1.0), (5.6, 2.2), (4.2, 1.5), (5.8, 1.8), (5.5, 1.8), (5.7, 2.3), (6.4, 2.0), (5.0, 1.7), (6.7, 2.0), (4.0, 1.3), (4.4, 1.4), (4.5, 1.5), (5.6, 2.4), (5.8, 1.6), (4.6, 1.3), (4.1, 1.3), (5.1, 2.3), (5.2, 2.3), (5.6, 1.4), (5.1, 1.8), (4.9, 1.5), (6.7, 2.2), (4.4, 1.3), (3.9, 1.1), (6.3, 1.8), (6.0, 1.8), (4.5, 1.6), (6.6, 2.1), (4.1, 1.3), (4.5, 1.5), (6.1, 2.5), (4.1, 1.0), (4.4, 1.2), (5.4, 2.1), (5.0, 1.5), (5.0, 2.0), (4.9, 1.5), (5.9, 2.1), (4.3, 1.3), (4.0, 1.3), (4.9, 2.0), (4.9, 1.8), (4.0, 1.3), (5.5, 1.8), (3.7, 1.0), (6.9, 2.3), (5.7, 2.1), (5.3, 1.9), (4.4, 1.4), (5.6, 1.8), (3.3, 1.0), (4.8, 1.8), (6.0, 2.5), (5.9, 2.3), (4.9, 1.8), (3.3, 1.0), (3.9, 1.2), (5.6, 2.1), (5.8, 2.2), (3.8, 1.1), (3.5, 1.0), (4.5, 1.5), (5.1, 1.9), (4.7, 1.4), (5.1, 1.6), (5.1, 2.0), (4.8, 1.4), (5.0, 1.9), (5.1, 2.4), (4.6, 1.5), (6.1, 1.9), (4.7, 1.6), (4.7, 1.4), (4.7, 1.2), (4.2, 1.3), (4.2, 1.3)]

data_y = [1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1]

x_train = np.array([[1, x[0], x[1]] for x in data_x])
y_train = np.array(data_y)

n_train = len(x_train)  # размер обучающей выборки
w = [0.0, 0.0, 0.0]  # начальные весовые коэффициенты
nt = np.array([0.1, 0.05, 0.05])  # шаг обучения для каждого параметра w0, w1, w2
lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
N = 200  # число итераций алгоритма SGD
batch_size = 10  # размер мини-батча (величина K = 10)

alpha = 0.7  # параметр для RMSProp
G = np.zeros(len(w))  # параметр для RMSProp
eps = 0.01  # параметр для RMSProp

Qe = np.mean(loss(w, x_train.T, y_train))  # начальное значение среднего эмпирического риска
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

for i in range(N):
    k = np.random.randint(0, n_train - batch_size - 1)  # n_train - размер выборки (массива x_train)
    batch_interval = range(k, k + batch_size)
    x_train_k = x_train[batch_interval]
    y_train_k = y_train[batch_interval]
    Qk = np.mean(loss(w, x_train_k.T, y_train_k))
    dQkdw = np.mean([df(w, x, y) for x, y in zip(x_train_k, y_train_k)], axis=0)
    G = alpha * G + (1 - alpha) * dQkdw * dQkdw
    w -= nt * dQkdw / (np.sqrt(G) + eps)
    Qe = lm * Qk + (1 - lm) * Qe

Q = np.mean(x_train @ w * y_train < 0)
























































