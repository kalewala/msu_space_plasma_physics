"""Численная модель кулоновского взимодействия частиц
Метод частица-частица
=====================
Python 3.10.7"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


t = 0  # начальное состояние - нулевое время
T = 0.05  # конечное время
N = 1000  # количество шагов интегрирования

# частицы
r1 = [0.0001, 0.0001, 0.0001]  # начальные координаты в м
r2 = [0.0004, 0.0003, 0.0002]
r1 = np.array(r1, dtype="float64")  # перевод в np.array
r2 = np.array(r2, dtype="float64")

v1 = [0.002, 0.001, 0.003]  # начальная скорость в м/с
v2 = [-0.002, -0.005, -0.001]
v1 = np.array(v1, dtype="float64")  # перевод в np.array
v2 = np.array(v2, dtype="float64")

m1 = 9.109_383_7015e-31  # масса электрона в кг
m2 = 9.109_383_7015e-31

q1 = -1.602_176_634e-19  # заряд электрона в Кл
q2 = -1.602_176_634e-19

# контейнер
X = 0.0005  # длины стенок по координатам в м
Y = 0.0005
Z = 0.0005
Q = -1.602_176_634e-19  # заряд стенки в Кл

c = 299_792_458  # скорость света по СИ (м/с)
m0 = 4 * np.pi * 10**(-7)  # магнитная постоянная (Гн/м)
e0 = 1/(m0 * c**2)  # электричекая постоянная
k = 1/(4 * np.pi * e0)  # диэлектрическая проницаемость среды

def Equations(params, t, k, q1, q2):
    r1 = params[0:3]  # координаты частицы
    r2 = params[3:6]
    v1 = params[6:9]  # вектор скорости частицы
    v2 = params[9:12]

    r12 = np.linalg.norm(r2 - r1)  # расстояние между частицами
    dv1 = k*q2*(r2 - r1) / r12**3  # вектор приращения скорости
    dv2 = k*q1*(r1 - r2) / r12**3

    r1x0 = r1[0]  # расстояние до стенки X
    r1x1 = X - r1[0]
    r2x0 = r2[0]
    r2x1 = X - r2[0]
    dv1x = k*Q*(0 - r1[0]) / r1x0**3 + k*Q*(X - r1[0]) / r1x1**3  # приращениe скорости от стенки X
    dv2x = k*Q*(0 - r2[0]) / r2x0**3 + k*Q*(X - r2[0]) / r2x1**3

    r1y0 = r1[1]  # расстояние до стенки Y
    r1y1 = Y - r1[1]
    r2y0 = r2[1]
    r2y1 = Y - r2[1]
    dv1y = k*Q*(0 - r1[1]) / r1y0**3 + k*Q*(Y - r1[1]) / r1y1**3  # приращениe скорости от стенки Y
    dv2y = k*Q*(0 - r2[1]) / r2y0**3 + k*Q*(Y - r2[1]) / r2y1**3

    r1z0 = r1[2]  # расстояние до стенки Z
    r1z1 = Z - r1[2]
    r2z0 = r2[2]
    r2z1 = Z - r2[2]
    dv1z = k*Q*(0 - r1[2]) / r1z0**3 + k*Q*(Z - r1[2]) / r1z1**3  # приращениe скорости от стенки Z
    dv2z = k*Q*(0 - r2[2]) / r2z0**3 + k*Q*(Z - r2[2]) / r2z1**3

    dv1xyz = np.array([dv1x, dv1y, dv1z], dtype="float64")  # общее приращение скорости от стенок
    dv2xyz = np.array([dv2x, dv2y, dv2z], dtype="float64")

    DV1 = dv1 + dv1xyz  # общее приращение скорости
    DV2 = dv2 + dv2xyz

    dr1 = v1  # вектор приращения пути
    dr2 = v2

    r_derivs = np.concatenate((dr1, dr2))
    v_derivs = np.concatenate((DV1, DV2))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs


params = np.array([r1, r2, v1, v2])  # массив координат и векторов скоростей
params = params.flatten()  # одномерный np.array
time_arr = np.linspace(0, 0.5, 1000)  # шаги по времени: старт, стоп, количество шагов
sol = scipy.integrate.odeint(Equations,  # уравнения движения частиц
                           params,  # основные параметры (координаты и скорости)
                           time_arr,  # шаги по времени
                           args=(k, q1, q2))  # дополнительные параметры
r1_sol = sol[:,0:3]  # полученные точки решения для частицы 1
r2_sol = sol[:,3:6]  # для частицы 2

# график
fig = plt.figure(figsize=(6, 6))  # размер полотна
ax = fig.add_subplot(111, projection="3d")  # 3D-проекция
# контейнер
ax.plot([0, X], [0, 0], [0, 0], color="orange")  # X рёбра
ax.plot([0, X], [Y, Y], [0, 0], color="orange")
ax.plot([0, X], [0, 0], [Z, Z], color="orange")
ax.plot([0, X], [Y, Y], [Z, Z], color="orange")
ax.plot([0, 0], [0, Y], [0, 0], color="orange")  # Y рёбра
ax.plot([X, X], [0, Y], [0, 0], color="orange")
ax.plot([0, 0], [0, Y], [Z, Z], color="orange")
ax.plot([X, X], [0, Y], [Z, Z], color="orange")
ax.plot([0, 0], [0, 0], [0, Z], color="orange")  # Z рёбра
ax.plot([X, X], [0, 0], [0, Z], color="orange")
ax.plot([0, 0], [Y, Y], [0, Z], color="orange")
ax.plot([X, X], [Y, Y], [0, Z], color="orange")
M = max(X, Y, Z)  # точка максимальной длины стенки для единого масштаба осей
ax.scatter(M, M, M, color="orange")  # угловая точка по всем осям
# кривые движения частиц
ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="b")
ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="r")
# конечные положения частиц
ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="b", label="электрон 1")
ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="r", label="электрон 2")
# подписи
ax.set_xlabel("x", fontsize=14)  # ось x
ax.set_ylabel("y", fontsize=14)  # ось y
ax.set_zlabel("z", fontsize=14)  # ось z
ax.set_title("Кулоновское взаимодействие электронов\nметод частица-частица")  # заголовок
ax.legend(loc="upper left")  # легенда

plt.show()
