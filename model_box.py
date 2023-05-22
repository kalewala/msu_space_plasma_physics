"""Численная модель кулоновского взимодействия частиц
в контейнере с заряженными стенками
Метод частица-частица
=====================
Python 3.10.7"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


t0 = 0  # начальное состояние - нулевое время
Tend = 0.005  # конечное время
N = 1000  # количество шагов интегрирования

# частицы
r1 = [0.04, 0.03, 0.05]  # начальные координаты в м
r2 = [0.06, 0.06, 0.09]
r1 = np.array(r1, dtype="float64")  # перевод в np.array
r2 = np.array(r2, dtype="float64")

v1 = [40.0, 50.0, 30.0]  # начальная скорость в м/с
v2 = [-30.0, -50.0, -60.0]
v1 = np.array(v1, dtype="float64")  # перевод в np.array
v2 = np.array(v2, dtype="float64")

m1 = 9.109_383_7015e-31  # масса электрона в кг
m2 = 9.109_383_7015e-31

q1 = -1.602_176_634e-19  # заряд электрона в Кл
q2 = -1.602_176_634e-19

# контейнер
X = 0.1  # длины стенок по координатам в м
Y = 0.1
Z = 0.1
Q = -1.602_176_634e-19  # заряд стенки в Кл

c = 299_792_458  # скорость света по СИ (м/с)
m0 = 4 * np.pi * 10**(-7)  # магнитная постоянная (Гн/м)
e0 = 1/(m0 * c**2)  # электричекая постоянная
k = 1/(4 * np.pi * e0)  # диэлектрическая проницаемость среды

def CoulombsEquations(params, t, k, q1, q2):
    r1 = params[0:3]  # координаты частицы
    r2 = params[3:6]
    v1 = params[6:9]  # вектор скорости частицы
    v2 = params[9:12]

    r12 = np.linalg.norm(r2 - r1)  # расстояние между частицами
    dv1 = k*q2*q1*(r1 - r2) / (m1 * r12**3)  # вектор приращения скорости
    dv2 = k*q1*q2*(r2 - r1) / (m2 * r12**3)

    r1x0 = r1[0]  # расстояние до стенки X
    r1x1 = X - r1[0]
    r2x0 = r2[0]
    r2x1 = X - r2[0]
    dv1x = k*Q*q1*(r1[0] - 0) / (m1 * r1x0**3) + k*Q*q1*(r1[0] - X) / (m1 * r1x1**3)  # приращениe скорости от стенки X
    dv2x = k*Q*q2*(r2[0] - 0) / (m2 * r2x0**3) + k*Q*q2*(r2[0] - X) / (m2 * r2x1**3)

    r1y0 = r1[1]  # расстояние до стенки Y
    r1y1 = Y - r1[1]
    r2y0 = r2[1]
    r2y1 = Y - r2[1]
    dv1y = k*Q*q1*(r1[1] - 0) / (m1 * r1y0**3) + k*Q*q1*(r1[1] - Y) / (m1 * r1y1**3)  # приращениe скорости от стенки Y
    dv2y = k*Q*q2*(r2[1] - 0) / (m2 * r2y0**3) + k*Q*q2*(r2[1] - Y) / (m2 * r2y1**3)

    r1z0 = r1[2]  # расстояние до стенки Z
    r1z1 = Z - r1[2]
    r2z0 = r2[2]
    r2z1 = Z - r2[2]
    dv1z = k*Q*q1*(r1[2] - 0) / (m1 * r1z0**3) + k*Q*q1*(r1[2] - Z) / (m1 * r1z1**3)  # приращениe скорости от стенки Z
    dv2z = k*Q*q2*(r2[2] - 0) / (m2 * r2z0**3) + k*Q*q2*(r2[2] - Z) / (m2 * r2z1**3)

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
time_arr = np.linspace(t0, Tend, N)  # шаги по времени: старт, стоп, количество шагов
sol = scipy.integrate.odeint(CoulombsEquations,  # уравнения движения частиц
                           params,  # основные параметры (координаты и скорости)
                           time_arr,  # шаги по времени
                           args=(k, q1, q2))  # дополнительные параметры
r1_sol = sol[:,0:3]  # полученные точки решения для частицы 1
r2_sol = sol[:,3:6]  # для частицы 2

# график
fig = plt.figure("Численная модель", figsize=(6, 6))  # размер полотна
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
# кривые движения частиц
ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="b")
ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="r")
# конечные положения частиц
ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2],
           color="b", label=f"Частица 1, v = {round(np.linalg.norm(v1), 2)} м/с")
ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2],
           color="r", label=f"Частица 2, v = {round(np.linalg.norm(v2), 2)} м/с")
# подписи
M = max(X, Y, Z)  # точка максимальной длины стенки для единого масштаба осей
ax.set(xlim3d=(0, M), xlabel='x')  # ось x
ax.set(ylim3d=(0, M), ylabel='y')  # ось y
ax.set(zlim3d=(0, M), zlabel='z')  # ось z
ax.set_title("""Кулоновское взаимодействие двух частиц
в контейнере с заряженными стенками
метод частица-частица""")  # заголовок
ax.legend(loc="upper left", title=f"t = {Tend} c")  # легенда

plt.show()
