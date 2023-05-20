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
r1 = [0.0, 0.0, 0.0]  # начальные координаты в м
r2 = [0.0001, 0.0001, 0.0001]
r1 = np.array(r1, dtype="float64")  # перевод в np.array
r2 = np.array(r2, dtype="float64")

v1 = [0.002, 0.001, 0.003]  # начальная скорость в м/с
v2 = [-0.003, -0.003, -0.001]
v1 = np.array(v1, dtype="float64")  # перевод в np.array
v2 = np.array(v2, dtype="float64")

m1 = 9.109_383_7015e-31  # масса электрона в кг
m2 = 9.109_383_7015e-31

q1 = -1.602_176_634e-19  # заряд электрона в Кл
q2 = -1.602_176_634e-19

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
    dr1 = v1  # вектор приращения пути
    dr2 = v2

    r_derivs = np.concatenate((dr1, dr2))
    v_derivs = np.concatenate((dv1, dv2))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs


params = np.array([r1, r2, v1, v2])
params = params.flatten()
time_arr = np.linspace(0, 0.05, 1001)  # шаги по времени: старт, стоп, количество шагов
sol = scipy.integrate.odeint(Equations,  # уравнения движения частиц
                           params,  # основные параметры (координаты и скорости)
                           time_arr,  # шаги по времени
                           args=(k, q1, q2))  # дополнительные параметры
r1_sol = sol[:,:3]
r2_sol = sol[:,3:6]

# график
fig = plt.figure(figsize=(7, 7))  # размер полотна
ax = fig.add_subplot(111, projection="3d")  # 3D-проекция
# кривые движения частиц
ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="b")
ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="r")
# точки останова частиц
ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="b", label="частица 1")
ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="r", label="частица 2")
# подписи
ax.set_xlabel("x", fontsize=14)  # ось x
ax.set_ylabel("y", fontsize=14)  # ось y
ax.set_zlabel("z", fontsize=14)  # ось z
ax.set_title("Кулоновское взаимодействие электронов\nметод частица-частица")  # заголовок
ax.legend(loc="upper left")  # легенда

plt.show()