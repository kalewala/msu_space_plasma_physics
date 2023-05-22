"""Численная модель кулоновского взимодействия частиц
Метод частица-частица
=====================
Python 3.10.7"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


t0 = 0  # начальное состояние - нулевое время
Tend = 0.0025  # конечное время
N = 1000  # количество шагов интегрирования

# частицы
r1 = [0.0, 0.0, 0.0]  # начальные координаты в м
r2 = [0.1, 0.1, 0.1]
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
    dv1 = k*q2*q1*(r1 - r2) / (m1 * r12**3) # вектор приращения скорости
    dv2 = k*q1*q2*(r2 - r1) / (m2 * r12**3)
    dr1 = v1   # вектор приращения пути
    dr2 = v2 

    r_derivs = np.concatenate((dr1, dr2))
    v_derivs = np.concatenate((dv1, dv2))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs


params = np.array([r1, r2, v1, v2])
params = params.flatten()
time_arr = np.linspace(t0, Tend, N)  # шаги по времени: старт, стоп, количество шагов
sol = scipy.integrate.odeint(CoulombsEquations,  # уравнения движения частиц
                           params,  # основные параметры (координаты и скорости)
                           time_arr,  # шаги по времени
                           args=(k, q1, q2))  # дополнительные параметры
r1_sol = sol[:,:3]
r2_sol = sol[:,3:6]

# график
fig = plt.figure("Численная модель", figsize=(6, 6))  # размер полотна
ax = fig.add_subplot(111, projection="3d")  # 3D-проекция
# кривые движения частиц
ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="b")
ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="r")
# точки останова частиц
ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2],
           color="b", label=f"Частица 1, v={round(np.linalg.norm(v1), 2)} м/с")
ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2],
           color="r", label=f"Частица 2, v={round(np.linalg.norm(v2), 2)} м/с")
# подписи
ax.set_title("Кулоновское взаимодействие двух частиц (электронов)\nметод частица-частица")  # заголовок
ax.legend(loc="upper left", title=f"t={Tend}")  # легенда
ax.set_xlabel("x", fontsize=14)  # ось x
ax.set_ylabel("y", fontsize=14)  # ось y
ax.set_zlabel("z", fontsize=14)  # ось z

plt.show()