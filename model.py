"""Численная модель кулоновского взимодействия частиц
Метод частица-частица
=====================
Python 3.10.7"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


t = 0  # начальное состояние - нулевое время
dt = 1  # шаг по времени
N = 10  # количество шагов

xi = np.array([0.0000000001, 0.0000000004, 0.0])  # начальные координаты частицы i(x, y, z)
vi = np.array([0.0000000002, 0.0000000001, 0.0])  # начальные векторы скоростей i(vx, vy, vz)
qi = -1.602_176_634e-19  # заряд частицы i
mi = 9.109_383_7015e-31  # масса электрона

xj = np.array([0.000000001, 0.0000000007, 0.000000001])  # начальные координаты частицы j(x, y, z)
vj = np.array([-0.0000000002, -0.0000000001, 0.0])  # начальные векторы скоростей j(vx, vy, vz)
qj = 1.602_176_634e-19  # заряд цастицы j
mj = 9.109_383_7015e-31  # масса эектрона

c = 299_792_458  # скорость света по СИ (м/с)
m0 = 4 * np.pi * 10**(-7)  # магнитная постоянная (Гн/м)
e0 = 1/(m0 * c**2)  # электричекая постоянная
k = 1/(4 * np.pi * e0)  # диэлектрическая проницаемость среды

dx = xj - xi  # векторная разность положения частиц
r = (dx[0]**2 + dx[1]**2 + dx[2]**2)**0.5  # расстояние между частицами

def Fij(r):
    F = (qi*qj * (r))/(4 * np.pi * e0 * np.absolute(r)**3)  # закон Кулона
    return F

def Fv(r, v):
    F = k * (qi * qj * v)/(r**2 * r)
    return F


Xi = np.empty([N, 3])  # матрица значений частицы i
Xj = np.empty([N, 3])  # матрица значений частицы j
Xi[0] = xi
Xj[0] = xj
for i in range(1, N):
    dx = xj - xi  # векторная разность положения частиц
    r = (dx[0]**2 + dx[1]**2 + dx[2]**2)**0.5  # расстояние между частицами
    
    F = np.array([0.0, 0.0, 0.0])
    for n in range(3):
        F[n] = Fv(r, dx[n])

    vi += F
    xi += vi
    Xi[i] = xi

    vj -= F
    xj += vj
    Xj[i] = xj



x1 = np.array([x[0] for x in Xi])
y1 = np.array([x[1] for x in Xi])
z1 = np.array([x[2] for x in Xi])

x2 = np.array([x[0] for x in Xj])
y2 = np.array([x[1] for x in Xj])
z2 = np.array([x[2] for x in Xj])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1[0], y1[0], z1[0], label='x_i', color='b')
ax.plot(x1, y1, z1, label='x_i', color='b')
ax.scatter(x2[0], y2[0], z2[0], label='x_j', color='r')
ax.plot(x2, y2, z2, label='x_j', color='r')
plt.show()
