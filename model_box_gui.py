"""Численная модель кулоновского взимодействия 
двух частиц в контейнере с заряженными стенками
с графическим пользовательским интерфейсом
Метод частица-частица
=====================
Python 3.10.7"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


c = 299_792_458  # скорость света по СИ (м/с)
mu0 = 4 * np.pi * 10**(-7)  # магнитная постоянная (Гн/м)
e0 = 1/(mu0 * c**2)  # электричекая постоянная
k = 1/(4 * np.pi * e0)  # коэффициент пропорциональности

# уравнения Кулона
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


# отрисовка графика
def RunGraph():
    # время интегрирования
    t0 = float(entry_t0.get())  # начальное состояние - нулевое время
    Tend = float(entry_Tend.get())  # конечное время
    N = int(entry_N.get())  # количество шагов интегрирования

    # частицы
    r1 = np.array((entry_r1.get()).split(), dtype="float64")  # начальные координаты в м
    r2 = np.array((entry_r2.get()).split(), dtype="float64")
    v1 = np.array((entry_v1.get()).split(), dtype="float64")  # начальная скорость в м/с
    v2 = np.array((entry_v2.get()).split(), dtype="float64")
    q1 = float(entry_q1.get())  # заряд электрона в Кл
    q2 = float(entry_q2.get())
    global m1, m2
    m1 = float(entry_m1.get())
    m2 = float(entry_m2.get())

    # контейнер
    global X, Y, Z, Q
    X = float(entry_X.get())  # длины стенок по координатам в м
    Y = float(entry_Y.get())
    Z = float(entry_Z.get())
    Q = float(entry_Q.get())  # заряд стенки в Кл

    params = np.array([r1, r2, v1, v2])  # массив координат и векторов скоростей
    params = params.flatten()  # одномерный np.array
    time_arr = np.linspace(t0, Tend, N)  # шаги по времени: старт, стоп, количество шагов
    sol = scipy.integrate.odeint(CoulombsEquations,  # уравнения движения частиц
                            params,  # основные параметры (координаты и скорости)
                            time_arr,  # шаги по времени
                            args=(k, q1, q2))  # дополнительные параметры
    r1_sol = sol[:,0:3]  # точки решений для частицы 1
    r2_sol = sol[:,3:6]  # точки решений для частицы 2

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
    # кривые движения частиц
    ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="b")
    ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="r")
    # конечные положения частиц
    ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="b", 
               label=f"Частица 1, v = {round(np.linalg.norm(v1), 2)} м/с")
    ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="r",
               label=f"Частица 2, v = {round(np.linalg.norm(v2), 2)} м/с")
    # подписи
    M = max(X, Y, Z)  # точка максимальной длины стенки для единого масштаба осей
    ax.set(xlim3d=(0, M), xlabel='x')  # ось x
    ax.set(ylim3d=(0, M), ylabel='y')  # ось y
    ax.set(zlim3d=(0, M), zlabel='z')  # ось z
    #ax.set_title("Кулоновское взаимодействие электронов\nметод частица-частица")  # заголовок
    ax.legend(loc="upper left", title=f"t = {Tend} c")  # легенда

    canvas = FigureCanvasTkAgg(fig, root)
    canvas.get_tk_widget().grid(row=2, column=0, rowspan=23)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)  # стандартный тулбар для графика
    toolbar.update()
    toolbar.grid(row=1, column=0, sticky="w")


# графический пользовательский интерфейс (gui)
root = Tk()
canvas = None
root.title("Численная модель кулоновского взаимодействия")  # заголовок окна
root.resizable(width=False, height=False)  # запрет масштабирования окна

# виджеты параметров
label_params  = Label(text="ПАРАМЕТРЫ")
label_params.grid(row=1, column=1, columnspan=2)

label_r = Label(text="Начальные координаты частиц (м):")
label_r.grid(row=3, column=1, columnspan=2)

label_r1 = Label(text="Частица 1 (x y z):")
label_r1.grid(row=4, column=1, sticky="e")
entry_r1 = Entry(root, width=20)
entry_r1.insert(0,"0.04 0.03 0.05")  # значения по умолчанию
entry_r1.grid(row=4, column=2)

label_r2 = Label(text="Частица 2 (x y z):")
label_r2.grid(row=5, column=1, sticky="e")
entry_r2 = Entry(root, width=20)
entry_r2.insert(0,"0.06 0.06 0.09")  # значения по умолчанию
entry_r2.grid(row=5, column=2)

label_v = Label(text="Начальные скорости частиц (м/с):")
label_v.grid(row=6, column=1, columnspan=2)

label_v1 = Label(text="Частица 1 (x y z):")
label_v1.grid(row=7, column=1, sticky="e")
entry_v1 = Entry(root, width=20)
entry_v1.insert(0,"40.0 50.0 30.0")  # значения по умолчанию
entry_v1.grid(row=7, column=2)

label_v2 = Label(text="Частица 2 (x y z):")
label_v2.grid(row=8, column=1, sticky="e")
entry_v2 = Entry(root, width=20)
entry_v2.insert(0,"-30.0 -50.0 -60.0")  # значения по умолчанию
entry_v2.grid(row=8, column=2)

label_q = Label(text="Заряды частиц (Кл):")
label_q.grid(row=9, column=1, columnspan=2)

label_q1 = Label(text="Частица 1:")
label_q1.grid(row=10, column=1, sticky="e")
entry_q1 = Entry(root, width=20)
entry_q1.insert(0,"-1.602_176_634e-19")  # значения по умолчанию
entry_q1.grid(row=10, column=2)

label_q2 = Label(text="Частица 2:")
label_q2.grid(row=11, column=1, sticky="e")
entry_q2 = Entry(root, width=20)
entry_q2.insert(0,"-1.602_176_634e-19")  # значения по умолчанию
entry_q2.grid(row=11, column=2)

label_m = Label(text="Массы частиц (кг):")
label_m.grid(row=12, column=1, columnspan=2)

label_m1 = Label(text="Частица 1:")
label_m1.grid(row=13, column=1, sticky="e")
entry_m1 = Entry(root, width=20)
entry_m1.insert(0,"9.109_383_7015e-31")  # значения по умолчанию
entry_m1.grid(row=13, column=2)

label_m2 = Label(text="Частица 2:")
label_m2.grid(row=14, column=1, sticky="e")
entry_m2 = Entry(root, width=20)
entry_m2.insert(0,"9.109_383_7015e-31")  # значения по умолчанию
entry_m2.grid(row=14, column=2)

label_box = Label(text="Параметры контейнера:")
label_box.grid(row=15, column=1, columnspan=2)

label_X = Label(text="Длина (X):")
label_X.grid(row=16, column=1, sticky="e")
entry_X = Entry(root, width=20)
entry_X.insert(0,"0.1")  # значения по умолчанию
entry_X.grid(row=16, column=2)

label_Y = Label(text="Ширина (Y):")
label_Y.grid(row=17, column=1, sticky="e")
entry_Y = Entry(root, width=20)
entry_Y.insert(0,"0.1")  # значения по умолчанию
entry_Y.grid(row=17, column=2)

label_Z = Label(text="Высота (Z):")
label_Z.grid(row=18, column=1, sticky="e")
entry_Z = Entry(root, width=20)
entry_Z.insert(0,"0.1")  # значения по умолчанию
entry_Z.grid(row=18, column=2)

label_Q = Label(text="Заряд стенок (Кл):")
label_Q.grid(row=19, column=1, sticky="e")
entry_Q = Entry(root, width=20)
entry_Q.insert(0,"-1.602_176_634e-19")  # значения по умолчанию
entry_Q.grid(row=19, column=2)

label_time = Label(text="Время интегрирования (с):")
label_time.grid(row=20, column=1, columnspan=2)

label_t0 = Label(text="Начальное время:")
label_t0.grid(row=21, column=1, sticky="e")
entry_t0 = Entry(root, width=20)
entry_t0.insert(0,"0")  # значения по умолчанию
entry_t0.grid(row=21, column=2)

label_Tend = Label(text="Конечное время:")
label_Tend.grid(row=22, column=1, sticky="e")
entry_Tend = Entry(root, width=20)
entry_Tend.insert(0,"0.005")  # значения по умолчанию
entry_Tend.grid(row=22, column=2)

label_N = Label(text="Шаги по времени:")
label_N.grid(row=23, column=1, sticky="e")
entry_N = Entry(root, width=20)
entry_N.insert(0,"1000")  # значения по умолчанию
entry_N.grid(row=23, column=2)

# кнопка для вызова функции отрисовки графика
run = Button(
    master=root,
    text="Рассчитать",
    command=RunGraph,
    width=10,
    padx = 5, pady = 5)
run.grid(row=24, column=2)#, sticky='e')


if __name__ == "__main__":
    RunGraph()
    root.mainloop()  # запуск цикла событий для окна