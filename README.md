# msu_space_plasma_physics

Дополнительный рекомендованный курс по выбору "Численные методы в физике космической плазмы" от Факультета космических исследований МГУ (весенний семестр 2023).

## Численная модель кулоновского взаимодействия двух частиц

#### Описание задачи

Закон Кулона описывает взаимодействие между двумя неподвижными точечными зарядами в вакууме и записывается в виде:

$$\mathnormal{F} = \mathnormal{k} \frac{\mathnormal{q_{1}q_{2}}}{\mathnormal{r^{2}}}{,}$$

где $\mathnormal{F}$ – сила взаимодействия между зарядами, $\mathnormal{k}$ – коэффициент пропорциональности (постоянная Кулона), $\mathnormal{q_{1}}$ и $\mathnormal{q_{2}}$ – величина зарядов (со знаком), $\mathnormal{r}$ – расстояние между зарядами.

В системе СИ величина ампера определена таким образом, что коэффициент $\mathnormal{k} = \mathnormal{c}^{2} \cdot 10^{-7}\text{Гн/м} = 8{,}9875517873681764 \cdot 10^{9}\text{Н} \cdot \text{м}^{2}/\text{Кл}^{2}$ (или $\text{Ф}^{-1} \cdot \text{м}$) и определяется по формуле:

$$\mathnormal{k} = \frac{1}{4\pi \varepsilon _{0}}{,}$$

где $\varepsilon _{0} \approx 8{,}85418781762 \cdot 10^{12} \text{Ф/м}$ – электрическая постоянная. По определению в СИ электрическая постоянная $\varepsilon _{0}$ связана со скоростью света $\mathnormal{c}$ и магнитной постоянной $\mu _{0}$ соотношением:

$$\varepsilon _{0} = \frac{1}{\mu _{0}\mathnormal{c}^{2}}$$

Скорость света в вакууме по СИ определяется как константа и равна $299792458 \text{м/с}$. Магнитной постоянной соотвутствует равенство $\mu _{0} = 4 \pi \cdot 10^{-7}\text{Гн/м}$, то есть $\mu _{0} \approx 1{,}2566370614 \cdot 10^{-6}\text{Н/А}^{2}$.

В векторном виде сила, с которой заряд $\mathnormal{q_{1}}$ действует на заряд $\mathnormal{q_{2}}$ определеяется по формуле:

$$\vec{\mathnormal{F}} _{12} = \mathnormal{k}\cdot\frac{\mathnormal{q} _{1} \cdot \mathnormal{q} _{2}}{\mathnormal{r} _{12} ^{2}} \cdot {\frac{\vec{\mathnormal{r}} _{12}}{\mathnormal{r} _{12}}}{,}$$

где $\vec{\mathnormal{F}} _{12}$ – сила, с которой заряд 1 действует на заряд 2, $\vec{\mathnormal{r}} _{12}$ –  вектор, направленный от заряда 1 к заряду 2 и по модулю равный расстоянию между зарядами ($\mathnormal{r} _{12}$). 

Масса частицы не учитывается в законе Кулона, так как он описывает только электрические свойства зарядов. Ускорение частицы можно вычислить с помощью второго закона Ньютона:

$$\vec{\mathnormal{a}} = \frac{\vec{\mathnormal{F}}}{\mathnormal{m}}{,}$$

где $\vec{\mathnormal{a}}$ – ускорение тела, $\vec{\mathnormal{F}}$ – сила, приложенная к телу, а $\mathnormal{m}$ – масса тела.

Таким образом, ускорение частицы зависит от величины и знака взаимодействующих зарядов, расстояния между ними (по закону Кулона) и от массы частицы (по второму закону Ньютона). Отсюда, динамика двух заряженных частиц в вакууме определяется системой уравнений движения для каждой из частиц:

$$\dot{\mathnormal{v}} = \frac{\vec{\mathnormal{F}}}{\mathnormal{m}}{,}$$

$$\dot{\mathnormal{r}} = \vec{\mathnormal{v}}{,}$$

где $\dot{\mathnormal{v}}$ – производная скорости по времени или ускорение и $\dot{\mathnormal{r}}$ – производная расстояния по времени или скорость определяются проекциями по трём осям координат $\mathnormal{x}$, $\mathnormal{y}$ и $\mathnormal{z}$ в трёхмерном евклидовом пространстве. Всего 12 уравнений, по 6 для каждой из частиц.

#### Программная реализация

**model.py** – модель кулоновского взаимодействия двух заряженных частиц (электронов) в вакууме. Переменные и константы заданы стандартными типами данных Python и массивами NumPy array. Система уравнений движения решается функцией `CoulombsEquations`. Численное интегрирование осуществляется с помощью `integrate.odeint` из библиотеки `scipy`, которая в свою очередь использует lsoda из FORTRAN library odepack. По документации это метод Адамса. График с траекториями частиц выводится с помощью `matplotlib.pyplot`:

![Скриншот model.py](/images/screen_model.jpg "Траектории движения двух электронов")

Модель позволяет определить траектории пролёта двух заряженных частиц в вакууме в соответсвии с действием закона Кулона. По умолчанию в качестве частиц используются два электрона. Все единицы измерения и константы соотвутсвуют СИ.

**model_box.py** – модель кулоновского взаимодействия двух заряженных частиц в контейнере с заряженными стенками. Добавлено интегрирование взаимодействия частиц с заряженными стенками, величина заряда которых равна заряду частиц.

![Скриншот model_box.py](/images/screen_model_box.jpg "Траектории движения двух частиц в контейнере с заряженными стенками")

**model_box_gui** – модель кулоновского взаимодействия двух заряженных частиц в контейнере с заряженными стенками. Добавлен графический пользовательский интерфейс из стандартного пакета Tkinter. Через поля ввода можно менять координаты, скорости, заряды и массы частиц, а также параметры контейнера и время интегрирования траекторий.

![Скриншот model_box_gui.py](/images/screen_model_box_gui.jpg "")