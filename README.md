# msu_space_plasma_physics

Дополнительный рекомендованный курс по выбору "Численные методы в физике космической плазмы" от Факультета космических исследований МГУ (весенний семестр 2023).

## Численная модель кулоновского взаимодействия двух частиц
Закон Кулона описывает взаимодействие между двумя неподвижными точечными зарядами в вакууме и записывается в виде:
$$\mathnormal{F} = \mathnormal{k} \frac{\mathnormal{q_{1}q_{2}}}{\mathnormal{r^{2}}}$$
где $\mathnormal{F}$ – сила взаимодействия между зарядами, $\mathnormal{k}$ – коэффициент пропорциональности (постоянная Кулона), $\mathnormal{q_{1}}$ и $\mathnormal{q_{2}}$ – величина зарядов (со знаком), $\mathnormal{r}$ – расстояние между зарядами.
В системе СИ величина ампера определена таким образом, что коэффициент $`\mathnormal{k} = \mathnormal{c^{2} \cdot 10^{-7}\text{Гн/м} = 8{,}9875517873681764 \cdot 10^{9}\text{Н^{2}м^{2}/Кл^{2}}}$ (или $\text{Ф^{-1} \cdot \text{м}}`$). 
Сила, с которой заряд $\mathnormal{q_{1}}$ действует на заряд $\mathnormal{q_{2}}$ определеяется по формуле:
$$\mathnormal{\vec{F}_{12}} = $$