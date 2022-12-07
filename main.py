import re
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import math

# заполнение списка смежности: каждому номеру вершины сопоставляются номера её соседей. Островом считается всё что >= H ([H] = нм)
def adjacency_list_filling(H):
    m = len(mas)
    n = len(mas[0])
    for i in range(m):
        for j in range(n):
            if mas[i,j] >= H:
                if i - 1 >= 0:
                    if j - 1 >= 0 and mas[i-1,j-1] >= H:
                        adjacency_list[n * i + j].append(n * (i-1) + (j-1))
                    if mas[i-1,j] >= H:
                        adjacency_list[n * i + j].append(n * (i-1) + (j))
                    if j + 1 < n and mas[i-1,j+1] >= H:
                        adjacency_list[n * i + j].append(n * (i-1) + (j+1))
                if i + 1 < m:
                    if j - 1 >= 0 and mas[i+1,j-1] >= H:
                        adjacency_list[n * i + j].append(n * (i+1) + (j-1))
                    if mas[i+1,j] >= H:
                        adjacency_list[n * i + j].append(n * (i+1) + (j))
                    if j + 1 < n and mas[i+1,j+1] >= H:
                        adjacency_list[n * i + j].append(n * (i+1) + (j+1))
                if j - 1 >= 0 and mas[i,j-1] >= H:
                    adjacency_list[n * i + j].append(n * (i) + (j-1))
                if j + 1 < n and mas[i,j+1] >= H:
                    adjacency_list[n * i + j].append(n * (i) + (j+1))


#Алгоритм поиска в ширину: передается номер вершины, он сохраняет остров, в котором эта вершина находится

def bfs(v):
    visited.add(v)
    island.append(v)
    Q.append(v)
    while Q:
        k = Q.pop(0)
        for i in adjacency_list[k]:
             if i not in visited:
                 visited.add(i)
                 island.append(i)
                 Q.append(i)


# Пробегаем по массиву и производим поиск остров. Островом считается всё что >= H ([H] = нм)
def island_search(H):
    m = len(mas)
    n = len(mas[0])
    adjacency_list_filling(H)
    for i in range(m):
        for j in range(n):
            g = n * i + j
            if mas[i, j] >= H and g not in visited:
                bfs(g)
                array_of_islands.append(island[:])
                island.clear()

    for i in range(len(array_of_islands)):
        array_of_island_squares.append(len(array_of_islands[i])*unit_of_square)
    array_of_island_count.append(len(array_of_islands))
    array_of_squares.append(sum(array_of_island_squares)/total_square)

#функция, которая находит близжайший к указанному элемент в массиве. В коде не используется, но может пригодиться
def get_nearest_value(iterable, value):
    return min(iterable, key=lambda x: abs(x - value))

# рисуем срез и сохраняем в папку Heatmaps. Передается значение H_proc — в процентах. Островом считается всё что >= H (H считается по формуле внутри)
def plotting_and_saving_heatmap(H_proc):
    H = np.around(H_proc*(max_value-min_value)/100 + min_value,3)
    masked_mas = np.where(mas >= H, 1, 0)
    hm = sn.heatmap(data=masked_mas,
                    xticklabels=False,
                    yticklabels=False,
                    cbar=False,
                    cmap='gray')
    fig = hm.get_figure()
    # Title = f'Отсечка = {np.around(H,3)} нм'
    Title = f'{np.around(H_proc,2)} %'
    plt.title(Title, fontsize=14)
    fig.savefig(r'C:\Users\ASTRO\PycharmProjects\Find_island_2dmass\Heatmaps\heatmap_{}%.png'.format(np.around(H_proc,2)), bbox_inches="tight")
    hm.clear()
    fig.clf()

# рисуем гистограммы распределения размеровов островов. Передается значение H в процентах, только для названия
def plotting_and_saving_histograms(H_proc):
    hist = sn.histplot(array_of_island_squares,
                       bins=50,
                       log_scale=True)
    fig = hist.get_figure()
    # Title = f'Отсечка = {np.around(H, 3)} нм'
    plt.xlabel("log(S)")
    plt.ylabel("Кол-во островов")
    Title = f'{H_proc} %'
    plt.title(Title, fontsize=14)
    fig.savefig(r'C:\Users\ASTRO\PycharmProjects\Find_island_2dmass\Histograms\histogram_{}%.png'.format(H_proc),
                bbox_inches="tight")
    hist.clear()
    fig.clf()

reg_num = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

with open("0.1_70.txt", encoding="utf-8", mode="r") as file:
    width_image = re.search(reg_num, file.readlines()[1])[0]
    file.seek(0)   #костыль, но можно наверно элегантнее считать две строки с числами
    length_image = re.search(reg_num, file.readlines()[2])[0]

koef = 10**9 # умножаем на 10^9, так как у нас нанометры
width_image = float(width_image) #ширина и длина области фото, считается что это микрометры. Пока нет возможности менять, в случае если единицы другие
length_image = float(length_image)
mas = np.loadtxt("0.1_70.txt", skiprows=4, delimiter='\t', dtype=float)*koef #считываем массив
m = len(mas) # количество строк
n = len(mas[0]) # количество столбцов
max_value = np.amax(mas) # максимальная интенсивность в массиве
min_value = np.amin(mas) # минимальная ненулевая интенсивность

total_square = width_image*length_image

unit_of_square = (width_image*length_image)/(m*n)
#считается, что образец квадратный, поэтому считаю единицу так:
# площадь фото делить на количество пикселей. Единицы измерения: микрометры
print('unit of square, micro_m', unit_of_square)

adjacency_list = [[] for _ in range(m * n)] #список смежности для элементов массива
# далее в adjacence_fulling номеру каждого элемента удовлетворяющего условию H приписывается список номеров вершин, которые тоже >= H
visited = set()  # Посещена ли вершина?
Q = []  # Очередь
island = [] #массив, в котором накапливаются номера точек отдельного острова, который потом присоединяется к массиву array_of_island
hole =[]
array_of_islands = [] #массив в котором содержатся массивы номеров точек для каждого отдельного острова
array_of_squares = [] #массив в который заносится суммарная площадь островов для каждой отсечки
array_of_island_squares = [] #массив, в который заносится площадь каждого острова для каждой отсечки: массив массивов.
array_of_island_count = [] #массив, в который заносится количество островов для каждой отсечки
array_of_average_square = [] #массив который равен делению суммарной площади на количество островов для каждой отсечки


# отсечка, при которой площадь над = площади под — отсечка 27.35075 nm с точностью до 4 знака

#Тут код для поиска равных площадей
# u = 0
# for y in np.linspace(27.318, 27.449, num=20, endpoint=False):
#     print(f'площадь {array_of_squares[u]} при отсечке {y} nm, номер в массиве {u}')
#     if math.isclose(need_to_find_square, array_of_squares[u],rel_tol=1e-4):
#         print(f'искомая отсечка {y} nm, номер в массиве {u}')
#     u += 1

H_mas_proc = np.linspace(1, 100, num=100) #ВХОДНОЙ ПАРАМЕТР: задаем разбиение в процентах
H_mas = np.around(H_mas_proc*(max_value-min_value)/100 + min_value, 3) #Перерасчет в нанометры
for h in H_mas_proc:
    island_search(np.around(h*(max_value-min_value)/100 + min_value, 3)) # основная функция, которая заполняет массивы
    plotting_and_saving_heatmap(h)
    # plotting_and_saving_histograms(h)


    # очистка буфферных массивов
    for i in range(len(adjacency_list)):
        adjacency_list[i].clear()
    array_of_islands.clear()
    array_of_island_squares.clear()
    island.clear()
    visited.clear()
    Q.clear()


for i in range(len(array_of_squares)):
    array_of_average_square.append(array_of_squares[i]/array_of_island_count[i])




# # Построение линейного графика. Зависимости площади островов от высоты
# xlabel = 'H, %'
# ylabel = 'S, %' #'S, \u00B5m\u00B2'
# lin_fig_1 = sn.lineplot(data=
#                       pd.DataFrame(
#                           data=list(zip(H_mas_proc, np.around(array_of_average_square,3))),
#                           columns=[xlabel, ylabel]),
#                       x=xlabel,
#                       y=ylabel).get_figure()
# lin_fig_2 = sn.lineplot(data=pd.DataFrame(data=list(zip(H_mas_proc, dydx)),columns=[xlabel, ylabel]),
#                       x=xlabel,
#                       y=ylabel).get_figure()
# lin_fig.savefig(r'C:\Users\ASTRO\PycharmProjects\Find_island_2dmass\Graphs\Average_ot_h%.png',
#                 bbox_inches="tight")
# plt.show()


# for i in range(len(array_of_island_count)):
#     print('Отсечка ',H_mas_proc[i],' % количество островов ', array_of_island_count[i])

# # Построение линейного графика Зависимости количества островов от высоты
# xlabel='H, %'
# ylabel='Кол-во островов'
# data = pd.DataFrame(data=list(zip(H_mas_proc, array_of_island_count)), columns=[xlabel, ylabel])
# lin_fig = sn.lineplot(data=data,x=xlabel,y=ylabel)
# plt_fig = lin_fig.get_figure()
# plt_fig.savefig(r'C:\Users\ASTRO\PycharmProjects\Find_island_2dmass\Graphs\Count_ot_h_2%.png',
#                 bbox_inches="tight")
# plt.show()



