import re
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import time

import math


def adjacency_list_filling(m, n, H):
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

def adjacency_list_filling_reverse(m, n, H):
    for i in range(m):
        for j in range(n):
            if mas[i,j] <= H:
                if i - 1 >= 0:
                    if j - 1 >= 0 and mas[i-1,j-1] <= H:
                        adjacency_list[n * i + j].append(n * (i-1) + (j-1))
                    if mas[i-1,j] <= H:
                        adjacency_list[n * i + j].append(n * (i-1) + (j))
                    if j + 1 < n and mas[i-1,j+1] <= H:
                        adjacency_list[n * i + j].append(n * (i-1) + (j+1))
                if i + 1 < m:
                    if j - 1 >= 0 and mas[i+1,j-1] <= H:
                        adjacency_list[n * i + j].append(n * (i+1) + (j-1))
                    if mas[i+1,j] <= H:
                        adjacency_list[n * i + j].append(n * (i+1) + (j))
                    if j + 1 < n and mas[i+1,j+1] <= H:
                        adjacency_list[n * i + j].append(n * (i+1) + (j+1))
                if j - 1 >= 0 and mas[i,j-1] <= H:
                    adjacency_list[n * i + j].append(n * (i) + (j-1))
                if j + 1 < n and mas[i,j+1] <= H:
                    adjacency_list[n * i + j].append(n * (i) + (j+1))

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

def island_search(m, n, H):
    adjacency_list_filling(m, n, H)
    for i in range(m):
        for j in range(n):
            g = n * i + j
            if mas[i, j] >= H and g not in visited:
                bfs(g)
                array_of_islands.append(island[:])
                island.clear()
    # print('для отсечки H=',H)
    for i in range(len(array_of_islands)):
        array_of_island_squares.append(len(array_of_islands[i]))
    #     print('площадь',i,'острова', len(array_of_islands[i]))
    array_of_island_count.append(len(array_of_islands))
    array_of_squares.append(sum(array_of_island_squares) * unit_of_square)
    # print('суммарная площадь выше отсечки',S)

def island_search_reverse(m, n, H):
    adjacency_list_filling_reverse(m, n, H)
    for i in range(m):
        for j in range(n):
            g = n * i + j
            if mas[i, j] <= H and g not in visited:
                bfs(g)
                array_of_islands.append(island[:])
                island.clear()
    # print('для отсечки H=',H)
    for i in range(len(array_of_islands)):
        array_of_island_squares.append(len(array_of_islands[i]))
    #     print('площадь',i,'острова', len(array_of_islands[i]))
    array_of_island_count.append(len(array_of_islands))
    array_of_squares.append(sum(array_of_island_squares) * unit_of_square)
    # print('суммарная площадь выше отсечки',S)

def get_nearest_value(iterable, value):
    return min(iterable, key=lambda x: abs(x - value))

def plotting_and_saving_heatmap(H_proc):
    H = H_proc * np.around(max_value, 3)/100
    masked_mas = np.where(mas >= H, 1, 0)
    hm = sn.heatmap(data=masked_mas,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='gray')
    # cbar = False
    fig = hm.get_figure()
    # Title = f'Отсечка = {np.around(H,3)} нм'
    Title = f'{H_proc} %'
    plt.title(Title, fontsize=14)
    fig.savefig(r'C:\Users\ASTRO\PycharmProjects\Find_island_2dmass\Heatmaps\heatmap_{}_prob%.png'.format(H_proc), bbox_inches="tight")
    hm.clear()
    fig.clf()

def plotting_and_saving_heatmap_reverse(H_proc):
    H = H_proc * np.around(max_value, 3)/100
    masked_mas = np.where(mas <= H, 1, 0)
    hm = sn.heatmap(data=masked_mas,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='gray'
                    )
    fig = hm.get_figure()
    # Title = f'Отсечка = {np.around(H,3)} нм'
    Title = f'{H_proc} %'
    plt.title(Title, fontsize=14)
    fig.savefig(r'C:\Users\ASTRO\PycharmProjects\Find_island_2dmass\Heatmaps_reverse\heatmap_reverse_{}%.png'.format(H_proc), bbox_inches="tight")
    hm.clear()
    fig.clf()

def plotting_and_saving_histograms(H_proc):
    hist = sn.histplot(array_of_island_squares,
                       bins=100,
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
mas = np.loadtxt("0.1_70.txt", skiprows=4, delimiter='\t', dtype=float)*koef #костыль, пока что сразу указываю, что надо умножить на 10 в 8
m = len(mas) # количество строк
n = len(mas[0]) # количество столбцов
max_value = np.amax(mas) # максимальная интенсивность в массиве
min_value = np.amin(mas[mas > 0.0]) # минимальная ненулевая интенсивность

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
array_of_islands = [] #массив в котором содержатся массивы номеров точек для каждого отдельного острова
array_of_squares = [] #массив в который заносится суммарная площадь островов для каждой отсечки
array_of_island_squares = [] #массив, в который заносится площадь каждого острова для каждой отсечки: массив массивов.
array_of_island_count = [] #массив, в который заносится количество островов для каждой отсечки

# искомая отсечка 27.35075 nm с точностью по площади до 4 знака, — отсечка, при которой площадь над = площади под

#Тут код для поиска равных площадей
# u = 0
# for y in np.linspace(27.318, 27.449, num=20, endpoint=False):
#     print(f'площадь {array_of_squares[u]} при отсечке {y} nm, номер в массиве {u}')
#     if math.isclose(need_to_find_square, array_of_squares[u],rel_tol=1e-4):
#         print(f'искомая отсечка {y} nm, номер в массиве {u}')
#     u += 1


H_mas_proc = np.linspace(10, 100, num=20, endpoint=False)
H_mas = np.around(H_mas_proc * np.around(max_value,3)/100, 2)
# H_mas = list(np.linspace(np.around(min_value,3), np.around(max_value,3), num=5, endpoint=False))
for h in H_mas_proc:
    # island_search(m, n, h * np.around(max_value,3)/100)
    # plotting_and_saving_heatmap(h)
    # plotting_and_saving_histograms(h)
    island_search_reverse(m, n, h * np.around(max_value,3)/100)
    # plotting_and_saving_heatmap_reverse(h)

    # очистка буфферных массивов
    for i in range(len(adjacency_list)):
        adjacency_list[i].clear()
    array_of_islands.clear()
    array_of_island_squares.clear()
    island.clear()
    visited.clear()
    Q.clear()

# xlabel='H, nm'
# ylabel='S, mcm\u00B2'
# data = pd.DataFrame(data=list(zip(H_mas, np.around(array_of_squares,2))), columns=[xlabel, ylabel])
# lin_fig = sn.lineplot(data=data,x=xlabel,y=ylabel)
# plt.show()
# print(H_mas)

xlabel='H, %'
ylabel='Кол-во островов'
data = pd.DataFrame(data=list(zip(H_mas_proc, array_of_island_count)), columns=[xlabel, ylabel])
lin_fig = sn.lineplot(data=data,x=xlabel,y=ylabel)
print(array_of_island_count)
plt.show()



