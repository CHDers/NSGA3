# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/2 11:57
# @Author  : Yanjun Hao
# @Site    : 
# @File    : a.py
# @Software: PyCharm 
# @Comment :


import random
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

# 设置参数
pop_size = 100  # 种群大小
gen_size = 100  # 进化代数
pc = 1  # 交叉概率
pm = 0.3  # 变异概率
num_obj = 2  # 目标函数个数
x_range = (-10, 10)  # 自变量取值范围


# 定义自变量的类
class Individual:
    def __init__(self, x):
        self.x = x
        self.objs = [None] * num_obj
        self.rank = None
        self.distance = 0.0

    # 计算目标函数的值
    def evaluate(self):
        self.objs[0] = self.x * self.x
        self.objs[1] = (2 - self.x) ** 2


# 初始化种群
pop = [Individual(random.uniform(*x_range)) for _ in range(pop_size)]

# 进化
for _ in range(1):
    print(f"第{_}次迭代")
    # 计算目标函数的值
    for ind in pop:
        ind.evaluate()

    # 非支配排序
    fronts = [set()]
    for ind in pop:
        ind.domination_count = 0
        ind.dominated_set = set()

        for other in pop:  # 通过循环计算每个个体可以支配的其他个体集合和能支配本个体的个体数
            if ind.objs[0] < other.objs[0] and ind.objs[1] < other.objs[1]:
                ind.dominated_set.add(other)
            elif ind.objs[0] > other.objs[0] and ind.objs[1] > other.objs[1]:
                ind.domination_count += 1

        if ind.domination_count == 0:  # 能够支配ind个体的个体数为0，也就是说没有其他个体能够支配本个体ind
            ind.rank = 1
            fronts[0].add(ind)

    print(len(fronts[0]))
    rank = 1
    while fronts[-1]:
        next_front = set()

        for ind in fronts[-1]:
            ind.rank = rank
            # 去除能支配这个个体的上一级非支配解
            for dominated_ind in ind.dominated_set:
                dominated_ind.domination_count -= 1

                if dominated_ind.domination_count == 0:
                    next_front.add(dominated_ind)

        fronts.append(next_front)
        rank += 1

    # 计算拥挤度距离
    pop_for_cross = set()
    for front in fronts:
        if len(front) == 0:
            continue

        sorted_front = sorted(list(front), key=lambda ind: ind.rank)  # front是set
        for i in range(num_obj):
            sorted_front[0].objs[i] = float('inf')
            sorted_front[-1].objs[i] = float('inf')
            for j in range(1, len(sorted_front) - 1):
                delta = sorted_front[j + 1].objs[i] - \
                        sorted_front[j - 1].objs[i]
                if delta == 0:
                    continue

                sorted_front[j].distance += delta / (x_range[1] - x_range[0])

        front_list = list(sorted_front)
        front_list.sort(key=lambda ind: (-ind.rank, -ind.distance))
        selected_inds = front_list
        if len(pop_for_cross) + len(selected_inds) <= pop_size:
            pop_for_cross.update(selected_inds)
        elif len(pop_for_cross) + len(selected_inds) >= pop_size and len(pop_for_cross) < pop_size:
            part_selected_inds = selected_inds[:(pop_size - len(pop_for_cross))]
            pop_for_cross.update(part_selected_inds)
            break