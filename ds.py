# Linking: https://blog.csdn.net/m0_72053284/article/details/131255000

import random
import matplotlib.pyplot as plt
import numpy as np
import time
 
start_time=time.time()
 
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
for _ in range(gen_size):
    print(f"第{_}次迭代")
    # 计算目标函数的值
    for ind in pop:
        ind.evaluate()
 
    # 非支配排序
    fronts = [set()]
    for ind in pop:
        ind.domination_count = 0
        ind.dominated_set = set()
 
        for other in pop:
            if ind.objs[0] < other.objs[0] and ind.objs[1] < other.objs[1]:
                ind.dominated_set.add(other)
            elif ind.objs[0] > other.objs[0] and ind.objs[1] > other.objs[1]:
                ind.domination_count += 1
 
        if ind.domination_count == 0:
            ind.rank = 1
            fronts[0].add(ind)
 
    rank = 1
    while fronts[-1]:
        next_front = set()
 
        for ind in fronts[-1]:
            ind.rank = rank
 
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
 
        sorted_front = sorted(list(front), key=lambda ind: ind.rank)
        for i in range(num_obj):
            sorted_front[0].objs[i] = float('inf')
            sorted_front[-1].objs[i] = float('inf')
            for j in range(1, len(sorted_front) - 1):
                delta = sorted_front[j + 1].objs[i] - sorted_front[j - 1].objs[i]
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
 
    # 计算每个目标函数的权重向量和参考点
    """
当num_obj=2时，定义的ref_vectors列表内容为[[1.0, 0], [0, 1.0]]，其中包含了所有的权重向量。因为在该问题中我们有两个目标函数，所以共需要两个权重向量。
那么ref_vectors中的第一个子列表[1.0, 0]代表的是第一个目标函数的权重向量，其中1.0表示在第一个目标函数上最大化目标函数值，0表示在第二个目标函数上最小化目标函数值。
同理，ref_vectors中的第二个子列表[0, 1.0]代表的是第二个目标函数的权重向量，其中1.0表示在第二个目标函数上最大化目标函数值，0表示在第一个目标函数上最小化目标函数值。
总之，ref_vectors中的每个子列表代表一个不同的权重向量，它们分别控制着各个目标函数的优化方向。
    """
    ref_vectors = []
    for i in range(num_obj):
        vec = [0] * num_obj
        vec[i] = 1.0
        ref_vectors.append(vec)
 
    for vec in ref_vectors:
        # 根据权重向量vec，计算出一个参考点ref_point，在目标函数空间中代表着该权重下的理想解。
        ref_point = [vec[j] * x_range[j] for j in range(num_obj)]
        # 根据权重向量vec，计算出一个参考点ref_point，在目标函数空间中代表着该权重下的理想解。
        weighted_objs = [(ind.objs[k] - ref_point[k]) * vec[k] for ind in pop_for_cross for k in range(num_obj)]
        # 对于当前的所有个体，在目标函数空间中的加权距离进行排序。
        sorted_objs = sorted(weighted_objs)
        # 在排序后的加权距离列表中选择中位数值，并将其作为拥挤度距离的计算基准。
        median_objs = [sorted_objs[len(sorted_objs) // 2 + offset] for offset in (-1, 0, 1)]
        # 根据当前参考点和中位数计算出其到其他个体最短距离。
        min_dist = np.linalg.norm(np.array(median_objs[:num_obj]) - ref_point)
        # 遍历种群中的每个个体ind，计算其在目标函数空间中针对当前权重向量vec的加权距离，并与之前计算出的最短距离min_dist比较，得到本次遍历中所有个体所能达到的最小距离值。
        for ind in pop_for_cross:
            dist = np.linalg.norm(np.array([(ind.objs[k] - ref_point[k]) * vec[k] for k in range(num_obj)]))
            if dist < min_dist:
                min_dist = dist
        # 再次遍历种群中的每个个体ind，根据之前得到的最小距离值，计算该个体的拥挤度距离。这里采用了一种计算公式，即将每个个体的拥挤度距离设定为其当前拥挤度距离值加上其到其他个体最小距离的倒数。
        for ind in pop_for_cross:
            dist = np.linalg.norm(np.array([(ind.objs[k] - ref_point[k]) * vec[k] for k in range(num_obj)]))
            ind.distance += (min_dist / (dist + min_dist))
 
    # 通过拥挤度距离与分配密度估计来选择进行交叉的个体
    new_pop = set()
    while len(new_pop) < pop_size:
        pool = random.sample(pop_for_cross, 2)
        pool_dist = [ind.distance for ind in pool]
        parent1 = pool[np.argmax(pool_dist)]
        parent2 = pool[1 - np.argmax(pool_dist)]
 
        child_x = (parent1.x + parent2.x) / 2
        delta_x = abs(parent1.x - parent2.x)
        child_x += delta_x * random.uniform(-1, 1)
 
        child = Individual(child_x)
        new_pop.add(child)
 
    # 变异
    for ind in new_pop:
        if random.random() < pm:
            delta_x = random.uniform(-1, 1) * (x_range[1] - x_range[0])
            ind.x += delta_x
            ind.x = max(x_range[0], min(x_range[1], ind.x))
 
    # 更新种群,把原来的精英（pop_for_cross）保留下来。即精英保留策略
    pop = list(new_pop) + list(pop_for_cross)
 
# 输出最优解集合
for ind in pop:
    ind.evaluate()
 
pareto_front = set()
for ind in pop:
    dominated = False
    for other in pop:
        if other.objs[0] < ind.objs[0] and other.objs[1] < ind.objs[1]:
            dominated = True
            break
    if not dominated:
        pareto_front.add(ind)
 
print("Pareto front:")
for ind in pareto_front:
    print(f"x={ind.x:.4f}, y1={ind.objs[0]:.4f}, y2={ind.objs[1]:.4f}")
 
# 可视化
plt.scatter([ind.objs[0] for ind in pop], [ind.objs[1] for ind in pop], c='gray', alpha=0.5)
plt.scatter([ind.objs[0] for ind in pareto_front], [ind.objs[1] for ind in pareto_front], c='r')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
end_time=time.time()
print(f"求得的帕累托解的个数为：{len(pareto_front)}")
print(f"\n程序执行时间为：{end_time-start_time}")
plt.show()
