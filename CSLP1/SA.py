import numpy as np
import random
import math

from CSLP1.capacity import capacity
from CSLP1.near import near
from CSLP1.population import initialization


class SA:
    def __init__(self, func, x_coords, y_coords, demands, m, max_iter, initial_temp = 1000.0,
cooling_rate = 0.99,min_distance=0.7):
        self.func = func
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.positions = initialization(1, self.m)[0]  # 初始化一个解
        self.best_pos = self.positions.copy()
        self.best_score = float('inf')
        self.current_pos = self.positions.copy()
        self.current_score = self.func(self.current_pos, self.x_coords, self.y_coords, self.demands, self.m)
        self.convergence_curve = []
        self.min_distance = min_distance

    def optimize(self):
        T = self.initial_temp
        for iter in range(self.max_iter):
            new_pos = self.generate_neighbor(self.current_pos)
            # print(new_pos)
            new_score = self.func(new_pos, self.x_coords, self.y_coords, self.demands, self.m)

            if new_score < self.current_score or random.uniform(0, 1) < math.exp((self.current_score - new_score) / T):
                self.current_pos = new_pos
                self.current_score = new_score

            if self.current_score < self.best_score:
                self.best_pos = self.current_pos.copy()
                self.best_score = self.current_score

            T *= self.cooling_rate
            self.convergence_curve.append(self.best_score)
            print(f"Iteration: {iter + 1}, Best fitness: {self.best_score}")

        # 分开输出充电站的位置和容量
        Capacity = capacity(self.best_pos, self.x_coords, self.y_coords, self.demands, self.m)
        print("*************Simulated Annealing****************")
        print("最优位置：", self.best_pos)
        print("最优容量：", Capacity)
        print("最优目标函数值：", self.best_score)
        return self.best_pos, self.best_score, self.convergence_curve, Capacity

    def generate_neighbor(self, position):
        neighbor = position.copy()
        idx = random.randint(0, len(neighbor) - 1)
        neighbor[idx] = neighbor[idx] + np.random.uniform(-self.min_distance, self.min_distance)
        neighbor = np.clip(neighbor, 0, 1)  # 确保位置在合理范围内
        near(neighbor, self.m)
        return neighbor

#
# # 示例目标函数
# def example_func(position, x_coords, y_coords, demands, m):
#     # 这里假设一个简单的目标函数，仅作为示例
#     total_cost = 0
#     for i in range(m):
#         if position[i] > 0.5:
#             total_cost += np.sum(demands)  # 假设充电站的成本与需求量相关
#     return total_cost
#
#
# # 示例用法
# np.random.seed(1)
# x_coords = np.random.uniform(0, 1, 10)  # 示例坐标
# y_coords = np.random.uniform(0, 1, 10)
# demands = np.random.uniform(0, 1, 10)  # 示例需求
# m = 10  # 设施数量
# initial_temp = 1000.0
# cooling_rate = 0.99
# max_iter = 1000  # 最大迭代次数
#
# # 创建并优化模拟退火算法
# sa = SA(example_func, x_coords, y_coords, demands, m,max_iter)
# best_pos, best_score, convergence_curve, Capacity = sa.optimize()
