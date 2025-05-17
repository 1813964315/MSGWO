import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

from CSLP1.capacity import capacity
from CSLP1.near import near
from CSLP1.population import initialization


def levy_flight(Lambda, size):
    sigma = (gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    step = u / np.abs(v) ** (1 / Lambda)
    return step

class HO:
    def __init__(self, objective_function,x_coords, y_coords, demands, m,num_wolves, max_iter):
        self.objective_function = objective_function
        self.max_iter = max_iter
        self.num_individuals = num_wolves
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
    def update_position_defensive(self, position):
        return position + np.random.rand(len(position)) * levy_flight(1.5, len(position))

    def update_position_escape(self, population, position, best_position, fitness, i):
        random_idx = np.random.randint(0, len(population))
        if fitness[i] < fitness[random_idx]:
            return position + np.random.rand(len(position)) * (position - population[random_idx])
        else:
            return position + np.random.rand(len(position)) * (best_position - position)

    def optimize(self):
        # 初始化种群
        population = initialization(self.num_individuals,self.m)
        fitness = np.apply_along_axis(self.objective_function, 1, population, self.x_coords, self.y_coords,
                                      self.demands, self.m)

        # 记录最优解
        best_fitness = np.min(fitness)
        best_position = population[np.argmin(fitness)].copy()

        # 用于记录每次迭代的最优适应度值
        convergence_curve = np.zeros(self.max_iter)

        # HO 算法主循环
        for t in range(self.max_iter):
            for i in range(self.num_individuals):
                r = np.random.rand()

                if r < 0.5:
                    # 河马防御行为
                    new_position = self.update_position_defensive(population[i])+np.random.rand() * levy_flight(1.5, 1)[0]
                else:
                    # 河马逃避捕食者行为
                    new_position = self.update_position_escape(population, population[i], best_position, fitness, i)
                near(new_position,self.m)
                # 计算新个体的适应度
                new_fitness = self.objective_function(new_position, self.x_coords, self.y_coords, self.demands, self.m)

                # 更新个体和最优解
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    population[i] = new_position

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_position = new_position

            # 记录当前迭代的最优适应度值
            convergence_curve[t] = best_fitness
            # print(f'Iteration {t + 1}/{self.max_iter}, Best Fitness: {best_fitness}')
        Capacity = capacity(best_position, self.x_coords, self.y_coords, self.demands, self.m)
        # 分开输出充电站的位置和容量
        print("*************HO****************")
        print("最优位置：", best_position)
        print("最优容量：", Capacity)
        print("最优目标函数值：", best_fitness)
        return best_position, best_fitness, convergence_curve,Capacity