import numpy as np
import matplotlib.pyplot as plt

from CSLP1.capacity import capacity
from CSLP1.near import near
from CSLP1.population import initialization


def cauchy_mutation(size):
    return np.random.standard_cauchy(size)

class BKA:
    def __init__(self, objective_function, x_coords, y_coords, demands, m, num_population, max_iter):
        self.objective_function = objective_function
        self.num_dimensions = m*2
        self.max_iter = max_iter
        self.num_population = num_population
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m

    def optimize(self):

        # 初始化种群
        population = initialization(self.num_population,self.m)
        fitness = np.apply_along_axis(self.objective_function, 1, population, self.x_coords, self.y_coords,
                                      self.demands, self.m)
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        convergence_curve = np.zeros(self.max_iter)
        p = 0.9  # 根据论文设置参数p

        for t in range(self.max_iter):
            n = 0.05 * np.exp(-2 * (t / self.max_iter) ** 2)

            for i in range(self.num_population):
                for j in range(self.num_dimensions):
                    r = np.random.rand()

                    if r < p:
                        # 攻击行为
                        population[i, j] = population[i, j] + n * (1 + np.sin(r)) * population[i, j] * np.random.randn()
                    else:
                        # 迁徙行为
                        random_idx = np.random.randint(0, self.num_population)
                        if fitness[i] < fitness[random_idx]:
                            population[i, j] = population[i, j] + cauchy_mutation(1)[0] * (population[i, j] - population[random_idx, j])
                        else:
                            population[i, j] = population[i, j] + cauchy_mutation(1)[0] * (best_position[j] - 2 * population[i, j])
                    near(population[i],self.m)

            fitness = np.apply_along_axis(self.objective_function, 1, population, self.x_coords, self.y_coords,
                                          self.demands,self.m)
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_position = population[current_best_idx].copy()

            convergence_curve[t] = best_fitness

            # 打印当前迭代次数和当前最优值
            # print(f"Iteration {t + 1}/{self.max_iter}, Current Best Fitness: {best_fitness}")

        # 分开输出充电站的位置和容量
        Capacity = capacity(best_position,self.x_coords, self.y_coords, self.demands, self.m)
        print("*************BKA****************")
        print("最优位置：", best_position)
        print("最优容量：", Capacity)
        print("最优目标函数值：", best_fitness)
        return best_position, best_fitness, convergence_curve,Capacity
