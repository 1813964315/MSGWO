import numpy as np
import matplotlib.pyplot as plt

from CSLP1.capacity import capacity
from CSLP1.near import near
from CSLP1.population import initialization


class IVY:
    def __init__(self, objective_function,x_coords, y_coords, demands, m, Npop, Itermax):
        self.objective_function = objective_function
        self.Itermax = Itermax
        self.Npop = Npop
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
        self.D = m*2

    def optimize(self):
        Iter = 1
        convergence_curve = []

        # 初始化种群
        I = initialization(self.Npop,self.m)

        # 评估初始种群
        fitness = np.array([self.objective_function(ind, self.x_coords, self.y_coords, self.demands, self.m) for ind in I])
        sorted_indices = np.argsort(fitness)
        I = I[sorted_indices]
        I_best = I[0]
        f_best = fitness[sorted_indices[0]]

        # convergence_curve.append(f_best)

        while Iter <= self.Itermax:
            for i in range(self.Npop):
                li = I[i]
                ΔGvi = li

                β = (2 + np.random.rand()) / 2 + np.random.rand() ** 2
                ΔGvi = np.random.rand(self.D) * ΔGvi

                if self.objective_function(li, self.x_coords, self.y_coords, self.demands, self.m) < β * self.objective_function(I_best, self.x_coords,
                                                                                     self.y_coords, self.demands,
                                                                                     self.m):
                    I_new = li + np.random.rand(self.D) * (I[np.random.randint(self.Npop)] - li) + np.random.rand(
                        self.D) * ΔGvi
                else:
                    I_new = I_best * np.random.rand(self.D) + np.random.rand(self.D) * ΔGvi
                near(I_new, self.m)
                fitness_new = self.objective_function(I_new, self.x_coords, self.y_coords, self.demands, self.m)
                if fitness_new < fitness[i]:
                    I[i] = I_new
                    fitness[i] = fitness_new

            # 按适应度排序种群
            sorted_indices = np.argsort(fitness)
            I = I[sorted_indices]
            I_best = I[0]
            f_best = fitness[sorted_indices[0]]

            convergence_curve.append(f_best)

            Iter += 1
            # print(f'Iteration {Iter}/{self.Itermax}, Best Fitness: {f_best}')

        # 分开输出充电站的位置和容量
        Capacity = capacity(I_best,self.x_coords, self.y_coords, self.demands, self.m)
        print("*************IVY****************")
        print("最优位置：", I_best)
        print("最优容量：", Capacity)
        print("最优目标函数值：", f_best)
        return I_best, f_best, convergence_curve,Capacity