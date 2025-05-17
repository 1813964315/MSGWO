import numpy as np
from scipy.special import gamma

from CSLP1.near import near
from CSLP1.population import initialization
from CSLP1.capacity import capacity


class APO:
    def __init__(self, objective_function, x_coords, y_coords, demands, m, N, T, epsilon=0.5):
        self.objective_function = objective_function
        self.T = T
        self.N = N
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
        self.epsilon = epsilon

    # def initialize_population(self):
    #     lower_bound, upper_bound = np.array(self.bounds).T
    #     return np.random.uniform(lower_bound, upper_bound, (self.N, self.num_dimensions))

    def levy_flight(self, size, alpha=1.5):
        sigma = (gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
                 (gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        step = u / np.abs(v) ** (1 / alpha)
        return step

    def update_position_eq2(self, x, best_position, B):
        return x + B * (best_position - x)

    def update_position_eq5(self, x):
        return x + self.levy_flight(len(x))

    def update_position_eq10(self, x, best_position, B):
        return x + B * np.random.rand(len(x)) * (best_position - x)

    def update_position_eq11(self, x):
        return x + np.random.uniform(-1, 1, len(x))

    def update_position_eq13(self, x):
        return x + np.random.normal(0, 1, len(x))
    def optimize(self):
        # lower_bound, upper_bound = np.array(self.bounds).T
        populations = initialization(self.N,self.m)
        # print(populations)
        fitness = np.array([self.objective_function(ind, self.x_coords, self.y_coords, self.demands, self.m) for ind in populations])
        best_index = np.argmin(fitness)
        best_position = populations[best_index]
        best_value = fitness[best_index]
        convergence_curve = []

        t = 0
        while t < self.T:
            B = np.random.rand()
            if B > self.epsilon:
                for i in range(self.N):
                    Y = self.update_position_eq2(populations[i], best_position, B)
                    Z = self.update_position_eq5(Y)
                    near(Z,self.m)
                    new_fitness = self.objective_function(Z, self.x_coords, self.y_coords, self.demands, self.m)
                    if new_fitness < fitness[i]:
                        populations[i] = Z
                        fitness[i] = new_fitness
            else:
                for i in range(self.N):
                    W = self.update_position_eq10(populations[i], best_position, B)
                    Y = self.update_position_eq11(W)
                    Z = self.update_position_eq13(Y)
                    near(Z, self.m)
                    new_fitness = self.objective_function(Z, self.x_coords, self.y_coords, self.demands, self.m)
                    if new_fitness < fitness[i]:
                        populations[i] = Z
                        fitness[i] = new_fitness

            new_best_index = np.argmin(fitness)
            if fitness[new_best_index] < best_value:
                best_value = fitness[new_best_index]
                best_position = populations[new_best_index]

            convergence_curve.append(best_value)
            # print(best_position)
            # print(f'Iteration {t + 1}/{self.T}, Best Fitness: {best_value}')
            t += 1

        # 分开输出充电站的位置和容量
        Capacity = capacity(best_position,self.x_coords, self.y_coords, self.demands, self.m)
        print("*************APO****************")
        print("最优位置：", best_position)
        print("最优容量：", Capacity)
        print("最优目标函数值：", best_value)
        return best_position, best_value, convergence_curve,Capacity