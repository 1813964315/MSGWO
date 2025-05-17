import numpy as np

from CSLP1.capacity import capacity
from CSLP1.near import near
from CSLP1.population import initialization

def update_position_eq3(x, best_position):
    return x + np.random.rand() * (best_position - x)

def update_position_eq4(x):
    return x + np.random.normal(0, 1, len(x))

def update_position_eq5(x, best_position):
    return x - np.random.rand() * (best_position - x)

def update_position_eq6(x):
    return x + np.random.uniform(-1, 1, len(x))

class RBMO:
    def __init__(self, objective_function,x_coords, y_coords, demands, m, n, T, epsilon=0.5, min_distance=0.001):
        self.objective_function = objective_function
        self.T = T
        self.n = n
        self.epsilon = epsilon
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
        self.min_distance = min_distance

    def optimize(self):
        positions = initialization(self.n, self.m)
        fitness = np.array([self.objective_function(pos, self.x_coords, self.y_coords, self.demands, self.m) for pos in positions])
        best_position = positions[np.argmin(fitness)].copy()
        best_value = np.min(fitness)
        convergence_curve = []

        t = 0
        while t < self.T:
            fitness = np.array([self.objective_function(pos, self.x_coords, self.y_coords, self.demands, self.m) for pos in positions])
            if np.min(fitness) < best_value:
                best_value = np.min(fitness)
                best_position = positions[np.argmin(fitness)].copy()

            for i in range(self.n):
                if np.random.rand() < self.epsilon:
                    positions[i] = update_position_eq3(positions[i], best_position)
                else:
                    positions[i] = update_position_eq4(positions[i])
                near(positions[i],self.m)

            convergence_curve.append(best_value)
            # print(f'Iteration {t + 1}/{self.T}, Best Fitness: {best_value}')
            t += 1

        # 分开输出充电站的位置和容量
        Capacity = capacity(best_position,self.x_coords, self.y_coords, self.demands, self.m)
        print("*************RBMO****************")
        print("最优位置：", best_position)
        print("最优容量：", Capacity)
        print("最优目标函数值：", best_value)
        return best_position, best_value, convergence_curve,Capacity
