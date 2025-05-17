import numpy as np

from CSLP1.capacity import capacity
from CSLP1.near import near
from CSLP1.population import initialization


class BSLO:
    def __init__(self, objective_function,x_coords, y_coords, demands, m, num_leeches, iterations):
        self.objective_function = objective_function
        self.num_dimensions = m*2
        self.iterations = iterations
        self.num_leeches = num_leeches
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
    def optimize(self):
        leeches = initialization(self.num_leeches,self.m)
        fitness = np.array(
            [self.objective_function(leech, self.x_coords, self.y_coords, self.demands, self.m) for leech in leeches])
        best_position = leeches[np.argmin(fitness)].copy()
        best_value = np.min(fitness)
        # best_value = float('inf')
        convergence_curve = []

        for _ in range(self.iterations):
            # 吸血和移动策略
            fitness = np.array([self.objective_function(leech, self.x_coords, self.y_coords, self.demands, self.m) for leech in leeches])
            if min(fitness) < best_value:
                best_value = min(fitness)
                best_position = leeches[np.argmin(fitness)].copy()
            for i in range(self.num_leeches):
                if np.random.rand() < 0.5:  # 随机选择更新策略
                    leeches[i] += np.random.normal(0, 1, self.num_dimensions)  # 随机走动模拟
                else:
                    leeches[i] += (best_position - leeches[i]) * np.random.rand()  # 向最优解移动
                near(leeches[i], self.m)
            convergence_curve.append(best_value)
            # print(f'Iteration {_ + 1}/{self.iterations}, Best Fitness: {best_value}')
        # 分开输出充电站的位置和容量
        Capacity = capacity(best_position,self.x_coords, self.y_coords, self.demands, self.m)
        print("*************BSLO****************")
        print("最优位置：", best_position)
        print("最优容量：", Capacity)
        print("最优目标函数值：", best_value)
        return best_position, best_value, convergence_curve,Capacity
