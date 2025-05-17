import numpy as np

from CSLP1.capacity import capacity
from CSLP1.near import near
from CSLP1.population import initialization

class WOA:
    def __init__(self, objective_function, x_coords, y_coords, demands, m, n, T, min_distance=0.001):
        self.objective_function = objective_function
        self.T = T
        self.n = n
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
        self.min_distance = min_distance

    def optimize(self):
        positions = initialization(self.T, self.m)
        fitness = np.array([self.objective_function(pos, self.x_coords, self.y_coords, self.demands, self.m) for pos in positions])
        best_position = positions[np.argmin(fitness)].copy()
        best_fitness = np.min(fitness)
        convergence_curve = []

        for t in range(self.T):
            a = 2 - t * (2 / self.T)
            a2 = -1 + t * ((-1) / self.T)

            for i in range(self.n):
                r1 = np.random.rand()
                r2 = np.random.rand()

                A = 2 * a * r1 - a
                C = 2 * r2

                p = np.random.rand()
                b = 1
                l = (a2 - 1) * np.random.rand() + 1

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * best_position - positions[i])
                        positions[i] = best_position - A * D
                    else:
                        rand_leader_index = np.random.randint(self.n)
                        X_rand = positions[rand_leader_index]
                        D = abs(C * X_rand - positions[i])
                        positions[i] = X_rand - A * D
                else:
                    distance_to_leader = abs(best_position - positions[i])
                    positions[i] = distance_to_leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + best_position
                near(positions[i],self.m)

            fitness = np.array([self.objective_function(pos, self.x_coords, self.y_coords, self.demands, self.m) for pos in positions])
            if np.min(fitness) < best_fitness:
                # print("改变：{argmin(fitness)}")
                best_fitness = np.min(fitness)
                best_position = positions[np.argmin(fitness)].copy()

            convergence_curve.append(best_fitness)
            Capacity = capacity(best_position, self.x_coords, self.y_coords, self.demands, self.m)
            # print(f'Iteration {t + 1}/{self.T}, Best Fitness: {best_fitness},best_p:{best_position},Capacity:{Capacity}')
        Capacity = capacity(best_position,self.x_coords, self.y_coords, self.demands, self.m)
        # 分开输出充电站的位置和容量
        print("*************WOA****************")
        print("最优位置：", best_position)
        print("最优容量：", Capacity)
        print("最优目标函数值：", best_fitness)
        return best_position, best_fitness, convergence_curve,Capacity
