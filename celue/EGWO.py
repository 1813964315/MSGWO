import numpy as np

from capacity import capacity
from near import near
from population import initialization

class EGWO:
    def __init__(self, func,x_coords, y_coords, demands, m,num_wolves, max_iter, min_distance=0.001):
        self.func = func
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.alpha_pos = np.zeros(m*2)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(m*2)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(m*2)
        self.delta_score = float('inf')
        self.positions = initialization(self.num_wolves, self.m)
        self.convergence_curve = []
        self.min_distance = min_distance  # 最小距离约束

    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.num_wolves):
                fitness = self.func(self.positions[i], self.x_coords, self.y_coords, self.demands, self.m)
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()

            a = 2 - iter * (2 / self.max_iter)
            for i in range(self.num_wolves):
                fitness = self.func(self.positions[i], self.x_coords, self.y_coords, self.demands, self.m)
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()

            for i in range(self.num_wolves):
                r1, r2 = np.random.rand(2)
                A1 = 1.618 * a * r1 - a
                C1 = 0.01 * r2
                D_alpha = abs(C1 * self.alpha_pos - self.positions[i])
                X1 = self.alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(2)
                A2 = 1.618 * a * r1 - a
                C2 = 0.01 * r2
                D_beta = abs(C2 * self.beta_pos - self.positions[i])
                X2 = self.beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(2)
                A3 = 1.618 * a * r1 - a
                C3 = 0.01 * r2
                D_delta = abs(C3 * self.delta_pos - self.positions[i])
                X3 = self.delta_pos - A3 * D_delta

                self.positions[i] = (X1 + X2 + X3) / 3
                near(self.positions[i],self.m)
            # print(f"Iteration: {iter + 1}, Best Score: {self.alpha_score}")
            self.convergence_curve.append(self.alpha_score)
        # 分开输出充电站的位置和容量
        Capacity = capacity(self.alpha_pos, self.x_coords, self.y_coords, self.demands, self.m)
        print("*************EGWO****************")
        print("最优位置：", self.alpha_pos)
        print("最优容量：", Capacity)
        print("最优目标函数值：", self.alpha_score)
        return self.alpha_pos, self.alpha_score, self.convergence_curve,Capacity
