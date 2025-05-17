import numpy as np

from canshu.capacity import capacity
from canshu.near import near
from canshu.population import initialization

class MSGWO:
    def __init__(self, func,x_coords, y_coords, demands, m, num_wolves, max_iter, min_distance=0.1):
        self.func = func
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.alpha_pos = np.zeros(self.m*2)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(self.m*2)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(self.m*2)
        self.delta_score = float('inf')
        self.positions = initialization(num_wolves, self.m)
        # print("chushi*******************************")
        # print(self.positions)
        self.convergence_curve = []
        self.min_distance = min_distance

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
            N = self.func(self.alpha_pos, self.x_coords, self.y_coords, self.demands, self.m) + self.func(self.beta_pos, self.x_coords, self.y_coords, self.demands, self.m) + self.func(self.delta_pos, self.x_coords, self.y_coords, self.demands, self.m) + 0.005
            while N =='inf':
                self.positions = initialization(self.num_wolves, self.m)
            w11 = self.func(self.alpha_pos, self.x_coords, self.y_coords, self.demands, self.m) / N
            w22 = self.func(self.beta_pos, self.x_coords, self.y_coords, self.demands, self.m) / N
            w33 = self.func(self.delta_pos, self.x_coords, self.y_coords, self.demands, self.m) / N

            a = np.maximum(0.01, 2 * np.exp(-0.1 * iter) * (1 + 0.3 * np.cos(iter))) + 2 * np.exp(-iter / self.max_iter)
            for i in range(self.num_wolves):
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_pos - self.positions[i])
                X1 = self.alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos - self.positions[i])
                X2 = self.beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos - self.positions[i])
                X3 = self.delta_pos - A3 * D_delta

                w = np.random.rand()
                r = np.random.rand()
                Z1 = w * ((X1 + X2 + X3) / 3) + (1 - w) * (r * (X1 - self.positions[i]) + (1 - r) * (X2 - self.positions[i]))
                Z2 = ((X1 + X2 + X3) / 3) + 0.5 * np.random.rand() * (self.alpha_pos - self.positions[i]) + 0.5 * np.random.rand() * (self.positions[np.random.randint(0, self.m)] - self.positions[i])
                Z3 = ((w11 * X1 + w22 * X2 + w33 * X3) / 3) * (1 - iter / self.max_iter) + (X1 - self.positions[i]) * (iter / self.max_iter)
                Z = [Z1, Z2, Z3]
                # print("调整位置***********************************")
                for z in Z:
                    near(z,self.m)
                Z_fitness = np.array([self.func(z, self.x_coords, self.y_coords, self.demands, self.m) for z in Z])
                best_idx = np.argmin(Z_fitness)
                self.positions[i] = Z[best_idx]
            # print(self.positions)
            # print(f"Iteration: {iter + 1}, Best fitness:{self.alpha_score}")
            self.convergence_curve.append(self.alpha_score)
        # 分开输出充电站的位置和容量
        Capacity = capacity(self.alpha_pos,self.x_coords, self.y_coords, self.demands, self.m)
        print("*************MSGWO****************")
        print("最优位置：", self.alpha_pos)
        print("最优容量：", Capacity)
        print("最优目标函数值：", self.alpha_score)
        return self.alpha_pos, self.alpha_score, self.convergence_curve,Capacity
