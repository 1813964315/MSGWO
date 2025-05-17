import numpy as np
import matplotlib.pyplot as plt

# 随机生成 100 个需求点
np.random.seed(42)  # For reproducibility
n = 500
x_coords = np.random.uniform(0, 1, n)
y_coords = np.random.uniform(0, 1, n)
demands = np.random.uniform(1, 10, n)  # 需求量在 1 到 10 之间

# 打印前几个需求点以确认
for i in range(10):
    print(f"需求点 {i + 1}: 坐标 ({x_coords[i]}, {y_coords[i]}), 需求量 {demands[i]}")


def objective_function(positions, x_coords, y_coords, demands, m, f_j, z_j, C_max):
    total_cost = 0
    station_capacity = np.zeros(m)

    # 初始化充电站信息
    for i in range(m):
        x, y, capacity = positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]
        if capacity > C_max:  # 确保容量不超过最大容量
            capacity = C_max
        total_cost += f_j + z_j * capacity
        station_capacity[i] = capacity

    total_distance = 0
    for i in range(len(x_coords)):
        distances = np.sqrt((positions[0::3] - x_coords[i]) ** 2 + (positions[1::3] - y_coords[i]) ** 2)
        nearest_station = np.argmin(distances)
        total_distance += distances[nearest_station]
        station_capacity[nearest_station] -= demands[i]
        if station_capacity[nearest_station] < 0:
            return float('inf')  # 如果容量不足，返回一个很大的成本值

    total_cost += total_distance
    return total_cost

def random_init(m, dim, low, high):
    """ 纯随机初始化 """
    return np.random.uniform(low, high, size=(m, dim))

def heuristic_init(m, dim, low, high):
    """ 启发式初始化，倾向于搜索空间的某个区域 """
    center = (low + high) / 2
    spread = (high - low) / 4  # 较小的扩展范围以集中于中心
    return np.random.uniform(center - spread, center + spread, size=(m, dim))

def mixed_init(m, dim, low, high):
    """ 混合初始化：一半随机，一半启发式 """
    half_m = m // 2
    pop_random = random_init(half_m, dim, low, high)
    pop_heuristic = heuristic_init(m - half_m, dim, low, high)
    return np.vstack((pop_random, pop_heuristic))


class GreyWolfOptimizer:
    def __init__(self, func, bounds, x_coords, y_coords, demands, m, f_j, z_j, C_max, num_wolves=5, max_iter=100):
        self.func = func
        self.bounds = bounds
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m
        self.f_j = f_j
        self.z_j = z_j
        self.C_max = C_max
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')
        self.positions = mixed_init(num_wolves, self.dim, bounds[:, 0], bounds[:, 1])
        self.convergence_curve = []

    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.num_wolves):
                fitness = self.func(self.positions[i], self.x_coords, self.y_coords, self.demands, self.m, self.f_j, self.z_j, self.C_max)
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            N = self.func(self.alpha_pos,self.x_coords, self.y_coords, self.demands, self.m, self.f_j, self.z_j, self.C_max) + self.func(self.beta_pos,self.x_coords, self.y_coords, self.demands, self.m, self.f_j, self.z_j, self.C_max) + self.func(self.delta_pos,self.x_coords, self.y_coords, self.demands, self.m, self.f_j, self.z_j, self.C_max) + 0.005
            w11 = self.func(self.alpha_pos,self.x_coords, self.y_coords, self.demands, self.m, self.f_j, self.z_j, self.C_max) / N
            w22 = self.func(self.beta_pos,self.x_coords, self.y_coords, self.demands, self.m, self.f_j, self.z_j, self.C_max) / N
            w33 = self.func(self.delta_pos,self.x_coords, self.y_coords, self.demands, self.m, self.f_j, self.z_j, self.C_max) / N
            # a = np.maximum(0.001, np.exp(-0.3 * iter/ self.max_iter) * (1 + 0.3 * np.cos(iter*50/(1.01**iter)))+np.random.rand()*self.m)
            # a = np.maximum(0.01, iter/n * np.exp(-0.1 * iter) * (1 + 0.3 * np.cos(iter))) + np.exp(-iter / self.max_iter)
            a = np.maximum(0.01, 2 * np.exp(-0.1 * iter) * (1 + 0.3 * np.cos(iter))) + 2 * np.exp(-iter / self.max_iter)
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i][j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i][j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i][j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    # self.positions[i][j] = (X1 + X2 + X3) / 3
                    w = np.random.rand()
                    r = np.random.rand()
                    Z1 = w * ((X1 + X2 + X3) / 3) + (1 - w) * (r * (X1 - self.positions[i][j]) + (1 - r) * (X2 -self.positions[i][j]))
                    Z2 = ((X1 + X2 + X3) / 3) + 0.5 * np.random.rand() * (self.alpha_pos[j] - self.positions[i][j]) + 0.5 * np.random.rand() * (
                            self.positions[np.random.randint(0, m - 1)][j] - self.positions[i][j])
                    Z3 = ((w11 * X1 + w22 * X2 + w33 * X3) / 3) * (1 - iter / self.max_iter) + (X1 - self.positions[i][j]) * (iter / self.max_iter)
                    f = self.positions[i].copy()
                    Z= [Z1, Z2, Z3]
                    FF = []
                    for x in Z:
                        f[j] = x
                        F = self.func(f,self.x_coords, self.y_coords, self.demands, self.m, self.f_j, self.z_j, self.C_max)
                        FF.append(F)
                    best_idx = np.argmin(FF)
                    self.positions[i][j] = Z[best_idx]
                    # 确保新的位置满足容量要求
            for i in range(self.num_wolves):
                station_demand = np.zeros(self.m)
                for j in range(len(self.x_coords)):
                    distances = np.sqrt((self.positions[i][0::3] - self.x_coords[j]) ** 2 + (
                                self.positions[i][1::3] - self.y_coords[j]) ** 2)
                    nearest_station = np.argmin(distances)
                    station_demand[nearest_station] += self.demands[j]

                for j in range(self.m):
                    if station_demand[j] > self.positions[i][j*3+2]:
                        self.positions[i][j*3+2] = station_demand[j]

                # 确保位置在边界内
                for j in range(self.dim):
                    if (j + 1) % 3 != 0:
                        self.positions[i][j] = np.clip(self.positions[i][j], self.bounds[j, 0], self.bounds[j, 1])

            # print(f"Iteration: {iter + 1}, Best Score: {self.alpha_score}")
            self.convergence_curve.append(self.alpha_score)

        return self.alpha_pos, self.alpha_score, self.convergence_curve


# 参数设置
m = 5  # 充电站数量
f_j = 1000  # 每个充电站的固定建设成本
z_j = 10  # 每单位容量的建设成本
C_max = 10 * n  # 每个充电站的最大容量
bounds = np.array([(0, 1), (0, 1), [n, C_max]] * m)  # 每个充电站的 x, y 坐标和容量的边界

# 实例化和优化
gwo = GreyWolfOptimizer(objective_function, bounds, x_coords, y_coords, demands, m, f_j, z_j, C_max, num_wolves=30, max_iter=100)
best_position, best_score, convergence_curve = gwo.optimize()

# 分开输出充电站的位置和容量
stations_positions = best_position.reshape((m, 3))[:, :2]
stations_capacities = best_position.reshape((m, 3))[:, 2]

print("最优位置：", stations_positions)
print("最优容量：", stations_capacities)
print("最优目标函数值：", best_score)

# 绘制收敛曲线
plt.plot(convergence_curve)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Convergence Curve')
plt.show()

# 绘制充电站和需求点的分配图
def plot_allocation(x_coords, y_coords, stations_positions):
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, c='blue', label='Demand Points')
    plt.scatter(stations_positions[:, 0], stations_positions[:, 1], c='red', marker='x', label='Charging Stations')
    for i in range(len(x_coords)):
        distances = np.sqrt((stations_positions[:, 0] - x_coords[i]) ** 2 + (stations_positions[:, 1] - y_coords[i]) ** 2)
        nearest_station = np.argmin(distances)
        plt.plot([x_coords[i], stations_positions[nearest_station, 0]], [y_coords[i], stations_positions[nearest_station, 1]], 'k-', lw=0.5)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title('Allocation of Demand Points to Charging Stations')
    plt.show()

plot_allocation(x_coords, y_coords, stations_positions)
