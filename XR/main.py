import numpy as np
from matplotlib import pyplot as plt

from CSLP.GWO import GWO
from CSLP.MSGWO import MSGWO
from CSLP.target import objective_function
from MSGWO1 import MSGWO1
from MSGWO2 import MSGWO2
from MSGWO3 import MSGWO3

n = 500
m = 5  # 充电站数量
f_j = 1000  # 每个充电站的固定建设成本
z_j = 10  # 每单位容量的建设成本
C_max = 10*n  # 每个充电站的最大容量
num_wolves=30
max_iter=100
bounds = np.array([[0, 1], [0, 1], [n, C_max]] * m)  # 每个充电站的 x, y 坐标和容量的边界
x_coords = np.random.uniform(0, 1, n)
y_coords = np.random.uniform(0, 1, n)
demands = np.random.uniform(1, 10, n)  # 需求量在 1 到 10 之间

# 打印前几个需求点以确认
# for i in range(10):
#     print(f"需求点 {i + 1}: 坐标 ({x_coords[i]}, {y_coords[i]}), 需求量 {demands[i]}")

# 实例化和优化
gwo = GWO(objective_function, bounds, x_coords, y_coords, demands, m, f_j, z_j, C_max, num_wolves, max_iter)
msgwo = MSGWO(objective_function, bounds, x_coords, y_coords, demands, m, f_j, z_j, C_max, num_wolves, max_iter)
msgwo1 = MSGWO1(objective_function, bounds, x_coords, y_coords, demands, m, f_j, z_j, C_max, num_wolves, max_iter)
msgwo2 = MSGWO2(objective_function, bounds, x_coords, y_coords, demands, m, f_j, z_j, C_max, num_wolves, max_iter)
msgwo3 = MSGWO3(objective_function, bounds, x_coords, y_coords, demands, m, f_j, z_j, C_max, num_wolves, max_iter)

# 优化
best_position1, best_score1, convergence_curve1 = gwo.optimize()
best_position2, best_score2, convergence_curve2 = msgwo1.optimize()
best_position3, best_score3, convergence_curve3 = msgwo2.optimize()
best_position4, best_score4, convergence_curve4 = msgwo3.optimize()
best_position, best_score, convergence_curve = msgwo.optimize()


stations_positions = best_position.reshape((m, 3))[:, :2]
stations_capacities = best_position.reshape((m, 3))[:, 2]
# 绘制收敛曲线
plt.plot(convergence_curve)
plt.plot(convergence_curve)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Convergence Curve')
plt.show()

# # 绘制充电站和需求点的分配图
# def plot_allocation(x_coords, y_coords, stations_positions):
#     plt.figure(figsize=(10, 8))
#     plt.scatter(x_coords, y_coords, c='blue', label='Demand Points')
#     plt.scatter(stations_positions[:, 0], stations_positions[:, 1], c='red', marker=',', label='Charging Stations')
#     for i in range(len(x_coords)):
#         distances = np.sqrt((stations_positions[:, 0] - x_coords[i]) ** 2 + (stations_positions[:, 1] - y_coords[i]) ** 2)
#         nearest_station = np.argmin(distances)
#         plt.plot([x_coords[i], stations_positions[nearest_station, 0]], [y_coords[i], stations_positions[nearest_station, 1]], 'k-', lw=0.5)
#
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.legend()
#     plt.title('Allocation of Demand Points to Charging Stations')
#     plt.show()
#
# plot_allocation(x_coords, y_coords, stations_positions)