import pulp
import numpy as np

from CSLP1.data import data


def solve_facility_location(x_coords, y_coords, demands, x, y, m):
    """
    解决设施选址问题：
    - x_coords, y_coords: 需求点的坐标
    - demands: 需求点的需求量
    - x, y: 备选站点的坐标
    - m: 充电站的数量
    """
    n = len(x_coords)  # 需求点数量
    k = len(x)  # 备选站点数量

    # 计算需求点到每个备选站点的距离
    distances = np.sqrt((x_coords[:, None] - x) ** 2 + (y_coords[:, None] - y) ** 2)

    # 创建线性规划模型
    model = pulp.LpProblem("Facility_Location", pulp.LpMinimize)

    # 决策变量
    # 1. 是否选择某个备选站点作为充电站 (0-1变量)
    is_selected = pulp.LpVariable.dicts("is_selected", range(k), cat="Binary")
    # 2. 每个需求点分配到哪个备选站点 (0-1变量)
    assignment = pulp.LpVariable.dicts("assignment", [(i, j) for i in range(n) for j in range(k)], cat="Binary")

    # 目标函数：最小化总距离
    model += pulp.lpSum(assignment[(i, j)] * distances[i, j] for i in range(n) for j in range(k))

    # 约束条件
    # 1. 每个需求点只能分配到一个备选站点
    for i in range(n):
        model += pulp.lpSum(assignment[(i, j)] for j in range(k)) == 1

    # 2. 如果需求点分配到某个备选站点，则该站点必须被选为充电站
    for i in range(n):
        for j in range(k):
            model += assignment[(i, j)] <= is_selected[j]

    # 3. 选择的充电站数量不超过m
    model += pulp.lpSum(is_selected[j] for j in range(k)) == m

    # 4. 每个充电站的容量不小于20
    for j in range(k):
        model += pulp.lpSum(demands[i] * assignment[(i, j)] for i in range(n)) >= 20 * is_selected[j]

    # 求解
    model.solve()
    print(f"状态: {pulp.LpStatus[model.status]}")
    print(f"最优目标函数值: {pulp.value(model.objective)}")

    # 输出结果
    selected_stations = [j for j in range(k) if pulp.value(is_selected[j]) == 1]
    print("选择的充电站索引:", selected_stations)
    print("选择的充电站位置:", [(x[j], y[j]) for j in selected_stations])

    # 输出每个需求点的分配情况
    print("需求点分配情况:")
    for i in range(n):
        assigned_station = [j for j in range(k) if pulp.value(assignment[(i, j)]) == 1][0]
        print(f"需求点 {i} 分配到充电站 {assigned_station}")


# 获取数据
x_coords, y_coords, demands, x, y = data()
m = 12  # 充电站数量

# 调用求解函数
solve_facility_location(x_coords, y_coords, demands, x, y, m)