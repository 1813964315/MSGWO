from ortools.linear_solver import pywraplp
import numpy as np
from CSLP1.data import data


def solve_with_ortools(x_coords, y_coords, demands, x, y, m, time_limit=5000):
    """使用OR-Tools的SCIP求解器"""
    n = len(x_coords)
    k = len(x)
    distances = np.sqrt((x_coords[:, None] - x) ** 2 + (y_coords[:, None] - y) ** 2)

    # 创建求解器
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP求解器未安装")

    # 变量
    is_selected = [solver.BoolVar(f"y_{j}") for j in range(k)]
    assignment = [[solver.BoolVar(f"x_{i}_{j}") for j in range(k)] for i in range(n)]

    # 目标函数
    objective = solver.Objective()
    for i in range(n):
        for j in range(k):
            objective.SetCoefficient(assignment[i][j], distances[i, j])
    objective.SetMinimization()

    # 约束条件
    for i in range(n):
        constraint = solver.RowConstraint(1, 1, f"assign_{i}")
        for j in range(k):
            constraint.SetCoefficient(assignment[i][j], 1)

    for i in range(n):
        for j in range(k):
            solver.Add(assignment[i][j] <= is_selected[j])

    solver.Add(sum(is_selected[j] for j in range(k)) == m)

    for j in range(k):
        constraint = solver.RowConstraint(0, solver.infinity(), f"capacity_{j}")
        for i in range(n):
            constraint.SetCoefficient(assignment[i][j], demands[i])
        constraint.SetCoefficient(is_selected[j], -20)

    # 参数设置
    solver.SetTimeLimit(time_limit * 1000)  # 毫秒为单位
    solver.EnableOutput()  # 显示求解日志

    # 求解
    status = solver.Solve()

    # 结果处理
    if status == pywraplp.Solver.OPTIMAL:
        print(f"最优值: {objective.Value():.2f}")
        selected = [j for j in range(k) if is_selected[j].solution_value() > 0.5]
        print(f"选择的充电站({len(selected)}个): {selected}")

        # 计算负载
        station_loads = {
            j: sum(demands[i] for i in range(n) if assignment[i][j].solution_value() > 0.5)
            for j in selected
        }
        print("充电站负载分布:", station_loads)
    else:
        print(f"求解未成功，状态: {status}")


# 使用数据
x_coords, y_coords, demands, x, y = data()
solve_with_ortools(x_coords, y_coords, demands, x, y, m=5)