# from ortools.linear_solver import pywraplp
# import numpy as np
#
#
# def generate_small_data():
#     """生成小规模测试数据"""
#     np.random.seed(42)  # 固定随机种子确保可重复性
#
#     # 10个需求点
#     n = 10
#     x_coords = np.random.uniform(0, 100, n)
#     y_coords = np.random.uniform(0, 100, n)
#     demands = np.random.randint(5, 20, n)  # 每个点需求5-20
#
#     # 5个备选站点
#     k = 5
#     x = np.random.uniform(0, 100, k)
#     y = np.random.uniform(0, 100, k)
#
#     return x_coords, y_coords, demands, x, y
#
#
# def solve_with_ortools(x_coords, y_coords, demands, x, y, m, time_limit=60):
#     """使用OR-Tools的SCIP求解器（小规模版）"""
#     n = len(x_coords)
#     k = len(x)
#     distances = np.sqrt((x_coords[:, None] - x) ** 2 + (y_coords[:, None] - y) ** 2)
#
#     # 创建求解器
#     solver = pywraplp.Solver.CreateSolver("SCIP")
#     if not solver:
#         raise RuntimeError("SCIP求解器未安装")
#
#     # 变量
#     is_selected = [solver.BoolVar(f"y_{j}") for j in range(k)]
#     assignment = [[solver.BoolVar(f"x_{i}_{j}") for j in range(k)] for i in range(n)]
#
#     # 目标函数（最小化总距离）
#     objective = solver.Objective()
#     for i in range(n):
#         for j in range(k):
#             objective.SetCoefficient(assignment[i][j], distances[i, j])
#     objective.SetMinimization()
#
#     # 约束条件
#     # 1. 每个需求点必须分配到一个站点
#     for i in range(n):
#         solver.Add(sum(assignment[i][j] for j in range(k)) == 1)
#
#     # 2. 只有被选中的站点才能服务需求点
#     for i in range(n):
#         for j in range(k):
#             solver.Add(assignment[i][j] <= is_selected[j])
#
#     # 3. 选择恰好m个站点
#     solver.Add(sum(is_selected[j] for j in range(k)) == m)
#
#     # 4. 每个站点的总需求至少为20
#     for j in range(k):
#         solver.Add(sum(demands[i] * assignment[i][j] for i in range(n)) >= 20 * is_selected[j])
#
#     # 参数设置
#     solver.SetTimeLimit(time_limit * 1000)  # 毫秒为单位
#     solver.EnableOutput()  # 显示求解日志
#
#     # 求解
#     print("\n开始求解...")
#     status = solver.Solve()
#
#     # 结果可视化
#     print("\n=== 求解结果 ===")
#     if status == pywraplp.Solver.OPTIMAL:
#         print(f"最优总距离: {objective.Value():.2f}")
#         selected = [j for j in range(k) if is_selected[j].solution_value() > 0.5]
#         print(f"选择的充电站({len(selected)}个): {selected}")
#
#         # 打印分配详情
#         print("\n需求点分配情况:")
#         for i in range(n):
#             for j in range(k):
#                 if assignment[i][j].solution_value() > 0.5:
#                     print(f"  需求点{i} (坐标({x_coords[i]:.1f}, {y_coords[i]:.1f}), 需求{demands[i]}) -> "
#                           f"站点{j} (坐标({x[j]:.1f}, {y[j]:.1f}), 距离{distances[i, j]:.1f}km)")
#
#         # 计算站点负载
#         print("\n充电站负载:")
#         for j in selected:
#             load = sum(demands[i] for i in range(n) if assignment[i][j].solution_value() > 0.5)
#             print(f"  站点{j}: 总负载={load}, 位置({x[j]:.1f}, {y[j]:.1f})")
#     else:
#         print(f"求解未成功，状态码: {status}")
#
#
# # 生成小规模数据
# x_coords, y_coords, demands, x, y = generate_small_data()
# print("=== 测试数据 ===")
# print(f"需求点数量: {len(x_coords)}")
# print(f"备选站点: {len(x)}")
# print(f"需求分布: {demands}")
# print(f"目标建站数: 2")
#
# # 求解（建2个站）
# solve_with_ortools(x_coords, y_coords, demands, x, y, m=2, time_limit=10)






from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt


def generate_small_data():
    """生成小规模测试数据"""
    np.random.seed(42)  # 固定随机种子

    # 10个需求点
    n = 10
    x_coords = np.random.uniform(0, 10, n)
    y_coords = np.random.uniform(0, 10, n)
    demands = np.random.randint(5, 15, n)  # 每个点需求5-15

    # 5个备选站点
    k = 5
    x = np.random.uniform(0, 10, k)
    y = np.random.uniform(0, 10, k)

    return x_coords, y_coords, demands, x, y


def plot_solution(x_coords, y_coords, demands, x, y, selected, assignments):
    """可视化结果"""
    plt.figure(figsize=(10, 8))

    # 绘制需求点（大小表示需求量）
    plt.scatter(x_coords, y_coords, c='red', s=demands * 10,
                alpha=0.7, label=f'Demand Points (n={len(x_coords)})')

    # 绘制备选站点
    plt.scatter(x, y, c='blue', marker='s', s=50,
                label='Candidate Sites')

    # 标记被选中的站点
    plt.scatter(x[selected], y[selected], c='green',
                marker='*', s=200, label='Selected Stations')

    # 绘制分配关系
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
    for j, color in zip(selected, colors):
        for i in np.where(assignments[:, j] > 0.5)[0]:
            plt.plot([x_coords[i], x[j]], [y_coords[i], y[j]],
                     c=color, linestyle='--', alpha=0.5)

    plt.title('Facility Location Solution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


def solve_small_example():
    """小规模示例求解"""
    # 生成数据
    x_coords, y_coords, demands, x, y = generate_small_data()
    m = 2  # 选择2个站点

    print("=== 测试数据 ===")
    print(f"需求点坐标:\n{np.column_stack((x_coords, y_coords))}")
    print(f"需求量: {demands}")
    print(f"备选站点坐标:\n{np.column_stack((x, y))}")

    # 计算距离矩阵
    distances = np.sqrt((x_coords[:, None] - x) ** 2 + (y_coords[:, None] - y) ** 2)

    # 创建模型
    model = ConcreteModel()
    model.I = RangeSet(0, len(x_coords) - 1)  # 需求点索引
    model.J = RangeSet(0, len(x) - 1)  # 备选站点索引

    # 变量
    model.y = Var(model.J, within=Binary)  # 是否选择站点
    model.x = Var(model.I, model.J, within=Binary)  # 分配关系

    # 目标函数：最小化总距离
    model.obj = Objective(
        expr=sum(distances[i, j] * model.x[i, j] for i in model.I for j in model.J),
        sense=minimize
    )

    # 约束条件
    model.assign_constr = Constraint(
        model.I,
        rule=lambda m, i: sum(m.x[i, j] for j in model.J) == 1
    )

    model.link_constr = Constraint(
        model.I, model.J,
        rule=lambda m, i, j: m.x[i, j] <= m.y[j]
    )

    # 修复：将 m 定义为一个参数
    model.m = Param(initialize=m)  # m 是选择的站点数

    model.station_constr = Constraint(
        rule=lambda m: sum(m.y[j] for j in model.J) == m.m  # 使用 m.m
    )

    model.capacity_constr = Constraint(
        model.J,
        rule=lambda m, j: sum(demands[i] * m.x[i, j] for i in model.I) >= 20 * m.y[j]
    )

    # 求解配置
    solver = SolverFactory('appsi_highs')
    solver.options = {
        'time_limit': 10,  # 10秒限制
        'mip_rel_gap': 0.05,  # 5%的gap
        'log_level': 'HIGH'  # 详细日志
    }

    print("\n开始求解...")
    results = solver.solve(model, tee=True)

    # 结果处理
    if results.solver.termination_condition == TerminationCondition.optimal:
        selected = [j for j in model.J if value(model.y[j]) > 0.5]
        assignments = np.array([[value(model.x[i, j]) for j in model.J] for i in model.I])

        print("\n=== 最优解 ===")
        print(f"选择的站点索引: {selected}")
        print(f"总距离: {value(model.obj):.2f}")

        # 打印分配详情
        print("\n分配关系:")
        for i in model.I:
            for j in model.J:
                if assignments[i, j] > 0.5:
                    print(f"  需求点{i} -> 站点{j}, 距离: {distances[i, j]:.2f}, 需求: {demands[i]}")

        # 计算负载
        print("\n站点负载:")
        for j in selected:
            load = sum(demands[i] for i in model.I if assignments[i, j] > 0.5)
            print(f"  站点{j}: 总负载={load}, 位置({x[j]:.2f}, {y[j]:.2f})")

        # 可视化
        plot_solution(x_coords, y_coords, demands, x, y, selected, assignments)
    else:
        print("\n求解失败:", results.solver.termination_condition)


# 运行示例
solve_small_example()