from pyomo.environ import *
import numpy as np
from CSLP1.data import data  # 确保这个模块和函数存在

def solve_with_highs(x_coords, y_coords, demands, x, y, m, time_limit=10000):
    """使用Pyomo+HiGHS求解设施选址问题"""
    n = len(x_coords)
    k = len(x)
    distances = np.sqrt((x_coords[:, None] - x) ** 2 + (y_coords[:, None] - y) ** 2)

    model = ConcreteModel()

    # 集合
    model.I = RangeSet(0, n - 1)  # 需求点
    model.J = RangeSet(0, k - 1)  # 备选站点

    # 变量
    model.y = Var(model.J, within=Binary)  # 是否选择站点
    model.x = Var(model.I, model.J, within=Binary)  # 分配关系

    # 目标函数
    def obj_rule(model):
        return sum(distances[i, j] * model.x[i, j] for i in range(n) for j in range(k))

    model.obj = Objective(rule=obj_rule, sense=minimize)

    # 约束条件
    def assign_rule(model, i):
        return sum(model.x[i, j] for j in range(k)) == 1

    model.assign_constr = Constraint(range(n), rule=assign_rule)

    def link_rule(model, i, j):
        return model.x[i, j] <= model.y[j]

    model.link_constr = Constraint(range(n), range(k), rule=link_rule)

    def station_count_rule(model):
        return sum(model.y[j] for j in range(k)) == m

    model.station_constr = Constraint(rule=station_count_rule)

    def capacity_rule(model, j):
        return sum(demands[i] * model.x[i, j] for i in range(n)) >= 20 * model.y[j]

    model.capacity_constr = Constraint(range(k), rule=capacity_rule)

    # 求解配置
    solver = SolverFactory('appsi_highs')  # 使用HiGHS接口
    solver.options['time_limit'] = time_limit
    solver.options['mip_rel_gap'] = 0.05  # 5%的Gap

    results = solver.solve(model, tee=True)  # tee=True显示求解过程

    # 结果处理
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("最优解找到！")
        selected = [j for j in model.J if value(model.y[j]) > 0.5]
        print(f"选择的充电站({len(selected)}个): {selected}")
        print(f"总成本: {value(model.obj):.2f}")

        # 计算负载分布
        load_dist = {
            j: sum(demands[i] for i in range(n) if value(model.x[i, j]) > 0.5)
            for j in selected
        }
        print("充电站负载:", load_dist)
    else:
        print(f"求解终止状态: {results.solver.termination_condition}")


# 使用数据
x_coords, y_coords, demands, x, y = data()
solve_with_highs(x_coords, y_coords, demands, x, y, m=8)