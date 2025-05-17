import numpy as np
import matplotlib.pyplot as plt
def BSLO(func, num_dimensions, iterations,num_leeches):
    # 初始化水蛭
    leeches = np.random.uniform(-10, 10, (num_leeches, num_dimensions))
    best_position = None
    best_value = float('inf')
    convergence_curve = []

    for _ in range(iterations):
        fitness = np.array([func(leech) for leech in leeches])
        if min(fitness) < best_value:
            best_value = min(fitness)
            best_position = leeches[np.argmin(fitness)]

        # 吸血和移动策略
        for i in range(num_leeches):
            if np.random.rand() < 0.5:  # 随机选择更新策略
                leeches[i] += np.random.normal(0, 1, num_dimensions)  # 随机走动模拟
            else:
                leeches[i] += (best_position - leeches[i]) * np.random.rand()  # 向最优解移动

        convergence_curve.append(best_value)
    print(best_value)
    return convergence_curve,best_position