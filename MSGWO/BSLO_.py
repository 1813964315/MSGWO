import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def levy_flight(Lambda, size):
    # 使用 Levy 飞行分布生成随机数
    sigma = (gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    step = u / np.abs(v) ** (1 / Lambda)
    return step


def BSLO(func, dim, max_iter, num_leeches, lb=-100, ub=100):
    # 初始化参数
    m = 0.8
    a = 0.97
    b = 0.001
    t1 = 20
    t2 = 20

    # 初始化种群
    Xall = lb + (ub - lb) * np.random.rand(num_leeches, dim)
    fitness = np.apply_along_axis(func, 1, Xall)
    best_idx = np.argmin(fitness)
    best_position = Xall[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.zeros(max_iter)

    # 主循环
    for t in range(max_iter):
        # 更新参数
        N1 = int(num_leeches * (m + (1 - m) * (t / max_iter) ** 2))
        N2 = num_leeches - N1

        for i in range(N1):
            for j in range(dim):
                PD = np.random.uniform(-1, 1)
                levy_step = levy_flight(1.5, 1)[0]  # 取数组中的一个标量值
                if PD >= 1:
                    # 探索阶段
                    if np.random.rand() < a:
                        Xall[i, j] += b * Xall[i, j] - levy_step
                    else:
                        Xall[i, j] += b * Xall[i, j] + levy_step
                else:
                    # 开发阶段
                    if np.random.rand() < a:
                        Xall[i, j] = best_position[j] + b * best_position[j] - levy_step
                    else:
                        Xall[i, j] = best_position[j] + b * best_position[j] + levy_step

        for i in range(N1, num_leeches):
            for j in range(dim):
                levy_step = levy_flight(1.5, 1)[0]  # 取数组中的一个标量值
                if np.random.rand() < 0.5:
                    Xall[i, j] += (t / max_iter) * np.abs(best_position[j] - Xall[i, j]) * levy_step
                else:
                    Xall[i, j] += (t / max_iter) * np.abs(best_position[j] - Xall[i, j]) * levy_step

        # 评估适应度
        fitness = np.apply_along_axis(func, 1, Xall)
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_position = Xall[current_best_idx].copy()

        # 重追踪策略
        if t > t1 and (t - t2 >= 0) and (len(fitness) > t - t2) and np.array_equal(fitness, fitness[t - t2]):
            Xall = lb + (ub - lb) * np.random.rand(num_leeches, dim)

        # 记录收敛曲线
        convergence_curve[t] = best_fitness
        print(best_fitness)
    return convergence_curve, best_position
