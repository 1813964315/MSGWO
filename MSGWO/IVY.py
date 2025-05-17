import numpy as np
import matplotlib.pyplot as plt


def IVY(func, dim, max_iter, num_population, lb=-100, ub=100):
    # 初始化种群
    population = lb + (ub - lb) * np.random.rand(num_population, dim)
    fitness = np.apply_along_axis(func, 1, population)
    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.zeros(max_iter)

    for t in range(max_iter):
        # 计算增长速度
        growth_velocity = np.random.rand(num_population, dim) * (
                    np.random.randn(num_population, dim) * fitness[:, np.newaxis])

        # 更新种群位置
        for i in range(num_population):
            if i != best_idx:
                new_position = population[i] + growth_velocity[i]
                new_position = np.clip(new_position, lb, ub)  # 限制在边界内
                new_fitness = func(new_position)

                # 更新位置和适应度
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                # 更新全局最优
                if new_fitness < best_fitness:
                    best_position = new_position.copy()
                    best_fitness = new_fitness

        # 记录收敛曲线
        convergence_curve[t] = best_fitness

        # 打印当前迭代次数和当前最优值
        print(f"Iteration {t + 1}/{max_iter}, Current Best Fitness: {best_fitness}")

    return convergence_curve, best_position