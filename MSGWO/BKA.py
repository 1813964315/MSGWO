import numpy as np
import matplotlib.pyplot as plt


def cauchy_mutation(size):
    return np.random.standard_cauchy(size)


def BKA(func, dim, max_iter, num_population, lb=-100, ub=100):
    # 初始化种群
    population = lb + (ub - lb) * np.random.rand(num_population, dim)
    fitness = np.apply_along_axis(func, 1, population)
    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.zeros(max_iter)
    p = 0.9  # 根据论文设置参数p

    for t in range(max_iter):
        n = 0.05 * np.exp(-2 * (t / max_iter) ** 2)

        for i in range(num_population):
            for j in range(dim):
                r = np.random.rand()

                if r < p:
                    # 攻击行为
                    population[i, j] = population[i, j] + n * (1 + np.sin(r)) * population[i, j] * np.random.randn()
                else:
                    # 迁徙行为
                    random_idx = np.random.randint(0, num_population)
                    if fitness[i] < fitness[random_idx]:
                        population[i, j] = population[i, j] + cauchy_mutation(1)[0] * (
                                    population[i, j] - population[random_idx, j])
                    else:
                        population[i, j] = population[i, j] + cauchy_mutation(1)[0] * (
                                    best_position[j] - 2 * population[i, j])

                population[i, j] = np.clip(population[i, j], lb, ub)

        fitness = np.apply_along_axis(func, 1, population)
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_position = population[current_best_idx].copy()

        convergence_curve[t] = best_fitness

        # 打印当前迭代次数和当前最优值
        print(f"Iteration {t + 1}/{max_iter}, Current Best Fitness: {best_fitness}")

    return convergence_curve, best_position