import numpy as np
import matplotlib.pyplot as plt


def levy_flight(Lambda, size):
    sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    step = u / np.abs(v) ** (1 / Lambda)
    return step


def HO(func, dim, max_iter, num_population, lb=-100, ub=100):
    # 初始化种群
    population = lb + (ub - lb) * np.random.rand(num_population, dim)
    fitness = np.apply_along_axis(func, 1, population)
    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.zeros(max_iter)

    for t in range(max_iter):
        for i in range(num_population):
            for j in range(dim):
                r = np.random.rand()

                if r < 0.5:
                    # 河马防御行为
                    population[i, j] = population[i, j] + np.random.rand() * levy_flight(1.5, 1)[0]
                else:
                    # 河马逃避捕食者行为
                    random_idx = np.random.randint(0, num_population)
                    if fitness[i] < fitness[random_idx]:
                        population[i, j] = population[i, j] + np.random.rand() * (
                                    population[i, j] - population[random_idx, j])
                    else:
                        population[i, j] = population[i, j] + np.random.rand() * (best_position[j] - population[i, j])

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