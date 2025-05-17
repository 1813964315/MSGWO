import numpy as np


def RBMO(func, dim, max_iter, num_population, lb=-100, ub=100):
    # 初始化种群
    population = lb + (ub - lb) * np.random.rand(num_population, dim)
    fitness = np.apply_along_axis(func, 1, population)
    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.zeros(max_iter)

    for t in range(max_iter):
        p = np.random.randint(2, 6)  # 小组狩猎数量
        q = np.random.randint(10, num_population + 1)  # 集群狩猎数量
        CF = (1 - t / max_iter) * (2 * t / max_iter)

        # 食物搜索阶段
        if np.random.rand() < 0.5:
            # 小组搜索
            for i in range(num_population):
                if i != best_idx:
                    selected = np.random.choice(num_population, p, replace=False)
                    new_position = population[i] + 1 / p * np.sum(population[selected] - population[i],
                                                                  axis=0) * np.random.rand()
                    new_position = np.clip(new_position, lb, ub)
                    new_fitness = func(new_position)
                    if new_fitness < fitness[i]:
                        population[i] = new_position
                        fitness[i] = new_fitness
                        if new_fitness < best_fitness:
                            best_position = new_position.copy()
                            best_fitness = new_fitness
        else:
            # 集群搜索
            for i in range(num_population):
                if i != best_idx:
                    selected = np.random.choice(num_population, q, replace=False)
                    new_position = population[i] + 1 / q * np.sum(population[selected] - population[i],
                                                                  axis=0) * np.random.rand()
                    new_position = np.clip(new_position, lb, ub)
                    new_fitness = func(new_position)
                    if new_fitness < fitness[i]:
                        population[i] = new_position
                        fitness[i] = new_fitness
                        if new_fitness < best_fitness:
                            best_position = new_position.copy()
                            best_fitness = new_fitness

        # 攻击猎物阶段
        for i in range(num_population):
            if i != best_idx:
                if np.random.rand() < 0.5:
                    # 小组攻击
                    selected = np.random.choice(num_population, p, replace=False)
                    new_position = best_position + CF * 1 / p * np.sum(population[selected] - population[i],
                                                                       axis=0) * np.random.randn()
                else:
                    # 集群攻击
                    selected = np.random.choice(num_population, q, replace=False)
                    new_position = best_position + CF * 1 / q * np.sum(population[selected] - population[i],
                                                                       axis=0) * np.random.randn()
                new_position = np.clip(new_position, lb, ub)
                new_fitness = func(new_position)
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_position = new_position.copy()
                        best_fitness = new_fitness

        # 食物储存阶段
        for i in range(num_population):
            if fitness[i] > best_fitness:
                population[i] = population[i]
            else:
                population[i] = best_position

        # 记录收敛曲线
        convergence_curve[t] = best_fitness

        # 打印当前迭代次数和当前最优值
        print(f"Iteration {t + 1}/{max_iter}, Current Best Fitness: {best_fitness}")

    return convergence_curve, best_position
