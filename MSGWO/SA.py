import numpy as np

def SA(func, dim, max_iter, initial_temp, cooling_rate, low=-100, high=100):
    # 初始化解
    current_solution = np.random.uniform(low, high, size=dim)
    current_fitness = func(current_solution)
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    temperature = initial_temp

    history_fitness = np.zeros(max_iter + 1)
    history_fitness[0] = best_fitness

    for i in range(max_iter):
        # 生成新解
        new_solution = current_solution + np.random.uniform(-1, 1, size=dim)
        new_solution = np.clip(new_solution, low, high)
        new_fitness = func(new_solution)

        # 判断是否接受新解
        if new_fitness < current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
        else:
            # 以一定概率接受较差解
            acceptance_prob = np.exp((current_fitness - new_fitness) / temperature)
            if np.random.rand() < acceptance_prob:
                current_solution = new_solution
                current_fitness = new_fitness

        # 更新最佳解
        if current_fitness < best_fitness:
            best_solution = current_solution.copy()
            best_fitness = current_fitness

        # 降低温度
        temperature *= cooling_rate

        history_fitness[i + 1] = best_fitness
        print(f"Iteration {i+1}: Best Fitness = {best_fitness}")

    return history_fitness, best_solution

# # 示例目标函数
# def example_func(x):
#     return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2
#
# # 示例用法
# dim = 3  # 维度
# max_iter = 1000  # 最大迭代次数
# initial_temp = 1000  # 初始温度
# cooling_rate = 0.99  # 降温速率
# low, high = -10, 10  # 搜索空间的下界和上界
#
# # 解决问题
# history_fitness, best_solution = SA(example_func, dim, max_iter, initial_temp, cooling_rate, low, high)
# print("最优解:", best_solution)
# print("最优目标值:", history_fitness[-1])
