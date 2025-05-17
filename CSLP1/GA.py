import numpy as np
import matplotlib.pyplot as plt

from CSLP1.near import near
from CSLP1.population import initialization
from CSLP1.capacity import capacity


class GA:
    def __init__(self, objective_function, x_coords, y_coords, demands, m, population_size, generations, crossover_rate=0.7, mutation_rate=0.01):
        self.num_dimensions = m*2
        self.objective_function = objective_function
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.num_dimensions - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, individual):
        for i in range(self.num_dimensions):
            if np.random.rand() < self.mutation_rate:
                mutation_value = np.random.uniform(-10, 10)
                individual[i] += mutation_value
                lower_bound, upper_bound =0,1
                individual[i] = np.clip(individual[i], lower_bound, upper_bound)
        return individual

    def select_parents(self, population, fitness):
        total_fitness = np.sum(fitness)
        probabilities = fitness / total_fitness
        selected_indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=probabilities)
        return population[selected_indices]

    def optimize(self):
        # 初始化种群
        population = initialization(self.population_size,self.m)
        fitness = np.apply_along_axis(self.objective_function, 1, population, self.x_coords, self.y_coords, self.demands, self.m)

        # 记录最优解
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        convergence_curve = np.zeros(self.generations)

        for generation in range(self.generations):
            new_population = []

            # 选择父代
            selected_population = self.select_parents(population, 1 / (fitness + 1e-6))  # 选择适应度值较高的个体作为父代

            # 生成下一代
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                near(child1, self.m)
                near(child1, self.m)
                new_population.append(child1)
                new_population.append(child2)

            # 更新种群
            population = np.array(new_population)
            fitness = np.apply_along_axis(self.objective_function, 1, population, self.x_coords, self.y_coords, self.demands, self.m)

            # 更新最优解
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_position = population[best_idx].copy()

            convergence_curve[generation] = best_fitness

            # 打印当前代数和当前最优值
            # print(f"Generation {generation + 1}/{self.generations}, Best Fitness: {best_fitness}")

        # 分开输出充电站的位置和容量
        Capacity = capacity(best_position,self.x_coords, self.y_coords, self.demands, self.m)
        print("*************GA****************")
        print("最优位置：", best_position)
        print("最优容量：", Capacity)
        print("最优目标函数值：", best_fitness)
        return best_position, best_fitness, convergence_curve,Capacity
