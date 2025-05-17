import numpy as np

def initialize_population(N, num_dimensions, bounds):
    lower_bound, upper_bound = np.array(bounds).T
    return np.random.uniform(lower_bound, upper_bound, (N, num_dimensions))

def select_parents(population, fitness):
    probabilities = fitness / np.sum(fitness)
    parents_idx = np.random.choice(np.arange(len(population)), size=len(population), p=probabilities)
    return population[parents_idx]

def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    else:
        child1, child2 = parent1, parent2
    return child1, child2

def mutate(individual, mutation_rate, bounds):
    lower_bound, upper_bound = np.array(bounds).T
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(lower_bound[i], upper_bound[i])
    return individual

def GA(func, num_dimensions, T, N, mutation_rate=0.1, crossover_rate=0.1):
    bounds = [(-100, 100)] * num_dimensions
    # 初始化
    populations = initialize_population(N, num_dimensions, bounds)
    fitness = np.array([func(ind) for ind in populations])
    best_index = np.argmin(fitness)
    best_position = populations[best_index]
    best_value = fitness[best_index]
    convergence_curve = []

    t = 0
    while t < T:
        selected_parents = select_parents(populations, fitness)
        next_population = []

        for i in range(0, N, 2):
            parent1, parent2 = selected_parents[i], selected_parents[i+1]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            next_population.append(mutate(child1, mutation_rate, bounds))
            next_population.append(mutate(child2, mutation_rate, bounds))

        populations = np.array(next_population)
        fitness = np.array([func(ind) for ind in populations])
        new_best_index = np.argmin(fitness)
        if fitness[new_best_index] < best_value:
            best_value = fitness[new_best_index]
            best_position = populations[new_best_index]

        convergence_curve.append(best_value)
        print(f'Generation {t + 1}/{T}, Best Fitness: {best_value}')
        t += 1

    print("*************GA****************")
    print("Best Position:", best_position)
    print("Best Fitness:", best_value)
    return convergence_curve, best_position