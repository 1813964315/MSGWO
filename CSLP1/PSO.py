import numpy as np

from CSLP1.capacity import capacity
from CSLP1.near import near
from CSLP1.population import initialization


class PSO:
    def __init__(self, objective_function, x_coords, y_coords, demands, m, num_particles,
                 max_iter, w=0.5, c1=2, c2=2):
        self.objective_function = objective_function
        self.num_dimensions = m*2
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demands = demands
        self.m = m

    def initialize_particles(self):
        particles = initialization(self.num_particles,self.m)
        velocities = np.random.uniform(-10, 10, (self.num_particles, self.num_dimensions))
        return particles, velocities

    def optimize(self):
        # 初始化粒子和速度
        particles, velocities = self.initialize_particles()
        personal_best_positions = particles.copy()
        personal_best_fitness = np.apply_along_axis(self.objective_function, 1, personal_best_positions, self.x_coords,
                                                    self.y_coords, self.demands, self.m)

        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]

        convergence_curve = np.zeros(self.max_iter)

        for t in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                near(particles[i],self.m)
                fitness = self.objective_function(particles[i], self.x_coords, self.y_coords, self.demands, self.m)

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = particles[i].copy()

                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles[i].copy()

            convergence_curve[t] = global_best_fitness

            # 打印当前迭代次数和当前最优值
            # print(f"Iteration {t + 1}/{self.max_iter}, Best Fitness: {global_best_fitness}")
        Capacity = capacity(global_best_position, self.x_coords, self.y_coords, self.demands, self.m)
        # 分开输出充电站的位置和容量
        print("*************PSO****************")
        print("最优位置：", global_best_position)
        print("最优容量：", Capacity)
        print("最优目标函数值：", global_best_fitness)
        return global_best_position, global_best_fitness, convergence_curve,Capacity