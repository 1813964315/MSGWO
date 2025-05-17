import numpy as np
from scipy.special import gamma


def initialize_population(N, num_dimensions, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (N, num_dimensions))


def levy_flight(L, alpha=1.5):
    sigma = (gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
             (gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
    u = np.random.randn() * sigma
    v = np.random.randn()
    step = u / abs(v) ** (1 / alpha)
    return L * step


def update_position_aerial_search(X, Xr, L, alpha):
    R = np.round(0.5 * (0.05 + np.random.rand())) * np.random.normal(0, 1)
    return X + (X - Xr) * levy_flight(L, alpha) + R


def update_position_swooping(X, S):
    return X * np.tan((np.random.rand() - 0.5) * np.pi)


def update_position_gathering(W, F, L, Xr2, Xr3, rand):
    if rand >= 0.5:
        return W + F * levy_flight(L) * (Xr2 - Xr3)
    else:
        return W + F * (Xr2 - Xr3)


def update_position_intensifying_search(W, f):
    return W * (1 + f)


def update_position_avoiding(Z, F, L, Xr1, Xr2, beta, rand):
    if rand >= 0.5:
        return Z + F * levy_flight(L) * (Xr1 - Xr2)
    else:
        return Z + beta * (Xr1 - Xr2)


def APO(func, num_dimensions, T, N):
    # 根据论文设置参数
    F = 0.5
    L = 1.5
    alpha = 1.5
    beta = np.random.uniform(0, 1)
    lower_bound, upper_bound = -10, 10  # 论文中的边界条件

    X = initialize_population(N, num_dimensions, lower_bound, upper_bound)
    fitness = np.array([func(ind) for ind in X])
    best_index = np.argmin(fitness)
    best_position = X[best_index]
    best_value = fitness[best_index]
    convergence_curve = []

    for t in range(T):
        Xr = X[np.random.choice(range(X.shape[0]))]

        # 空中飞行阶段
        if np.random.rand() < 0.5:
            Y = update_position_aerial_search(X, Xr, L, alpha)
        else:
            Y = update_position_swooping(X, np.tan((np.random.rand() - 0.5) * np.pi))

        # 水下觅食阶段
        W = X.copy()
        for i in range(N):
            Xr2, Xr3 = X[np.random.choice(range(X.shape[0]), 2, replace=False)]
            if np.random.rand() < 0.5:
                W[i] = update_position_gathering(W[i], F, L, Xr2, Xr3, np.random.rand())
            else:
                f = 0.1 * (np.random.rand() - 1) * (T - t) / T
                W[i] = update_position_intensifying_search(W[i], f)

        # 避开捕食者阶段
        Z = X.copy()
        for i in range(N):
            Xr1, Xr2 = X[np.random.choice(range(X.shape[0]), 2, replace=False)]
            Z[i] = update_position_avoiding(Z[i], F, L, Xr1, Xr2, beta, np.random.rand())

        P = np.concatenate((Y, W, Z))
        fitness = np.array([func(ind) for ind in P])
        sorted_indices = np.argsort(fitness)
        X = P[sorted_indices[:N]]

        new_best_index = np.argmin(fitness)
        if fitness[new_best_index] < best_value:
            best_value = fitness[new_best_index]
            best_position = P[new_best_index]

        convergence_curve.append(best_value)
        print(f'Iteration {t + 1}/{T}, Best Fitness: {best_value}')

    print("*************APO****************")
    print("Best Position:", best_position)
    print("Best Fitness:", best_value)
    return convergence_curve, best_position