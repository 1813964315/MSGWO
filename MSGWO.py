import numpy as np

def MSGWO(cec_fun,dim,imax,m):
    # 初始化种群
    pop = np.random.uniform(-100, 100, size=(m, dim))
    m = len(pop)
    pop_fitness = np.array([cec_fun(individual) for individual in pop])
    allbestpop, allbestfit = pop[np.argmin(pop_fitness)].copy(), pop_fitness.min()
    his_bestfit = np.zeros(imax + 1)
    his_bestfit[0] = allbestfit
    for i in range(imax):
        three = np.argsort(pop_fitness)[:3]
        Xalpha, Xbeta, Xdelta = pop[three[0]], pop[three[1]], pop[three[2]]
        a = np.maximum(0.01,2*np.exp(-0.1*i)*(1+0.3*np.cos(i)))
        for j in range(m):
            # 分别计算在适应度值最优的前三个个体的影响下，个体的位置移动量X1、X2、X3
            C1 = 2 * np.random.rand()
            Dalpha = np.abs(C1 * Xalpha - pop[j])
            A1 = 2 * a * np.random.rand() - a
            X1 = Xalpha - A1 * Dalpha

            C2 = 2 * np.random.rand()
            Dbeta = np.abs(C2 * Xbeta - pop[j])
            A2 = 2 * a * np.random.rand() - a
            X2 = Xbeta - A2 * Dbeta

            C3 = 2 * np.random.rand()
            Ddelta = np.abs(C3 * Xdelta - pop[j])
            A3 = 2 * a * np.random.rand() - a
            X3 = Xdelta - A3 * Ddelta

            # w = ((imax - i)/imax)*(w1-w2)+ w2
            w = np.random.rand()
            r = np.random.rand()
            Z1 = w * ((X1 + X2 + X3) / 3) + (1 - w) * (r * (Xalpha - pop[j]) + (1 - r) * (X2 - pop[j]))
            Z2 = ((X1 + X2 + X3) / 3) + 0.5 * np.random.rand() * (Xalpha - pop[j]) + 0.5 * np.random.rand() * (
                        pop[np.random.randint(0, m - 1)] - pop[j])
            N = cec_fun(Xalpha) + cec_fun(Xbeta) + cec_fun(Xdelta) + d
            w11 = cec_fun(Xalpha) / N
            w22 = cec_fun(Xbeta) / N
            w33 = cec_fun(Xdelta) / N
            Z3 = ((w11 * X1 + w22 * X2 + w33 * X3) / 3) * (1 - i / imax) + (X1 - pop[j]) * (i / imax)
            # Evaluate fitness of new positions
            Z = [Z1, Z2, Z3]
            Z_fitness = np.array([cec_fun(z) for z in Z])

            # Choose the best new position
            best_idx = np.argmin(Z_fitness)
            if Z_fitness[best_idx] < pop_fitness[j]:
                pop[j], pop_fitness[j] = Z[best_idx], Z_fitness[best_idx]
                if Z_fitness[best_idx] < allbestfit:
                    allbestfit, allbestpop = Z_fitness[best_idx], Z[best_idx].copy()

        his_bestfit[i + 1] = allbestfit
        print(f"Iteration {i+1}: Best Fitness = {allbestfit}")
    print(allbestfit)
    return his_bestfit, allbestpop
