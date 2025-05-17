import heapq

import numpy as np


def NGWO(cec_fun,dim,imax,m):
    a1 = 2
    a2 = 0
    k1 = 0.01
    k2 = 0.001
    pop_fitness = np.zeros(m)
    pop = np.random.uniform(low=-100, high=100, size=(m, dim))  # 种群大小50
    #对种群个体进行初始化并计算对应适应度值
    for j in range(m):
        pop_fitness[j] = cec_fun(pop[j])

    #allbestpop,allbestfit分别存储种群在历史迭代过程中最优个体解及对应适应度
    allbestpop,allbestfit = pop[pop_fitness.argmin()].copy(),pop_fitness.min()

    #通过排序找出种群中适应度值最优的前三个个体，并获得它们的位置信息
    pop_fitness1 = pop_fitness.flatten()
    pop_fitness1 = pop_fitness1.tolist()
    three = list(map(pop_fitness1.index, heapq.nsmallest(3, pop_fitness1)))
    Xalpha = pop[three[0]]
    Xbeta = pop[three[1]]
    Xdelta = pop[three[2]]

    #his_bestfit存储每次迭代时种群历史适应度值最优的个体适应度
    his_bestfit=np.zeros(imax+1)
    his_bestfit[0] = allbestfit
    #开始训练
    for i in range(imax):
#         print("The iteration is:", i + 1)
        #对系数向量的计算参数a进行计算
        # iratio = i / imax
        # a = 2 * (1 - iratio)
        a = a1+(a2-a1)*((1-i/imax)**k1)**k2
        #对每个个体进行位置更新
        for j in range(m):
            #分别计算在适应度值最优的前三个个体的影响下，个体的位置移动量X1、X2、X3
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
            #计算个体移动后的位置及适应度值
            pop[j] = ((X1 + X2 + X3) / 3)*(1-i/imax)+(X1-pop[j])*(i/imax)
            pop_fitness[j] = cec_fun(pop[j])
        #对种群历史最优位置信息与适应度值进行更新
        if pop_fitness.min() < allbestfit:
            allbestfit = pop_fitness.min()
            allbestpop = pop[pop_fitness.argmin()].copy()
        #通过排序找出种群中适应度值最优的前三个个体，并获得它们的位置信息
        pop_fitness1 = pop_fitness.flatten()
        pop_fitness1 = pop_fitness1.tolist()
        three = list(map(pop_fitness1.index, heapq.nsmallest(3, pop_fitness1)))
        Xalpha = pop[three[0]]
        Xbeta = pop[three[1]]
        Xdelta = pop[three[2]]

        #存储当前迭代下的种群历史最优适应度值并输出
        his_bestfit[i+1] = allbestfit
        print(f"Iteration {i+1}: Best Fitness = {allbestfit}")
    return his_bestfit,allbestpop