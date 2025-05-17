# -*- coding: utf-8 -*-
"""
CEC 测试函数 系列
@author: 微信公众号：优化算法侠，Swarm-Opti

"""

from IPython import get_ipython
# get_ipython().magic('reset -sf') #清除所有变量
import numpy as np
from matplotlib import pyplot as plt

# In[]:
import opfunu  # 参考文档：https://github.com/thieu1995/opfunu
import mealpy
from APO import APO
from BSLO import BSLO
from MSGWO import MSGWO
from GWO import GWO
from HO import HO
from IVY import IVY
from BKA import BKA
from GA import GA
from NOA import NOA
from RBMO import RBMO
from SBOA import SBOA
# from mealpy.swarm_based import WOA, GWO
from mealpy import get_optimizer_by_name
from mealpy.evolutionary_based.GA import BaseGA
plt.rcParams['font.family'] = 'Times New Roman'

'''
适应度函数及维度dim的选择
cec函数名字格式：函数名+年份，比如要选择2022的F1函数，func_num = 'F1'+'2022'
cec2005：F1-F25, 可选 dim = 10, 30, 50
cec2008：F1-F7,  可选 2 <= dim <= 1000
cec2010：F1-F20, 可选 100 <= dim <= 1000
cec2013：F1-F28, 可选 dim = 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
cec2014：F1-F30, 可选 dim = 10, 20, 30, 50, 100
cec2015：F1-F15, 可选 dim = 10, 30
cec2017：F1-F29, 可选 dim = 2, 10, 20, 30, 50, 100
cec2019：F1-F10, 可选 dim: F1=9,F2=16,F3=18,其他=10
cec2020：F1-F10, 可选 dim = 2, 5, 10, 15, 20, 30, 50, 100
cec2021：F1-F10, 可选 dim = 2, 10, 20
cec2022：F1-F12, 可选 dim = 2, 10, 20

'''
fun_name = 'F4'  # 按需修改
year = '2017'  # 按需修改

func_num = fun_name + year

dim = 50  # 维度，根据cec函数 选择对应维度

'''定义的 cec函数 '''


def cec_fun(x):
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    F = func.evaluate(x)
    return F


''' fit_func->目标函数, lb->下限, ub->上限 '''
problem_dict = {
    "fit_func": cec_fun,
    "lb": opfunu.get_functions_by_classname(func_num)[0](ndim=dim).lb.tolist(),
    "ub": opfunu.get_functions_by_classname(func_num)[0](ndim=dim).ub.tolist(),
    "minmax": "min",
}

''' 调用优化算法 '''
epoch = 100  # 最大迭代次数
pop_size = 50  # 种群数量
''' 第二种方式，需：from mealpy import get_optimizer_by_name'''
woa_model = get_optimizer_by_name("OriginalWOA")(epoch, pop_size)
pso_model = get_optimizer_by_name("OriginalPSO")(epoch, pop_size)
ga_model = get_optimizer_by_name("BaseGA")(epoch, pop_size)

'''求解 cec函数 '''
woa_best_x, woa_best_f = woa_model.solve(problem_dict)
# print(f"woa最优解: {woa_best_x}, \nwoa最优函数值: {woa_best_f}")
pso_best_x, pso_best_f = pso_model.solve(problem_dict)
# ga_best_x, ga_best_f = ga_model.solve(problem_dict)

fitness_curve1, best_individual1 = GWO(cec_fun,dim,epoch,pop_size)
# print("**************NOA*************************")
# fitness_curve2, best_individual2 = NOA(cec_fun,dim,epoch,pop_size)
# print("**************SBOA************************")
# fitness_curve3, best_individual3 = SBOA(cec_fun,dim,epoch,pop_size)
print("***************HO*************************")
fitness_curve4, best_individual4 = HO(cec_fun,dim,epoch,pop_size)
print("***************BSLO*********************")
fitness_curve5, best_individual5 = BSLO(cec_fun,dim,epoch,pop_size)
print("***************RBMO*********************")
fitness_curve6, best_individual6 = RBMO(cec_fun,dim,epoch,pop_size)
print("***************APO*********************")
fitness_curve7, best_individual7 = APO(cec_fun,dim,epoch,pop_size)
print("***************IVY*********************")
fitness_curve8, best_individual8 = IVY(cec_fun,dim,epoch,pop_size)
print("***************BKA************************")
fitness_curve9, best_individual9 = BKA(cec_fun,dim,epoch,pop_size)
print("***************GA************************")
fitness_curve10, best_individual10 = GA(cec_fun,dim,epoch,pop_size)
print("***************MSGWO************************")
fitness_curve, best_individual = MSGWO(cec_fun,dim,epoch,pop_size)
print("******************************************")
# In[]:

''' 
    绘制适应度曲线
    model.history.list_global_best_fit：适应度曲线
'''
plt.figure(figsize=(12, 8))
# plt.semilogy(Curve,'r-',linewidth=2)
l = 3
plt.plot(woa_model.history.list_global_best_fit,'purple',linewidth=l,label = 'WOA')
plt.plot(pso_model.history.list_global_best_fit,'b-',linewidth=l,label = 'PSO')
# plt.plot(ga_model.history.list_global_best_fit,'yellow',linewidth=l,label = 'GA')
plt.plot(fitness_curve10,'yellow',linewidth=l,label = 'GA')
plt.plot(fitness_curve1, 'g-', linewidth=l, label='GWO')
plt.plot(fitness_curve4, 'orange', linewidth=l, label='HO')
plt.plot(fitness_curve5, 'cadetblue', linewidth=l, label='BSLO')
plt.plot(fitness_curve6, 'pink', linewidth=l, label='RBMO')
plt.plot(fitness_curve7, 'm', linewidth=l, label='APO')
plt.plot(fitness_curve8, 'brown', linewidth=l, label='IVY')
plt.plot(fitness_curve9, 'cyan', linewidth=l, label='BKA')
plt.plot(fitness_curve, 'r-', linewidth=l, label='MSGWO')
plt.xlabel('Iteration',fontsize=24)
plt.ylabel('Fitness',fontsize=24)
plt.grid()
plt.legend(fontsize=22)
# plt.grid(True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.title('Convergence curve: ' + 'cec' + year + '-' + fun_name + ', Dim=' + str(dim),fontsize=24)
plt.show()
#
# # In[]:
#
# ''' 绘制三维函数图 '''
# # 仅修改n_space 和 show -> 可视化选择参数
# opfunu.plot_3d(opfunu.get_functions_by_classname(func_num)[0](ndim=2), n_space=500, show=True)
# plt.title('cec' + year + '-' + fun_name)
