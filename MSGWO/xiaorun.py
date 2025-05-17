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
from MSGWO1 import MSGWO1
from MSGWO2 import MSGWO2
from MSGWO3 import MSGWO3
from MSGWO import MSGWO
from GWO import GWO
from HO import HO
from IVY import IVY
from NOA import NOA
from RBMO import RBMO
from SBOA import SBOA
# from mealpy.swarm_based import WOA, GWO
from mealpy import get_optimizer_by_name

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
fun_name = 'F20'  # 按需修改
year = '2017'  # 按需修改

func_num = fun_name + year

dim = 30  # 维度，根据cec函数 选择对应维度

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

'''求解 cec函数 '''
fitness_curve1, best_individual1 = GWO(cec_fun,dim,epoch,pop_size)
print("***************MSGWO1*************************")
fitness_curve2, best_individual2 = MSGWO1(cec_fun,dim,epoch,pop_size)
print("***************MSGWO2*********************")
fitness_curve3, best_individual3 = MSGWO2(cec_fun,dim,epoch,pop_size)
print("***************MSGWO3*******************")
fitness_curve4, best_individual4 = MSGWO3(cec_fun,dim,epoch,pop_size)
print("***************MSGWO************************")
fitness_curve, best_individual = MSGWO(cec_fun,dim,epoch,pop_size)
print("******************************************")
# In[]:

''' 
    绘制适应度曲线
    model.history.list_global_best_fit：适应度曲线
'''
plt.figure
plt.plot(fitness_curve1, 'g-', linewidth=2, label='GWO')
plt.plot(fitness_curve2, 'orange', linewidth=2, label='MSGWO1')
plt.plot(fitness_curve3, 'm', linewidth=2, label='MSGWO2')
plt.plot(fitness_curve4, 'brown', linewidth=2, label='MSGWO3')
plt.plot(fitness_curve, 'r-', linewidth=2, label='MSGWO')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.grid()
plt.title('Convergence curve: ' + 'cec' + year + '-' + fun_name + ', Dim=' + str(dim))
plt.legend()
plt.show()
#
# # In[]:
#
# ''' 绘制三维函数图 '''
# # 仅修改n_space 和 show -> 可视化选择参数
# opfunu.plot_3d(opfunu.get_functions_by_classname(func_num)[0](ndim=2), n_space=500, show=True)
# plt.title('cec' + year + '-' + fun_name)
