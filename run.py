
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
import opfunu # 参考文档：https://github.com/thieu1995/opfunu
import mealpy
from mealpy.swarm_based import WOA,GWO
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
fun_name = 'F1' #按需修改
year = '2017'   #按需修改

func_num = fun_name + year

dim = 20 # 维度，根据cec函数 选择对应维度

'''定义的 cec函数 '''
def cec_fun(x):
    
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim = dim)
    F = func.evaluate(x)
    return F

''' fit_func->目标函数, lb->下限, ub->上限 '''
problem_dict = {
    "fit_func": cec_fun,
    "lb": opfunu.get_functions_by_classname(func_num)[0](ndim = dim).lb.tolist(), 
    "ub": opfunu.get_functions_by_classname(func_num)[0](ndim = dim).ub.tolist(), 
    "minmax": "min",
} 
   
''' 调用优化算法 '''
epoch = 100 #最大迭代次数
pop_size = 50 #种群数量
''' 第一种方式，需：from mealpy.swarm_based import WOA,GWO '''
# woa_model = WOA.OriginalWOA(epoch, pop_size)
# gwo_model = GWO.OriginalGWO(epoch, pop_size)
''' 第二种方式，需：from mealpy import get_optimizer_by_name'''
woa_model = get_optimizer_by_name("OriginalWOA")(epoch, pop_size)
gwo_model = get_optimizer_by_name("OriginalGWO")(epoch, pop_size)

'''求解 cec函数 '''
woa_best_x, woa_best_f = woa_model.solve(problem_dict)
# print(f"woa最优解: {woa_best_x}, \nwoa最优函数值: {woa_best_f}")   
gwo_best_x, gwo_best_f = gwo_model.solve(problem_dict)
# print(f"gwo最优解: {gwo_best_x}, \ngwo最优函数值: {gwo_best_f}")  
# In[]: 
    
''' 
    绘制适应度曲线
    model.history.list_global_best_fit：适应度曲线
'''
plt.figure
# plt.semilogy(Curve,'r-',linewidth=2)
plt.plot(woa_model.history.list_global_best_fit,'r-',linewidth=2,label = 'WOA')
plt.plot(gwo_model.history.list_global_best_fit,'b-',linewidth=2,label = 'GWO')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.grid()
plt.title('Convergence curve: '+ 'cec' + year + '-' + fun_name + ', Dim=' + str(dim))
plt.legend()
plt.show()

# In[]: 
    
''' 绘制三维函数图 '''
# 仅修改n_space 和 show -> 可视化选择参数
opfunu.plot_3d(opfunu.get_functions_by_classname(func_num)[0](ndim = 2), n_space=500, show = True) 
plt.title('cec' + year + '-' + fun_name)
