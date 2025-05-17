# 参数设置
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
# n = 500
def data():
    import pandas as pd
    # 正确的方式读取Excel文件的第一个工作表
    # ******************************
    # excel_path = 'data/data1.xlsx'  # Excel文件的路径
    # sheet_name = 2  # 第一个工作表的索引
    # **********************************
    # excel_path = 'data/yantianqu.xlsx'
    excel_path = 'data/luohuqu.xlsx'
    sheet_name = 0  # 第一个工作表的索引
    # 使用read_excel读取指定工作表
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # 假设Excel文件中的列分别命名为 'X', 'Y', 'Demand'
    # 将列数据转换为NumPy数组
    x_coords = df['x'].values
    y_coords = df['y'].values
    demands = df['C'].values
    x = df['x1'].values
    y = df['y1'].values
    # 如果你只想保留第一个NaN值之前的所有元素，可以使用np.where找到NaN值的索引
    nan_index = np.where(np.isnan(x))[0]
    first_nan_index = nan_index[0] if len(nan_index) > 0 else len(x)
    x = x[:first_nan_index]
    nan_index = np.where(np.isnan(y))[0]
    first_nan_index = nan_index[0] if len(nan_index) > 0 else len(y)
    y = y[:first_nan_index]
    return x_coords, y_coords, demands,x,y
x_coords,y_coords,demands,x,y = data()
# print(x,y)