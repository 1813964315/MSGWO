import numpy as np
from data import x_coords, y_coords
from near import near

def initialization(N,m):
    random_array = np.zeros((N, m*2))
    for i in range(N):  # 遍历每一行
        for j in range(m*2):
            random_array[i, j] = np.random.uniform(0,1)
    for i in range(N):
        near(random_array[i],m)
    return random_array

