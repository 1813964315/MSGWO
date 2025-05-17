import numpy as np

from CSLP1.data import x, y


def near1(positions, m):
    I = []
    for i in range(m):
        N = []
        for j in range(len(x)):
            N.append(np.sqrt((x[j] - positions[i * 2])**2 + (y[j] - positions[i * 2 + 1])**2))
        N = np.array(N)
        sorted_indices = np.argsort(N)
        # print(f"sorted_indices: {sorted_indices}")

        # 找到第一个未被选中的最小值索引
        for index in sorted_indices:
            if index not in I:
                I.append(index)
                break

    return I

# 示例数据
# x_coords = np.random.rand(10)
# y_coords = np.random.rand(10)
# positions = np.random.rand(2 * 5)  # 假设有 5 个位置
# m = 5
#
# I = near(x_coords, y_coords, positions, m)
# print("Resulting indices:", I)
