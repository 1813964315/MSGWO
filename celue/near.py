import numpy as np

from celue.data import x, y


def near(positions,m):
    # print("***************",positions)
    I = []
    for i in range(m):
        N = []
        for j in range(len(x)):
            N.append(np.sqrt((x[j] - positions[i * 2]) ** 2 + (y[j] - positions[i * 2 + 1]) ** 2))
        N = np.array(N)
        sorted_indices = np.argsort(N)
        # print(f"sorted_indices: {sorted_indices}")

        # 找到第一个未被选中的最小值索引
        for index in sorted_indices:
            if index not in I:
                # print(index)
                I.append(index)
                # print("near:", x_coords[index], y_coords[index])
                positions[i * 2] = x[index]
                positions[i * 2 + 1] = y[index]
                # print("***************",positions)
                break
