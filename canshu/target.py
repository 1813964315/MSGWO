import numpy as np


def objective_function(positions, x_coords, y_coords, demands, m):
    station_capacity = np.zeros(m)
    # print(positions)
    total_distance = 0
    # print(len(x_coords))
    # print(len(positions))
    for i in range(len(x_coords)):
        distances = np.sqrt((positions[0::2] - x_coords[i]) ** 2 + (positions[1::2] - y_coords[i]) ** 2)
        nearest_station = np.argmin(distances)
        total_distance += distances[nearest_station]
        station_capacity[nearest_station] += demands[i]
    # 初始化充电站信息
    for i in range(m):
        if station_capacity[i] < 20:  # 确保容量不超过最xiao容量
            # print("容量不足")
            return float('inf')
    # print(positions)
    # print(station_capacity)
    # # print("返回总距离")
    return total_distance