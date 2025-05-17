import numpy as np


def capacity(positions, x_coords, y_coords, demands, m):
    station_capacity = np.zeros(m)
    total_distance = 0
    for i in range(len(x_coords)):
        distances = np.sqrt((positions[0::2] - x_coords[i]) ** 2 + (positions[1::2] - y_coords[i]) ** 2)
        nearest_station = np.argmin(distances)
        total_distance += distances[nearest_station]
        station_capacity[nearest_station] += demands[i]
    return station_capacity