import numpy as np


def zhibiao(positions, x_coords, y_coords):
    total_distance = 0
    max = 0
    for i in range(len(x_coords)):
        distances = np.sqrt((positions[0::2] - x_coords[i]) ** 2 + (positions[1::2] - y_coords[i]) ** 2)
        if min(distances) > max:
            max = min(distances)
        nearest_station = np.argmin(distances)
        total_distance += distances[nearest_station]
    return total_distance,total_distance/len(x_coords), max