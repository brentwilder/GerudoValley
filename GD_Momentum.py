import sys
import csv
import rasterio
import numpy as np
import math

# Goal: optimize alpha using Adaptive Moment Estimation (Adam)

# Death Valley
# min elev = -86 meters
# Easy point = 36.2833290, -116.8716455
# Difficult point = 36.2691711, -117.0563655

# Dead Sea
# min elev = -428 meters
# very difficult point = 36.1588269°E 31.3953303°N 2,947.278 ft
# Difficult point = 35.1970443°E 31.5575025°N 2,652.192 ft

lat =31.3953303
lon =36.1588269
#tif = "E:\SDSU\MATH693A\FinalProject\srtm_13_05.tif"
tif = "E:\SDSU\MATH693A\FinalProject\srtm_44_06.tif"
output = "E:\SDSU\MATH693A\FinalProject\output.csv"

# Open the elevation dataset
src = rasterio.open(tif)
band = src.read(1)

def get_elevation(lat, lon):
    vals = src.index(lon, lat)
    return band[vals]

def compute_cost(theta):
    lat, lon = theta[0], theta[1]
    J = get_elevation(lat, lon)
    return J


def gradient_descent(theta, alpha, gamma, num_iters):
    J_history = np.zeros(shape=(num_iters, 3))
    velocity = [ 0, 0 ]

    for i in range(num_iters):

        try:
            cost = compute_cost(theta)

            elev1 = get_elevation(theta[0] + 0.001, theta[1])
            elev2 = get_elevation(theta[0] - 0.001, theta[1])
            elev3 = get_elevation(theta[0], theta[1] + 0.001)
            elev4 = get_elevation(theta[0], theta[1] - 0.001)
        except IndexError:
            print('The boundary of elevation map has been reached')
            break

        J_history[i] = [ cost, theta[0], theta[1] ]
        if cost <= -413: return theta, J_history

        lat_slope = elev1 / elev2 - 1
        lon_slope = elev3 / elev4 - 1 
        if lat_slope == float('inf'):
            return theta, J_history
        
        if lon_slope == float('inf'):
            return theta, J_history

        velocity[0] = gamma * velocity[0] + alpha * lat_slope
        velocity[1] = gamma * velocity[1] + alpha * lon_slope
        
        print('Update is', velocity[0])
        print('Update is', velocity[1])
        print('Elevation at', theta[0], theta[1], 'is', cost)

        theta[0][0] = theta[0][0] - velocity[0]
        theta[1][0] = theta[1][0] - velocity[1]

    return theta, J_history

theta = np.array([ [lat], [lon] ])
theta, J_history = gradient_descent(theta,0.01,0.90,10000)

with open(output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for weight in J_history:
        if weight[1] != 0 and weight[2] != 0:
            writer.writerow([ weight[1], weight[2],weight[0] ])
