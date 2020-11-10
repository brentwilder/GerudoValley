import sys
import csv
import rasterio
import argparse
import numpy as np
import plotly.express as px
import math

# Goal: optimize alpha using Adaptive Moment Estimation (Adam)

# Death Valley
# min elev = -86 meters
# Easy point = 36.2833290, -116.8716455
# Difficult point = 36.2691711, -117.0563655

# Dead Sea
# min elev = -415 meters
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

a = 0.01
b_1 = 0.9
b_2 = 0.999
e=1e-8

def gradient_descent(theta, num_iters):
    J_history = np.zeros(shape=(num_iters, 3))
    m_t_lat=0
    m_t_lon=0

    v_t_lat=0
    v_t_lon=0

    for i in range(num_iters):
        i+=1
        cost = compute_cost(theta)
        try:
            # Fetch elevations at offsets in each dimension
            elev1 = get_elevation(theta[0] + 0.001, theta[1])
            elev2 = get_elevation(theta[0] - 0.001, theta[1])
            elev3 = get_elevation(theta[0], theta[1] + 0.001)
            elev4 = get_elevation(theta[0], theta[1] - 0.001)
        except IndexError:
            print('The boundary of elevation map has been reached')
            break
        J_history[i] = [ cost, theta[0], theta[1] ]
        if cost <= -413: return theta, J_history

        # Calculate slope
        lat_slope = elev1 / elev2 - 1
        lon_slope = elev3 / elev4 - 1 

        if lat_slope == float('inf'):
            return theta, J_history
        
        if lon_slope == float('inf'):
            return theta, J_history
    
        # Adam
        m_t_lat = b_1*m_t_lat+(1-b_1)*lat_slope
        m_t_lon = b_1*m_t_lon+(1-b_1)*lon_slope

        v_t_lat = b_1*v_t_lat+(1-b_2)*lat_slope*lat_slope
        v_t_lon = b_1*v_t_lon+(1-b_2)*lon_slope*lon_slope

        m_cap_t_lat = m_t_lat / (1-b_1**i)
        m_cap_t_lon = m_t_lon / (1-b_1**i)

        v_cap_t_lat = v_t_lat / (1-b_2**i)
        v_cap_t_lon = v_t_lon / (1-b_2**i)


        # Update variables
        theta[0][0] = theta[0][0] - (a*m_cap_t_lat) / (math.sqrt(v_cap_t_lat)+e)
        theta[1][0] = theta[1][0] - (a*m_cap_t_lon) / (math.sqrt(v_cap_t_lon)+e)

        print('lat slope is', lat_slope)
        print('lon slope is', lon_slope)
        print('Elevation at', theta[0], theta[1], 'is', cost)

    return theta, J_history

theta = np.array([ [lat], [lon] ])
theta, J_history = gradient_descent(theta,2000)

with open(output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for weight in J_history:
        if weight[1] != 0 and weight[2] != 0:
            writer.writerow([ weight[1], weight[2],weight[0] ])
