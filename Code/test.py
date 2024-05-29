import numpy as np
import gurobipy as gp
from functions import *
from data_store import *

# Defining parameters
pi = 100 # Penalty for unmet demand
T = 1 # Time horizon 1 hour  
e = 4 # maximal number of open depots
p = 15 # maximal number of open launch points
Q_T = 2000 # capacity of the vehicle
c_SD = 10 # cost for using a small drone
c_LD = 20 # cost for using a large drone
v_T = 45 # speed of the vehicle: 45 km/hr
tau = 2 # setup time for small drone to deliver: 2 miniutes

S = 12
gp_map, gp_location = generate_customer_demand_map(k=40)

# convert array into dataframe 
# DF = pd.DataFrame(gp_map)
# DF.to_csv("data\scenario_1_map_gather_points.csv", index=False, header=False)

save_to_csv("GP", gp_map, gp_location, 1)

map_list , location_list = read_csv_files("GP", S)

print(map_list[0].shape)