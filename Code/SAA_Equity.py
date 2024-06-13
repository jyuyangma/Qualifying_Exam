import numpy as np
import gurobipy as gp
from functions import *
from data_store import *
from Class_UAV import *
import time

def SAA_equity_constr( set_NO , params , earthquake_info = None):

    """
    This function is designed to solve the sample average approximation algorithm for the UAV routing problem.
    The input parameters contain following:
    - num_scenarios: number of scenarios
    - n_scenarios: number of scenarios
    - pi: Penalty for unmet demand
    - T: Time horizon hour
    - e: maximal number of open depots
    - p: maximal number of open launch points
    - Q_T: capacity of the vehicle
    - c_SD: cost for using a small drone
    - c_LD: cost for using a large drone
    - v_T: speed of the vehicle: 45 km/hr
    - tau: setup time for small drone to deliver: 2 miniutes
    - K_SD: maximal number of small drones
    - K_LD: maximal number of large drones
    - num_candidates_depots: number of candidate depots
    - num_candidates_launch_points: number of candidate launch points
    - num_gathering_points: number of gathering points
    - printlevel: 1 for printing the results
    """

    S = params.n_scenarios
    pi = params.pi
    T = params.T
    e = params.e
    p = params.p
    Q_T = params.Q_T
    c_SD = params.c_SD
    c_LD = params.c_LD
    v_T = params.v_T
    tau = params.tau
    K_SD = params.K_SD
    K_LD = params.K_LD
    num_candidates_depots = params.num_candidates_depots
    num_candidates_launch_points = params.num_candidates_launch_points
    num_gathering_points = params.num_gathering_points
    printlevel = params.printlevel

    if printlevel > 0:
        printFlag = True
    else:
        printFlag = False

    magnitude_list = earthquake_info[0]
    depth_list = earthquake_info[1]

    gp_demand_and_location, D_location, LP_location , earthquake_location = generate_all_locations( S=S , num_GP=num_gathering_points , cand_D=num_candidates_depots , cand_LP=num_candidates_launch_points , size=(100,100))
    gp_demand = gp_demand_and_location[0]
    gp_location = gp_demand_and_location[1]

    for s in range(S):
        save_to_csv(set_NO ,"GP", s + 1, gp_demand, gp_location)
        save_to_csv(set_NO ,"DP", s + 1, None, D_location)
        save_to_csv(set_NO ,"LP", s + 1, None, LP_location)
        wind_speed , wind_angle = generate_wind_speed_and_angle()
        wind_array = np.array([[wind_speed],[wind_angle]])
        save_to_csv(set_NO ,"Wind", s + 1, None, None, wind_array)
        save_to_csv(set_NO ,"EQ", s + 1, demand_array=None, location_array=earthquake_location[s])

    if printFlag:
        print(f"Data arrays have been saved\n")

    demand_list , GP_location_list = read_csv_files(set_NO,"GP", S)
    DP_location_list = read_csv_files(set_NO,"DP", S)
    LP_location_list = read_csv_files(set_NO,"LP", S)
    weather_list = read_csv_files(set_NO,"Wind", S)
    EQ_location_list = read_csv_files(set_NO,"EQ", S)

    if printFlag:
        print(f"Data arrays have been read\n")

    smalldrones = SmallDrone("SD")
    largedrones = LargeDrone("LD")

    Q_SD = smalldrones.max_payload
    Q_LD = largedrones.max_payload

    if printFlag:
        print("SAA Algorithm is running...\n")
    current_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SAA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    m = gp.Model("SAA")
    m.setParam('OutputFlag', 0)

    # Add first stage variables
    u = m.addVars(num_candidates_depots, vtype=gp.GRB.BINARY, name="depot")
    v = m.addVars(num_candidates_launch_points, vtype=gp.GRB.BINARY, name="launch_point")
    w = m.addVars(num_candidates_depots, num_candidates_launch_points, vtype=gp.GRB.BINARY, name="LP_to_DP")

    # Add second stage variables
    g_SD = m.addVars(S, K_SD, vtype=gp.GRB.BINARY, name="small_drone_usage")
    g_LD = m.addVars(S, K_LD, vtype=gp.GRB.BINARY, name="large_drone_usage")
    x = m.addVars(S, num_gathering_points, num_candidates_launch_points + num_candidates_depots, K_SD, vtype=gp.GRB.BINARY, name="small_drone_assignment")
    y = m.addVars(S, num_gathering_points, num_candidates_depots, K_LD, vtype=gp.GRB.BINARY, name="large_drone_assignment")
    o = m.addVars(S, num_gathering_points, lb=0, vtype=gp.GRB.CONTINUOUS, name="unmet_demand")

    # Add objective function
    m.setObjective( (gp.quicksum(pi * o[s,i]  for s in range(S) for i in range(num_gathering_points)) +
                    gp.quicksum(c_SD * g_SD[s,k] for s in range(S) for k in range(K_SD)) +
                    gp.quicksum(c_LD * g_LD[s,k] for s in range(S) for k in range(K_LD)))/S,
                    gp.GRB.MINIMIZE)
    
    # Add first stage constraints
    m.addConstr(gp.quicksum(u[d] for d in range(num_candidates_depots)) <= e)
    m.addConstr(gp.quicksum(v[l] for l in range(num_candidates_launch_points)) <= p)
    for l in range(num_candidates_launch_points):
        m.addConstr(gp.quicksum(w[d,l] for d in range(num_candidates_depots)) == v[l])

    GP_location = GP_location_list[0]
    DP_location = DP_location_list[0]
    LP_location = LP_location_list[0]

    GP_DP_matrix = construct_distance_angle_matrix(GP_location, DP_location)
    GP_LP_matrix = construct_distance_angle_matrix(GP_location, LP_location)
    DP_LP_matrix = construct_distance_angle_matrix(DP_location, LP_location)

    # Add second stage constraints
    for s in range(S):
        q = demand_list[s]
        EQ_location = EQ_location_list[s]
        mag = magnitude_list[s]
        depth = depth_list[s]

        EQ_LP_matrix = construct_distance_angle_matrix(LP_location, EQ_location)
        EQ_DP_matrix = construct_distance_angle_matrix(DP_location, EQ_location)

        DP_affected_percentage_array = np.zeros((num_candidates_depots,1))
        for d in range(num_candidates_depots):
            distance , angle = EQ_DP_matrix[d][0]
            DP_affected_percentage_array[d] = affected_percentage(mag, depth , distance)
        
        LP_affected_percentage_array = np.zeros((num_candidates_launch_points,1))
        for l in range(num_candidates_launch_points):
            distance , angle = EQ_LP_matrix[l][0]
            LP_affected_percentage_array[l] = affected_percentage(mag, depth , distance)
        
        DP_LP_matrix_new = np.zeros((num_candidates_depots,num_candidates_launch_points))
        for d in range(num_candidates_depots):
            for l in range(num_candidates_launch_points):
                DP_LP_matrix_new[d,l] = DP_LP_matrix[d,l][0] * ( 1 + DP_affected_percentage_array[d] + LP_affected_percentage_array[l] )

        wind_array = weather_list[s][0]
        wind_speed = wind_array[0]
        wind_angle = wind_array[1]

        for l in range(num_candidates_launch_points):
            m.addConstr(gp.quicksum( q[i][0] * x[s,i,l,k] for i in range(num_gathering_points) for k in range(K_SD)) <= Q_T * v[l] )
        
        for k in range(K_SD):
            m.addConstr(gp.quicksum(x[s,i,l,k] for i in range(num_gathering_points) for l in range(num_candidates_launch_points + num_candidates_depots)) <= g_SD[s,k])
            if k < K_SD - 1:
                m.addConstr(g_SD[s,k] >= g_SD[s,k+1]) # Remove symmetry
            for l in range(num_candidates_launch_points):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_LP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        m.addConstr(x[s,i,l,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                m.addConstr( gp.quicksum( w[d,l] * DP_LP_matrix_new[d,l]/v_T for d in range(num_candidates_depots)) + total_airtime <= T )

            for l in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        m.addConstr(x[s,i,l+num_candidates_launch_points,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l+num_candidates_launch_points,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                m.addConstr( total_airtime <= T )

        for k in range(K_LD):
            m.addConstr(gp.quicksum(y[s,i,d,k] for i in range(num_gathering_points) for d in range(num_candidates_depots)) <= g_LD[s,k])
            if k < K_LD - 1:
                m.addConstr(g_LD[s,k] >= g_LD[s,k+1]) # Remove symmetry
            for d in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,d] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_LD = airspeed_calculation( d = distance , m_pack= Q_LD, drone=largedrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_LD <= wind_speed:
                        m.addConstr(y[s,i,d,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += y[s,i,d,k] * calculate_time_to_deliver( air_speed_LD , angle , wind_speed, wind_angle , distance )
                m.addConstr( total_airtime <= T )

        for i in range(num_gathering_points):
            m.addConstr( q[i][0] 
                        -  Q_SD * gp.quicksum( x[s,i,l,k] for l in range(num_candidates_depots+num_candidates_launch_points) for k in range(K_SD)) 
                        -  Q_LD * gp.quicksum( y[s,i,d,k] for d in range(num_candidates_depots) for k in range(K_LD))
                        <= o[s,i])
            m.addConstr( gp.quicksum( x[s,i,l,k] for l in range(num_candidates_depots+num_candidates_launch_points) for k in range(K_SD) ) 
                        + gp.quicksum( y[s,i,d,k] for d in range(num_candidates_depots) for k in range(K_LD) ) >= 1)
            for d in range(num_candidates_depots):
                for k in range(K_LD):
                    m.addConstr(y[s,i,d,k] <= u[d])
                for k in range( K_SD ):
                    m.addConstr(x[s,i,(num_candidates_launch_points+d),k] <= u[d])
            for l in range(num_candidates_launch_points):
                for k in range(K_SD):
                    m.addConstr(x[s,i,l,k] <= v[l])

    # Optimize the model
    m.optimize()
    # Retrieve and store optimal variable values
    if m.status == gp.GRB.OPTIMAL:
        u_values = [u[d].X for d in range(num_candidates_depots)]
        v_values = [v[l].X for l in range(num_candidates_launch_points)]
        w_values = np.zeros((num_candidates_depots,num_candidates_launch_points))
        g_SD_values = np.zeros((S,K_SD))
        unmet_demands = np.zeros((S,num_gathering_points))
        for s in range(S):
            for i in range(num_gathering_points):
                unmet_demands[s,i] = o[s,i].X
            for k in range(K_SD):
                g_SD_values[s,k] = g_SD[s,k].X
        g_LD_values = np.zeros((S,K_LD))
        for s in range(S):
            for k in range(K_LD):
                g_LD_values[s,k] = g_LD[s,k].X
        for d in range(num_candidates_depots):
            for l in range(num_candidates_launch_points):
                w_values[d,l] = w[d,l].X

    if printFlag:
        print("SAA Algorithm has finished\n")
    final_time = time.time()
    if printFlag:
        print(f"Objective Value: {m.ObjVal:.4f}")
        print(f"Time used: {final_time - current_time:.2f}\n")

    return m.ObjVal , final_time-current_time ,(u_values,v_values,w_values) , (g_SD_values , g_LD_values) , unmet_demands

def SAA_equity_obj( set_NO , params , earthquake_info = None):

    """
    This function is designed to solve the sample average approximation algorithm for the UAV routing problem.
    The input parameters contain following:
    - num_scenarios: number of scenarios
    - n_scenarios: number of scenarios
    - pi: Penalty for unmet demand
    - T: Time horizon hour
    - e: maximal number of open depots
    - p: maximal number of open launch points
    - Q_T: capacity of the vehicle
    - c_SD: cost for using a small drone
    - c_LD: cost for using a large drone
    - v_T: speed of the vehicle: 45 km/hr
    - tau: setup time for small drone to deliver: 2 miniutes
    - K_SD: maximal number of small drones
    - K_LD: maximal number of large drones
    - num_candidates_depots: number of candidate depots
    - num_candidates_launch_points: number of candidate launch points
    - num_gathering_points: number of gathering points
    - printlevel: 1 for printing the results
    """

    S = params.n_scenarios
    pi = params.pi
    T = params.T
    e = params.e
    p = params.p
    Q_T = params.Q_T
    c_SD = params.c_SD
    c_LD = params.c_LD
    v_T = params.v_T
    tau = params.tau
    K_SD = params.K_SD
    K_LD = params.K_LD
    num_candidates_depots = params.num_candidates_depots
    num_candidates_launch_points = params.num_candidates_launch_points
    num_gathering_points = params.num_gathering_points
    printlevel = params.printlevel

    if printlevel > 0:
        printFlag = True
    else:
        printFlag = False

    magnitude_list = earthquake_info[0]
    depth_list = earthquake_info[1]

    gp_demand_and_location, D_location, LP_location , earthquake_location = generate_all_locations( S=S , num_GP=num_gathering_points , cand_D=num_candidates_depots , cand_LP=num_candidates_launch_points)
    gp_demand = gp_demand_and_location[0]
    gp_location = gp_demand_and_location[1]

    for s in range(S):
        save_to_csv(set_NO ,"GP", s + 1, gp_demand, gp_location)
        save_to_csv(set_NO ,"DP", s + 1, None, D_location)
        save_to_csv(set_NO ,"LP", s + 1, None, LP_location)
        wind_speed , wind_angle = generate_wind_speed_and_angle()
        wind_array = np.array([[wind_speed],[wind_angle]])
        save_to_csv(set_NO ,"Wind", s + 1, None, None, wind_array)
        save_to_csv(set_NO ,"EQ", s + 1, demand_array=None, location_array=earthquake_location[s])

    if printFlag:
        print(f"Data arrays have been saved\n")

    demand_list , GP_location_list = read_csv_files(set_NO,"GP", S)
    DP_location_list = read_csv_files(set_NO,"DP", S)
    LP_location_list = read_csv_files(set_NO,"LP", S)
    weather_list = read_csv_files(set_NO,"Wind", S)
    EQ_location_list = read_csv_files(set_NO,"EQ", S)

    if printFlag:
        print(f"Data arrays have been read\n")

    smalldrones = SmallDrone("SD")
    largedrones = LargeDrone("LD")

    Q_SD = smalldrones.max_payload
    Q_LD = largedrones.max_payload

    if printFlag:
        print("SAA Algorithm is running...\n")
    current_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SAA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    m = gp.Model("SAA")
    m.setParam('OutputFlag', 0)

    # Add first stage variables
    u = m.addVars(num_candidates_depots, vtype=gp.GRB.BINARY, name="depot")
    v = m.addVars(num_candidates_launch_points, vtype=gp.GRB.BINARY, name="launch_point")
    w = m.addVars(num_candidates_depots, num_candidates_launch_points, vtype=gp.GRB.BINARY, name="LP_to_DP")

    # Add second stage variables
    g_SD = m.addVars(S, K_SD, vtype=gp.GRB.BINARY, name="small_drone_usage")
    g_LD = m.addVars(S, K_LD, vtype=gp.GRB.BINARY, name="large_drone_usage")
    x = m.addVars(S, num_gathering_points, num_candidates_launch_points + num_candidates_depots, K_SD, vtype=gp.GRB.BINARY, name="small_drone_assignment")
    y = m.addVars(S, num_gathering_points, num_candidates_depots, K_LD, vtype=gp.GRB.BINARY, name="large_drone_assignment")
    o = m.addVars(S, num_gathering_points, lb=0, vtype=gp.GRB.CONTINUOUS, name="unmet_demand")
    o_max = m.addVars(S, lb=0, vtype=gp.GRB.CONTINUOUS, name="max_unmet_demand")
    o_min = m.addVars(S, lb=0, vtype=gp.GRB.CONTINUOUS, name="min_unmet_demand") 

    # Add objective function
    m.setObjective( (gp.quicksum(pi * o[s,i]  for s in range(S) for i in range(num_gathering_points)) +
                    gp.quicksum(c_SD * g_SD[s,k] for s in range(S) for k in range(K_SD)) +
                    gp.quicksum(c_LD * g_LD[s,k] for s in range(S) for k in range(K_LD))
                    + 50 * gp.quicksum( (o_max[s] - o_min[s])**2 for s in range(S) ))/S,
                    gp.GRB.MINIMIZE)
    
    # Add first stage constraints
    m.addConstr(gp.quicksum(u[d] for d in range(num_candidates_depots)) <= e)
    m.addConstr(gp.quicksum(v[l] for l in range(num_candidates_launch_points)) <= p)
    for l in range(num_candidates_launch_points):
        m.addConstr(gp.quicksum(w[d,l] for d in range(num_candidates_depots)) == v[l])

    GP_location = GP_location_list[0]
    DP_location = DP_location_list[0]
    LP_location = LP_location_list[0]

    GP_DP_matrix = construct_distance_angle_matrix(GP_location, DP_location)
    GP_LP_matrix = construct_distance_angle_matrix(GP_location, LP_location)
    DP_LP_matrix = construct_distance_angle_matrix(DP_location, LP_location)

    # Add second stage constraints
    for s in range(S):
        m.addGenConstrMax(o_max[s], [o[s,i] for i in range(num_gathering_points)] , constant=0)
        m.addGenConstrMin(o_min[s], [o[s,i] for i in range(num_gathering_points)] , constant=0)
        q = demand_list[s]
        EQ_location = EQ_location_list[s]
        mag = magnitude_list[s]
        depth = depth_list[s]

        EQ_LP_matrix = construct_distance_angle_matrix(LP_location, EQ_location)
        EQ_DP_matrix = construct_distance_angle_matrix(DP_location, EQ_location)

        DP_affected_percentage_array = np.zeros((num_candidates_depots,1))
        for d in range(num_candidates_depots):
            distance , angle = EQ_DP_matrix[d][0]
            DP_affected_percentage_array[d] = affected_percentage(mag, depth , distance)
        
        LP_affected_percentage_array = np.zeros((num_candidates_launch_points,1))
        for l in range(num_candidates_launch_points):
            distance , angle = EQ_LP_matrix[l][0]
            LP_affected_percentage_array[l] = affected_percentage(mag, depth , distance)
        
        DP_LP_matrix_new = np.zeros((num_candidates_depots,num_candidates_launch_points))
        for d in range(num_candidates_depots):
            for l in range(num_candidates_launch_points):
                DP_LP_matrix_new[d,l] = DP_LP_matrix[d,l][0] * ( 1 + DP_affected_percentage_array[d] + LP_affected_percentage_array[l] )

        wind_array = weather_list[s][0]
        wind_speed = wind_array[0]
        wind_angle = wind_array[1]

        for l in range(num_candidates_launch_points):
            m.addConstr(gp.quicksum( q[i][0] * x[s,i,l,k] for i in range(num_gathering_points) for k in range(K_SD)) <= Q_T * v[l] )
        
        for k in range(K_SD):
            m.addConstr(gp.quicksum(x[s,i,l,k] for i in range(num_gathering_points) for l in range(num_candidates_launch_points + num_candidates_depots)) <= g_SD[s,k])
            if k < K_SD - 1:
                m.addConstr(g_SD[s,k] >= g_SD[s,k+1]) # Remove symmetry
            for l in range(num_candidates_launch_points):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_LP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        m.addConstr(x[s,i,l,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                m.addConstr( gp.quicksum( w[d,l] * DP_LP_matrix_new[d,l]/v_T for d in range(num_candidates_depots)) + total_airtime <= T )

            for l in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        m.addConstr(x[s,i,l+num_candidates_launch_points,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l+num_candidates_launch_points,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                m.addConstr( total_airtime <= T )

        for k in range(K_LD):
            m.addConstr(gp.quicksum(y[s,i,d,k] for i in range(num_gathering_points) for d in range(num_candidates_depots)) <= g_LD[s,k])
            if k < K_LD - 1:
                m.addConstr(g_LD[s,k] >= g_LD[s,k+1]) # Remove symmetry
            for d in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,d] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_LD = airspeed_calculation( d = distance , m_pack= Q_LD, drone=largedrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_LD <= wind_speed:
                        m.addConstr(y[s,i,d,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += y[s,i,d,k] * calculate_time_to_deliver( air_speed_LD , angle , wind_speed, wind_angle , distance )
                m.addConstr( total_airtime <= T )

        for i in range(num_gathering_points):
            m.addConstr( q[i][0] 
                        -  Q_SD * gp.quicksum( x[s,i,l,k] for l in range(num_candidates_depots+num_candidates_launch_points) for k in range(K_SD)) 
                        -  Q_LD * gp.quicksum( y[s,i,d,k] for d in range(num_candidates_depots) for k in range(K_LD))
                        <= o[s,i])
            for d in range(num_candidates_depots):
                for k in range(K_LD):
                    m.addConstr(y[s,i,d,k] <= u[d])
                for k in range( K_SD ):
                    m.addConstr(x[s,i,(num_candidates_launch_points+d),k] <= u[d])
            for l in range(num_candidates_launch_points):
                for k in range(K_SD):
                    m.addConstr(x[s,i,l,k] <= v[l])
    # Set the time limit (e.g., 60 seconds)
    m.setParam(gp.GRB.Param.TimeLimit, 3000)
    # Optimize the model
    m.optimize()
    # Retrieve and store optimal variable values

    u_values = [u[d].X for d in range(num_candidates_depots)]
    v_values = [v[l].X for l in range(num_candidates_launch_points)]
    w_values = np.zeros((num_candidates_depots,num_candidates_launch_points))
    unmet_demands = np.zeros((S,num_gathering_points))
    g_SD_values = np.zeros((S,K_SD))
    for s in range(S):
        for i in range(num_gathering_points):
            unmet_demands[s,i] = o[s,i].X
        for k in range(K_SD):
            g_SD_values[s,k] = g_SD[s,k].X
    g_LD_values = np.zeros((S,K_LD))
    for s in range(S):
        for k in range(K_LD):
            g_LD_values[s,k] = g_LD[s,k].X
    for d in range(num_candidates_depots):
        for l in range(num_candidates_launch_points):
            w_values[d,l] = w[d,l].X

    if printFlag:
        print("SAA Algorithm has finished\n")
    final_time = time.time()
    if printFlag:
        print(f"Objective Value: {m.ObjVal:.4f}")
        print(f"Time used: {final_time - current_time:.2f}\n")

    return m.ObjVal , final_time-current_time ,(u_values,v_values,w_values) , (g_SD_values , g_LD_values) , unmet_demands


def SAA_equity_obj_constr( set_NO , params , earthquake_info = None):

    """
    This function is designed to solve the sample average approximation algorithm for the UAV routing problem.
    The input parameters contain following:
    - num_scenarios: number of scenarios
    - n_scenarios: number of scenarios
    - pi: Penalty for unmet demand
    - T: Time horizon hour
    - e: maximal number of open depots
    - p: maximal number of open launch points
    - Q_T: capacity of the vehicle
    - c_SD: cost for using a small drone
    - c_LD: cost for using a large drone
    - v_T: speed of the vehicle: 45 km/hr
    - tau: setup time for small drone to deliver: 2 miniutes
    - K_SD: maximal number of small drones
    - K_LD: maximal number of large drones
    - num_candidates_depots: number of candidate depots
    - num_candidates_launch_points: number of candidate launch points
    - num_gathering_points: number of gathering points
    - printlevel: 1 for printing the results
    """

    S = params.n_scenarios
    pi = params.pi
    T = params.T
    e = params.e
    p = params.p
    Q_T = params.Q_T
    c_SD = params.c_SD
    c_LD = params.c_LD
    v_T = params.v_T
    tau = params.tau
    K_SD = params.K_SD
    K_LD = params.K_LD
    num_candidates_depots = params.num_candidates_depots
    num_candidates_launch_points = params.num_candidates_launch_points
    num_gathering_points = params.num_gathering_points
    printlevel = params.printlevel

    if printlevel > 0:
        printFlag = True
    else:
        printFlag = False

    magnitude_list = earthquake_info[0]
    depth_list = earthquake_info[1]

    gp_demand_and_location, D_location, LP_location , earthquake_location = generate_all_locations( S=S , num_GP=num_gathering_points , cand_D=num_candidates_depots , cand_LP=num_candidates_launch_points)
    gp_demand = gp_demand_and_location[0]
    gp_location = gp_demand_and_location[1]

    for s in range(S):
        save_to_csv(set_NO ,"GP", s + 1, gp_demand, gp_location)
        save_to_csv(set_NO ,"DP", s + 1, None, D_location)
        save_to_csv(set_NO ,"LP", s + 1, None, LP_location)
        wind_speed , wind_angle = generate_wind_speed_and_angle()
        wind_array = np.array([[wind_speed],[wind_angle]])
        save_to_csv(set_NO ,"Wind", s + 1, None, None, wind_array)
        save_to_csv(set_NO ,"EQ", s + 1, demand_array=None, location_array=earthquake_location[s])

    if printFlag:
        print(f"Data arrays have been saved\n")

    demand_list , GP_location_list = read_csv_files(set_NO,"GP", S)
    DP_location_list = read_csv_files(set_NO,"DP", S)
    LP_location_list = read_csv_files(set_NO,"LP", S)
    weather_list = read_csv_files(set_NO,"Wind", S)
    EQ_location_list = read_csv_files(set_NO,"EQ", S)

    if printFlag:
        print(f"Data arrays have been read\n")

    smalldrones = SmallDrone("SD")
    largedrones = LargeDrone("LD")

    Q_SD = smalldrones.max_payload
    Q_LD = largedrones.max_payload

    if printFlag:
        print("SAA Algorithm is running...\n")
    current_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SAA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    m = gp.Model("SAA")
    m.setParam('OutputFlag', 0)

    # Add first stage variables
    u = m.addVars(num_candidates_depots, vtype=gp.GRB.BINARY, name="depot")
    v = m.addVars(num_candidates_launch_points, vtype=gp.GRB.BINARY, name="launch_point")
    w = m.addVars(num_candidates_depots, num_candidates_launch_points, vtype=gp.GRB.BINARY, name="LP_to_DP")

    # Add second stage variables
    g_SD = m.addVars(S, K_SD, vtype=gp.GRB.BINARY, name="small_drone_usage")
    g_LD = m.addVars(S, K_LD, vtype=gp.GRB.BINARY, name="large_drone_usage")
    x = m.addVars(S, num_gathering_points, num_candidates_launch_points + num_candidates_depots, K_SD, vtype=gp.GRB.BINARY, name="small_drone_assignment")
    y = m.addVars(S, num_gathering_points, num_candidates_depots, K_LD, vtype=gp.GRB.BINARY, name="large_drone_assignment")
    o = m.addVars(S, num_gathering_points, lb=0, vtype=gp.GRB.CONTINUOUS, name="unmet_demand")

    o_max = m.addVars(S, lb=0, vtype=gp.GRB.CONTINUOUS, name="max_unmet_demand")
    o_min = m.addVars(S, lb=0, vtype=gp.GRB.CONTINUOUS, name="min_unmet_demand") 

    # Add objective function
    m.setObjective( (gp.quicksum(pi * o[s,i]  for s in range(S) for i in range(num_gathering_points)) +
                    gp.quicksum(c_SD * g_SD[s,k] for s in range(S) for k in range(K_SD)) +
                    gp.quicksum(c_LD * g_LD[s,k] for s in range(S) for k in range(K_LD))
                    + 50 * gp.quicksum( o_max[s] - o_min[s] for s in range(S) ))/S,
                    gp.GRB.MINIMIZE)
    
    # Add first stage constraints
    m.addConstr(gp.quicksum(u[d] for d in range(num_candidates_depots)) <= e)
    m.addConstr(gp.quicksum(v[l] for l in range(num_candidates_launch_points)) <= p)
    for l in range(num_candidates_launch_points):
        m.addConstr(gp.quicksum(w[d,l] for d in range(num_candidates_depots)) == v[l])

    GP_location = GP_location_list[0]
    DP_location = DP_location_list[0]
    LP_location = LP_location_list[0]

    GP_DP_matrix = construct_distance_angle_matrix(GP_location, DP_location)
    GP_LP_matrix = construct_distance_angle_matrix(GP_location, LP_location)
    DP_LP_matrix = construct_distance_angle_matrix(DP_location, LP_location)

    # Add second stage constraints
    for s in range(S):
        m.addGenConstrMax(o_max[s], [o[s,i] for i in range(num_gathering_points)] , constant=0)
        m.addGenConstrMin(o_min[s], [o[s,i] for i in range(num_gathering_points)] , constant=0)
        q = demand_list[s]
        EQ_location = EQ_location_list[s]
        mag = magnitude_list[s]
        depth = depth_list[s]

        EQ_LP_matrix = construct_distance_angle_matrix(LP_location, EQ_location)
        EQ_DP_matrix = construct_distance_angle_matrix(DP_location, EQ_location)

        DP_affected_percentage_array = np.zeros((num_candidates_depots,1))
        for d in range(num_candidates_depots):
            distance , angle = EQ_DP_matrix[d][0]
            DP_affected_percentage_array[d] = affected_percentage(mag, depth , distance)
        
        LP_affected_percentage_array = np.zeros((num_candidates_launch_points,1))
        for l in range(num_candidates_launch_points):
            distance , angle = EQ_LP_matrix[l][0]
            LP_affected_percentage_array[l] = affected_percentage(mag, depth , distance)
        
        DP_LP_matrix_new = np.zeros((num_candidates_depots,num_candidates_launch_points))
        for d in range(num_candidates_depots):
            for l in range(num_candidates_launch_points):
                DP_LP_matrix_new[d,l] = DP_LP_matrix[d,l][0] * ( 1 + DP_affected_percentage_array[d] + LP_affected_percentage_array[l] )

        wind_array = weather_list[s][0]
        wind_speed = wind_array[0]
        wind_angle = wind_array[1]

        for l in range(num_candidates_launch_points):
            m.addConstr(gp.quicksum( q[i][0] * x[s,i,l,k] for i in range(num_gathering_points) for k in range(K_SD)) <= Q_T * v[l] )
        
        for k in range(K_SD):
            m.addConstr(gp.quicksum(x[s,i,l,k] for i in range(num_gathering_points) for l in range(num_candidates_launch_points + num_candidates_depots)) <= g_SD[s,k])
            if k < K_SD - 1:
                m.addConstr(g_SD[s,k] >= g_SD[s,k+1]) # Remove symmetry
            for l in range(num_candidates_launch_points):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_LP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        m.addConstr(x[s,i,l,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                m.addConstr( gp.quicksum( w[d,l] * DP_LP_matrix_new[d,l]/v_T for d in range(num_candidates_depots)) + total_airtime <= T )

            for l in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        m.addConstr(x[s,i,l+num_candidates_launch_points,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l+num_candidates_launch_points,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                m.addConstr( total_airtime <= T )

        for k in range(K_LD):
            m.addConstr(gp.quicksum(y[s,i,d,k] for i in range(num_gathering_points) for d in range(num_candidates_depots)) <= g_LD[s,k])
            if k < K_LD - 1:
                m.addConstr(g_LD[s,k] >= g_LD[s,k+1]) # Remove symmetry
            for d in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,d] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_LD = airspeed_calculation( d = distance , m_pack= Q_LD, drone=largedrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_LD <= wind_speed:
                        m.addConstr(y[s,i,d,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += y[s,i,d,k] * calculate_time_to_deliver( air_speed_LD , angle , wind_speed, wind_angle , distance )
                m.addConstr( total_airtime <= T )

        for i in range(num_gathering_points):
            m.addConstr( q[i][0] 
                        -  Q_SD * gp.quicksum( x[s,i,l,k] for l in range(num_candidates_depots+num_candidates_launch_points) for k in range(K_SD)) 
                        -  Q_LD * gp.quicksum( y[s,i,d,k] for d in range(num_candidates_depots) for k in range(K_LD))
                        <= o[s,i])
            m.addConstr( gp.quicksum( x[s,i,l,k] for l in range(num_candidates_depots+num_candidates_launch_points) for k in range(K_SD) ) 
                        + gp.quicksum( y[s,i,d,k] for d in range(num_candidates_depots) for k in range(K_LD) ) >= 1)
            for d in range(num_candidates_depots):
                for k in range(K_LD):
                    m.addConstr(y[s,i,d,k] <= u[d])
                for k in range( K_SD ):
                    m.addConstr(x[s,i,(num_candidates_launch_points+d),k] <= u[d])
            for l in range(num_candidates_launch_points):
                for k in range(K_SD):
                    m.addConstr(x[s,i,l,k] <= v[l])
    m.setParam(gp.GRB.Param.TimeLimit, 3000)
    # Optimize the model
    m.optimize()

    # Retrieve and store optimal variable values
    u_values = [u[d].X for d in range(num_candidates_depots)]
    v_values = [v[l].X for l in range(num_candidates_launch_points)]
    w_values = np.zeros((num_candidates_depots,num_candidates_launch_points))
    g_SD_values = np.zeros((S,K_SD))
    unmet_demands = np.zeros((S,num_gathering_points))
    for s in range(S):
        for i in range(num_gathering_points):
            unmet_demands[s,i] = o[s,i].X
        for k in range(K_SD):
            g_SD_values[s,k] = g_SD[s,k].X
    g_LD_values = np.zeros((S,K_LD))
    for s in range(S):
        for k in range(K_LD):
            g_LD_values[s,k] = g_LD[s,k].X
    for d in range(num_candidates_depots):
        for l in range(num_candidates_launch_points):
            w_values[d,l] = w[d,l].X

    if printFlag:
        print("SAA Algorithm has finished\n")
    final_time = time.time()
    if printFlag:
        print(f"Objective Value: {m.ObjVal:.4f}")
        print(f"Time used: {final_time - current_time:.2f}\n")

    return m.ObjVal , final_time-current_time ,(u_values,v_values,w_values) , (g_SD_values , g_LD_values) , unmet_demands