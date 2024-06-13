import numpy as np
import gurobipy as gp
from functions import *
from data_store import *
from Class_UAV import *
import time
from functools import partial

def benders_decomposition( set_NO, params , earthquake_info ):

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

    e = params.e
    p = params.p
    num_candidates_depots = params.num_candidates_depots
    num_candidates_launch_points = params.num_candidates_launch_points
    printlevel = params.printlevel

    current_time = time.time()

    if printlevel > 0:
        printFlag = True
    else:
        printFlag = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Benders Decomposition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define the master problem
    master = gp.Model("master")
    master.Params.OutputFlag = 0
    # Set tolerance parameters
    master.setParam(gp.GRB.Param.MIPGap, 0.05) 

    # Add first stage variables
    u = master.addVars(num_candidates_depots, vtype=gp.GRB.BINARY, name="u")
    v = master.addVars(num_candidates_launch_points, vtype=gp.GRB.BINARY, name="v")
    w = master.addVars(num_candidates_depots, num_candidates_launch_points, vtype=gp.GRB.BINARY, name="w")
    # Define variable for the benders decomposition
    theta = master.addVar(vtype=gp.GRB.CONTINUOUS, name="theta")
    # Enable lazy constraints
    master.setParam(gp.GRB.Param.LazyConstraints, 1)
    # Define objective function for the master problem
    master.setObjective( theta, gp.GRB.MINIMIZE)
    # Add first stage constraints
    master.addConstr(gp.quicksum(u[d] for d in range(num_candidates_depots)) <= e, name="c0")
    master.addConstr(gp.quicksum(v[l] for l in range(num_candidates_launch_points)) <= p, name="c1")
    for l in range(num_candidates_launch_points):
        master.addConstr(gp.quicksum(w[d,l] for d in range(num_candidates_depots)) == v[l], name=f"c2[{l}]")

    # Create the partial function
    callback_with_args = partial(benders_callback, theta = theta,  params=params, earthquake_info=earthquake_info, set_NO=set_NO)
    
    master.optimize(callback_with_args)

    # Retrieve and store optimal variable values
    if master.status == gp.GRB.OPTIMAL:
        u_values = [u[d].X for d in range(num_candidates_depots)]
        v_values = [v[l].X for l in range(num_candidates_launch_points)]
        w_values = np.zeros((num_candidates_depots,num_candidates_launch_points))

    final_time = time.time()
    if printFlag:
        print(f"Objective Value: {master.ObjVal:.4f}")
        print(f"Time used: {final_time - current_time:.2f}\n")

    return master.ObjVal , final_time-current_time ,(u_values,v_values,w_values)


def solve_subproblem(params , fixed_u, fixed_v, fixed_w , demand_list , GP_location_list , DP_location_list , LP_location_list , weather_list , EQ_location_list , drones , earthquake_info):

    S = params.n_scenarios
    pi = params.pi
    T = params.T
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

    smalldrones = drones[0]
    largedrones = drones[1]

    Q_SD = smalldrones.max_payload
    Q_LD = largedrones.max_payload

    magnitude_list = earthquake_info[0]
    depth_list = earthquake_info[1]

    fixed_w = np.asarray(fixed_w).reshape(num_candidates_depots,num_candidates_launch_points)

    # Define the subproblem
    sub = gp.Model("subproblem")

    # Add second stage variables
    g_SD = sub.addVars(S, K_SD, vtype=gp.GRB.BINARY, name="small_drone_usage")
    g_LD = sub.addVars(S, K_LD, vtype=gp.GRB.BINARY, name="large_drone_usage")
    x = sub.addVars(S, num_gathering_points, num_candidates_launch_points + num_candidates_depots, K_SD, vtype=gp.GRB.BINARY, name="small_drone_assignment")
    y = sub.addVars(S, num_gathering_points, num_candidates_depots, K_LD, vtype=gp.GRB.BINARY, name="large_drone_assignment")
    o = sub.addVars(S, num_gathering_points, lb=0, vtype=gp.GRB.CONTINUOUS, name="unmet_demand")

    # Add objective function
    sub.setObjective( (gp.quicksum(pi * o[s,i]  for s in range(S) for i in range(num_gathering_points)) +
                    gp.quicksum(c_SD * g_SD[s,k] for s in range(S) for k in range(K_SD)) +
                    gp.quicksum(c_LD * g_LD[s,k] for s in range(S) for k in range(K_LD)))/S,
                    gp.GRB.MINIMIZE)
    
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
        LP_affected_percentage_array = np.zeros((num_candidates_launch_points,1))

        DP_LP_matrix_new = np.zeros((num_candidates_depots,num_candidates_launch_points))
        
        for d in range(num_candidates_depots):

            distance , angle = EQ_DP_matrix[d][0]
            DP_affected_percentage_array[d] = affected_percentage(mag, depth , distance)

            for l in range(num_candidates_launch_points):
                DP_LP_matrix_new[d,l] = DP_LP_matrix[d,l][0] * ( 1 + DP_affected_percentage_array[d] + LP_affected_percentage_array[l] )

            if fixed_u[d] == 0:
                for k in range(K_SD):
                    for i in range(num_gathering_points):
                        sub.addConstr(x[s,i,(num_candidates_launch_points+d),k] == 0)
                for k in range(K_LD):
                    for i in range(num_gathering_points):
                        sub.addConstr(y[s,i,d,k] == 0)


        wind_array = weather_list[s][0]
        wind_speed = wind_array[0]
        wind_angle = wind_array[1]

        for l in range(num_candidates_launch_points):

            distance , angle = EQ_LP_matrix[l][0]
            LP_affected_percentage_array[l] = affected_percentage(mag, depth , distance)

            sub.addConstr(gp.quicksum( q[i][0] * x[s,i,l,k] for i in range(num_gathering_points) for k in range(K_SD)) <= Q_T * fixed_v[l] )
            if fixed_v[l] == 0:
                for k in range(K_SD):
                    for i in range(num_gathering_points):
                        sub .addConstr(x[s,i,l,k] == 0)
        
        for k in range(K_SD):
            sub.addConstr(gp.quicksum(x[s,i,l,k] for i in range(num_gathering_points) for l in range(num_candidates_launch_points + num_candidates_depots)) <= g_SD[s,k])
            if k < K_SD - 1:
                sub.addConstr(g_SD[s,k] >= g_SD[s,k+1]) # Remove symmetry
            for l in range(num_candidates_launch_points):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_LP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        sub.addConstr(x[s,i,l,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                sub.addConstr( gp.quicksum( fixed_w[d,l] * DP_LP_matrix_new[d,l]/v_T for d in range(num_candidates_depots)) + total_airtime <= T )

            for l in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        sub.addConstr(x[s,i,l+num_candidates_launch_points,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l+num_candidates_launch_points,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                sub.addConstr( total_airtime <= T )

        for k in range(K_LD):
            sub.addConstr(gp.quicksum(y[s,i,d,k] for i in range(num_gathering_points) for d in range(num_candidates_depots)) <= g_LD[s,k])
            if k < K_LD - 1:
                sub.addConstr(g_LD[s,k] >= g_LD[s,k+1]) # Remove symmetry
            for d in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,d] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_LD = airspeed_calculation( d = distance , m_pack= Q_LD, drone=largedrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_LD <= wind_speed:
                        sub.addConstr(y[s,i,d,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += y[s,i,d,k] * calculate_time_to_deliver( air_speed_LD , angle , wind_speed, wind_angle , distance )
                sub.addConstr( total_airtime <= T )

        for i in range(num_gathering_points):
            sub.addConstr( q[i][0] 
                        -  Q_SD * gp.quicksum( x[s,i,l,k] for l in range(num_candidates_depots+num_candidates_launch_points) for k in range(K_SD)) 
                        -  Q_LD * gp.quicksum( y[s,i,d,k] for d in range(num_candidates_depots) for k in range(K_LD))
                        <= o[s,i] , name=f"o[{s},{i}]")
            for d in range(num_candidates_depots):
                for k in range(K_LD):
                    sub.addConstr(y[s,i,d,k] <= fixed_u[d])
                for k in range( K_SD ):
                    sub.addConstr(x[s,i,(num_candidates_launch_points+d),k] <= fixed_u[d])
            for l in range(num_candidates_launch_points):
                for k in range(K_SD):
                    sub.addConstr(x[s,i,l,k] <= fixed_v[l])
    
    sub.setParam('OutputFlag', 0)  # Turn off solver output
    sub.optimize()
    
    if sub.status == gp.GRB.Status.OPTIMAL:
        return sub
    else:
        return None
    
# Callback function for the Benders decomposition
def benders_callback(model, where, theta , params, earthquake_info, set_NO):

    S = params.n_scenarios
    num_candidates_depots = params.num_candidates_depots
    num_candidates_launch_points = params.num_candidates_launch_points
    num_gathering_points = params.num_gathering_points
    printlevel = params.printlevel

    if printlevel > 0:
        printFlag = True
    else:
        printFlag = False

    for s in range(S):
        gp_demand_and_location, D_location, LP_location , earthquake_location = generate_all_locations(S=S,num_GP=num_gathering_points , cand_D=num_candidates_depots , cand_LP=num_candidates_launch_points)
        gp_demand = gp_demand_and_location[0]
        gp_location = gp_demand_and_location[1]
        wind_speed , wind_angle = generate_wind_speed_and_angle()
        wind_array = np.array([[wind_speed],[wind_angle]])
        save_to_csv(set_NO ,"GP", s + 1, gp_demand, gp_location)
        save_to_csv(set_NO ,"DP", s + 1, None, D_location)
        save_to_csv(set_NO ,"LP", s + 1, None, LP_location)
        save_to_csv(set_NO ,"Wind", s + 1, None, None, wind_array)
        save_to_csv(set_NO ,"EQ", s + 1, demand_array=None, location_array=earthquake_location)

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

    if printFlag:
        print("SAA Algorithm is running...\n")

    if where == gp.GRB.Callback.MIPSOL:
        # Get the values of the binary variables
        fixed_u = model.cbGetSolution([model.getVarByName(f"u[{d}]") for d in range(num_candidates_depots)])
        fixed_v = model.cbGetSolution([model.getVarByName(f"v[{l}]") for l in range(num_candidates_launch_points)])
        fixed_w = model.cbGetSolution([model.getVarByName(f"w[{d},{l}]") for d in range(num_candidates_depots) for l in range(num_candidates_launch_points)])
    
        # Solve the subproblem with fixed y, g, and z
        sub_problem = solve_subproblem( params , fixed_u, fixed_v, fixed_w , demand_list , GP_location_list , DP_location_list , LP_location_list , weather_list , EQ_location_list , [smalldrones, largedrones] , earthquake_info)
        sub_objval = sub_problem.objVal
        # Solve the LP relaxation to find a L
        L = solve_lprelaxation( params , fixed_u, fixed_v, fixed_w , demand_list , GP_location_list , DP_location_list , LP_location_list , weather_list , EQ_location_list , [smalldrones, largedrones] , earthquake_info)

        if sub_objval is not None:
            # Get the value of theta in the current solution
            theta_val = model.cbGetSolution(model.getVarByName("theta"))

            # Add Benders cut if necessary
            if theta_val < sub_objval:

                # Initialize sums
                sum_ones_u = 0
                sum_others_u = 0

                # Iterate through the list and sum up the values accordingly
                for value in fixed_u:
                    if value == 1:
                        sum_ones_u += value
                    else:
                        sum_others_u += value

                # Calculate the final result
                result_u = sum_ones_u - sum_others_u

                # Initialize sums
                sum_ones_v = 0
                sum_others_v = 0

                # Iterate through the list and sum up the values accordingly
                for value in fixed_v:
                    if value == 1:
                        sum_ones_v += value
                    else:
                        sum_others_v += value

                # Calculate the final result
                result_v = sum_ones_v - sum_others_v

                # Initialize sums
                sum_ones_w = 0
                sum_others_w = 0

                # Iterate through the list and sum up the values accordingly
                for value in fixed_w:
                    if value == 1:
                        sum_ones_w += value
                    else:
                        sum_others_w += value

                # Calculate the final result
                result_w = sum_ones_w - sum_others_w
                # Formulate the Benders cut
                expr = ( sub_objval - L ) * (1 + result_v - sum_ones_v + result_u - sum_ones_u + result_w - sum_ones_w) + L

                model.cbLazy(theta >= expr)
        





def solve_lprelaxation(params , fixed_u, fixed_v, fixed_w , demand_list , GP_location_list , DP_location_list , LP_location_list , weather_list , EQ_location_list , drones , earthquake_info):

    S = params.n_scenarios
    pi = params.pi
    T = params.T
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

    smalldrones = drones[0]
    largedrones = drones[1]

    Q_SD = smalldrones.max_payload
    Q_LD = largedrones.max_payload

    magnitude_list = earthquake_info[0]
    depth_list = earthquake_info[1]

    fixed_w = np.asarray(fixed_w).reshape(num_candidates_depots,num_candidates_launch_points)

    # Define the subproblem
    sub = gp.Model("subproblem")

    # Add second stage variables
    g_SD = sub.addVars(S, K_SD, vtype=gp.GRB.CONTINUOUS, name="small_drone_usage")
    g_LD = sub.addVars(S, K_LD, vtype=gp.GRB.CONTINUOUS, name="large_drone_usage")
    x = sub.addVars(S, num_gathering_points, num_candidates_launch_points + num_candidates_depots, K_SD, vtype=gp.GRB.CONTINUOUS, name="small_drone_assignment")
    y = sub.addVars(S, num_gathering_points, num_candidates_depots, K_LD, vtype=gp.GRB.CONTINUOUS, name="large_drone_assignment")
    o = sub.addVars(S, num_gathering_points, lb=0, vtype=gp.GRB.CONTINUOUS, name="unmet_demand")

    # Add objective function
    sub.setObjective( (gp.quicksum(pi * o[s,i]  for s in range(S) for i in range(num_gathering_points)) +
                    gp.quicksum(c_SD * g_SD[s,k] for s in range(S) for k in range(K_SD)) +
                    gp.quicksum(c_LD * g_LD[s,k] for s in range(S) for k in range(K_LD)))/S,
                    gp.GRB.MINIMIZE)
    
    # Add second stage constraints
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
            sub.addConstr(gp.quicksum( q[i][0] * x[s,i,l,k] for i in range(num_gathering_points) for k in range(K_SD)) <= Q_T * fixed_v[l] )
        
        for k in range(K_SD):
            sub.addConstr(gp.quicksum(x[s,i,l,k] for i in range(num_gathering_points) for l in range(num_candidates_launch_points + num_candidates_depots)) <= g_SD[s,k])
            if k < K_SD - 1:
                sub.addConstr(g_SD[s,k] >= g_SD[s,k+1]) # Remove symmetry
            for l in range(num_candidates_launch_points):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_LP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        sub.addConstr(x[s,i,l,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                sub.addConstr( gp.quicksum( fixed_w[d,l] * DP_LP_matrix_new[d,l]/v_T for d in range(num_candidates_depots)) + total_airtime <= T )

            for l in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_SD <= wind_speed:
                        sub.addConstr(x[s,i,l+num_candidates_launch_points,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += x[s,i,l+num_candidates_launch_points,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
                sub.addConstr( total_airtime <= T )

        for k in range(K_LD):
            sub.addConstr(gp.quicksum(y[s,i,d,k] for i in range(num_gathering_points) for d in range(num_candidates_depots)) <= g_LD[s,k])
            if k < K_LD - 1:
                sub.addConstr(g_LD[s,k] >= g_LD[s,k+1]) # Remove symmetry
            for d in range(num_candidates_depots):
                total_airtime = 0
                for i in range(num_gathering_points):
                    distance , angle = GP_DP_matrix[i,d] # retrieve the distance and angle between the current launch point and the gathering point
                    air_speed_LD = airspeed_calculation( d = distance , m_pack= Q_LD, drone=largedrones) # calculates the maximum airspeed (km/hr) of the small drone
                    if air_speed_LD <= wind_speed:
                        sub.addConstr(y[s,i,d,k] == 0)
                        total_airtime += 0
                    else:
                        total_airtime += y[s,i,d,k] * calculate_time_to_deliver( air_speed_LD , angle , wind_speed, wind_angle , distance )
                sub.addConstr( total_airtime <= T )

        for i in range(num_gathering_points):
            sub.addConstr( q[i][0] 
                        -  Q_SD * gp.quicksum( x[s,i,l,k] for l in range(num_candidates_depots+num_candidates_launch_points) for k in range(K_SD)) 
                        -  Q_LD * gp.quicksum( y[s,i,d,k] for d in range(num_candidates_depots) for k in range(K_LD))
                        <= o[s,i])
            for d in range(num_candidates_depots):
                for k in range(K_LD):
                    sub.addConstr(y[s,i,d,k] <= fixed_u[d])
                for k in range( K_SD ):
                    sub.addConstr(x[s,i,(num_candidates_launch_points+d),k] <= fixed_u[d])
            for l in range(num_candidates_launch_points):
                for k in range(K_SD):
                    sub.addConstr(x[s,i,l,k] <= fixed_v[l])
    
    sub.setParam('OutputFlag', 0)  # Turn off solver output
    sub.optimize()
    
    if sub.status == gp.GRB.Status.OPTIMAL:
        return sub.objVal
    else:
        return None