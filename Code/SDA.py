import numpy as np
import gurobipy as gp
from functions import *
from data_store import *
from Class_UAV import *
import time
from SDA_evaluations import *

def scenario_decomposition_algorithm( set_NO , params, earthquake_info = None,  max_iter=15 , tolerance=0.05):

    """
    This function is designed to solve the scenario decomposition algorithm for the UAV routing problem.
    The input parameters are the following:
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

    magnitude_list = earthquake_info[0]
    depth_list = earthquake_info[1]

    if printlevel > 0:
        printFlag = True
    else:
        printFlag = False

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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    UB = 1e6
    LB = -1e6

    R = []
    solutions = []
    iter = 0
    last_time = time.time()
    initial_time = last_time
    
    if printFlag:
        print("SAA Algorithm is running...\n")

    if printFlag:
        print(f"{'Iteration':>10} | {'Upper Bound':>15} | {'UB Update':>10} | {'Lower Bound':>15} | {'LB Update':>10} | {'Time (s)':>10}")
        print('-' * 90)

    GP_location = GP_location_list[0]
    DP_location = DP_location_list[0]
    LP_location = LP_location_list[0]
    GP_DP_matrix = construct_distance_angle_matrix(GP_location, DP_location)
    GP_LP_matrix = construct_distance_angle_matrix(GP_location, LP_location)
    DP_LP_matrix = construct_distance_angle_matrix(DP_location, LP_location)

    while (iter < max_iter) and ((UB-LB)/UB > tolerance):
        UB_flag = 0
        LB_flag = 0
        # Solve the optimization problem for each scenario
        lbs = []
        R_hat = []
        if printFlag:
            print("LB Evaluation Started...")
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

            obj , val_tuple = lb_evaluation( GP_DP_matrix , GP_LP_matrix , DP_LP_matrix_new , wind_speed , wind_angle , K_SD , K_LD , num_candidates_depots , num_candidates_launch_points , num_gathering_points , pi , T , e , p , Q_T , c_SD , c_LD , v_T , tau , R , smalldrones , largedrones ,Q_SD , Q_LD , q)

            lbs.append(obj)
            R_hat.append(val_tuple)
        
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Line 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if LB != np.mean(lbs):
            LB_flag = 1
        LB = np.mean(lbs)
        if printFlag:
            print("LB Evaluation Ended...")
        R.extend(R_hat) 
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Line 7 ~ 21 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if printFlag:
            print("UB Evaluation Started...")
        for solutions_set in R_hat:
            ub = 0
            u_values = solutions_set[0]
            v_values = solutions_set[1]
            w_values = solutions_set[2]
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

                ub = ub + ub_evaluation( u_values , v_values , w_values , GP_DP_matrix , GP_LP_matrix , DP_LP_matrix_new , wind_speed , wind_angle , K_SD , K_LD , num_candidates_depots , num_candidates_launch_points , num_gathering_points , pi , T , e , p , Q_T , c_SD , c_LD , v_T , tau , smalldrones , largedrones ,Q_SD , Q_LD , q , S)

            if UB > ub:
                UB_flag = 1
                UB = ub
                u_opt = u_values
                v_opt = v_values
                w_opt = w_values

                solutions.append((u_opt, v_opt, w_opt, UB))
        if printFlag:
            print("UB Evaluation Ended...")
        iter += 1
        current_time = time.time()
        iteration_time = current_time - last_time
        last_time = current_time

        if printFlag:
            print(f"{iter:>10} | {UB:>15.4f} | {UB_flag:>10} | {LB:>15.4f} | {LB_flag:>10} | {iteration_time:>10.2f}")
        # print(f"Iteration number: {iter}")
        # print(f"Upper Bound: {UB:.4f}; UB flag: {UB_flag}")
        # print(f"Lower Bound: {LB:.4f}; LB flag: {LB_flag}")
        # print(f"Time used: {iteration_time:.2f}\n")
        # print('-' * 90)
    if printFlag:
        print(f"{'Total Time':>10} | {current_time - initial_time:>15.2f}")
    return UB , current_time - initial_time , (u_opt, v_opt, w_opt) 
        