import numpy as np
import gurobipy as gp
from functions import *
from data_store import *
from Class_UAV import *
import time
       
def ub_evaluation( u , v , w , GP_DP_matrix , GP_LP_matrix , DP_LP_matrix , wind_speed , wind_angle , K_SD , K_LD , num_candidates_depots , num_candidates_launch_points , num_gathering_points , pi , T , e , p , Q_T , c_SD , c_LD , v_T , tau , smalldrones , largedrones ,Q_SD , Q_LD , q , S):

    # Create a new model for UB evaluation
    m2 = gp.Model("UB_evaluation")
    m2.setParam('OutputFlag', 0)

    # Add second stage variables
    g_SD = m2.addVars(K_SD, vtype=gp.GRB.BINARY, name="small_drone_usage")
    g_LD = m2.addVars(K_LD, vtype=gp.GRB.BINARY, name="large_drone_usage")
    x = m2.addVars(num_gathering_points, num_candidates_launch_points + num_candidates_depots, K_SD, vtype=gp.GRB.BINARY, name="small_drone_assignment")
    y = m2.addVars(num_gathering_points, num_candidates_depots, K_LD, vtype=gp.GRB.BINARY, name="large_drone_assignment")
    o = m2.addVars(num_gathering_points, lb=0, vtype=gp.GRB.CONTINUOUS, name="unmet_demand")

    # Add objective function
    m2.setObjective(gp.quicksum(pi * o[i] for i in range(num_gathering_points)) +
                    gp.quicksum(c_SD * g_SD[k] for k in range(K_SD)) +
                    gp.quicksum(c_LD * g_LD[k] for k in range(K_LD)),
                    gp.GRB.MINIMIZE)

    # Add constraints
    for l in range(num_candidates_launch_points):
        m2.addConstr(gp.quicksum( q[i][0] * x[i,l,k] for i in range(num_gathering_points) for k in range(K_SD)) <= Q_T * v[l] )
        if v[l] == 0:
            for k in range(K_SD):
                for i in range(num_gathering_points):
                    m2.addConstr(x[i,l,k] == 0)
    for d in range(num_candidates_depots):
        if u[d] == 0:
            for k in range(K_SD):
                for i in range(num_gathering_points):
                    m2.addConstr(x[i,(num_candidates_launch_points+d),k] == 0)
            for k in range(K_LD):
                for i in range(num_gathering_points):
                    m2.addConstr(y[i,d,k] == 0)


    for k in range(K_SD):
        m2.addConstr(gp.quicksum(x[i,l,k] for i in range(num_gathering_points) for l in range(num_candidates_launch_points + num_candidates_depots)) <= g_SD[k])
        if k < K_SD - 1:
            m2.addConstr(g_SD[k] >= g_SD[k+1]) # Remove symmetry
        for l in range(num_candidates_launch_points):
            total_airtime = 0
            for i in range(num_gathering_points):
                distance , angle = GP_LP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                if air_speed_SD <= wind_speed:
                    m2.addConstr(x[i,l,k] == 0)
                    total_airtime += 0
                else:
                    total_airtime += x[i,l,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
            m2.addConstr( gp.quicksum( w[d,l] * DP_LP_matrix[d,l]/v_T for d in range(num_candidates_depots)) + total_airtime <= T )
        for l in range(num_candidates_depots):
            total_airtime = 0
            for i in range(num_gathering_points):
                distance , angle = GP_DP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                if air_speed_SD <= wind_speed:
                    m2.addConstr(x[i,l+num_candidates_launch_points,k] == 0)
                    total_airtime += 0
                else:
                    total_airtime += x[i,l+num_candidates_launch_points,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
            m2.addConstr( total_airtime <= T )

    for k in range(K_LD):
        m2.addConstr(gp.quicksum(y[i,d,k] for i in range(num_gathering_points) for d in range(num_candidates_depots)) <= g_LD[k])
        if k < K_LD - 1:
            m2.addConstr(g_LD[k] >= g_LD[k+1]) # Remove symmetry
        for d in range(num_candidates_depots):
            total_airtime = 0
            for i in range(num_gathering_points):
                distance , angle = GP_DP_matrix[i,d] # retrieve the distance and angle between the current launch point and the gathering point
                air_speed_LD = airspeed_calculation( d = distance , m_pack= Q_LD, drone=largedrones) # calculates the maximum airspeed (km/hr) of the small drone
                if air_speed_LD <= wind_speed:
                    m2.addConstr(y[i,d,k] == 0)
                    total_airtime += 0
                else:
                    total_airtime += y[i,d,k] * calculate_time_to_deliver( air_speed_LD , angle , wind_speed, wind_angle , distance )
            m2.addConstr( total_airtime <= T )

    for i in range(num_gathering_points):
        m2.addConstr( q[i][0] 
                    -  Q_SD * gp.quicksum( x[i,l,k] for l in range(num_candidates_depots+num_candidates_launch_points) for k in range(K_SD)) 
                    -  Q_LD * gp.quicksum( y[i,d,k] for d in range(num_candidates_depots) for k in range(K_LD))
                    <= o[i])
        for d in range(num_candidates_depots):
            for k in range(K_LD):
                m2.addConstr(y[i,d,k] <= u[d])
            for k in range( K_SD ):
                m2.addConstr(x[i,(num_candidates_launch_points+d),k] <= u[d])
        for l in range(num_candidates_launch_points):
            for k in range(K_SD):
                m2.addConstr(x[i,l,k] <= v[l])

    # Optimize the model
    m2.optimize()

    if m2.status == gp.GRB.Status.INFEASIBLE:
        value = float('inf')
    else:
        value = 1/S * m2.ObjVal

    return value


def lb_evaluation( GP_DP_matrix , GP_LP_matrix , DP_LP_matrix , wind_speed , wind_angle , K_SD , K_LD , num_candidates_depots , num_candidates_launch_points , num_gathering_points , pi , T , e , p , Q_T , c_SD , c_LD , v_T , tau , R , smalldrones , largedrones ,Q_SD , Q_LD , q):

    """
    This function is designed to solve the scenario decomposition algorithm for the UAV routing problem.
    The input parameters are the following:
    - GP_DP_matrix: distance and angle matrix between gathering points and depots
    - GP_LP_matrix: distance and angle matrix between gathering points and launch points
    - DP_LP_matrix: distance and angle matrix between depots and launch points
    - wind_speed: wind speed
    - wind_angle: wind angle
    - K_SD: maximal number of small drones
    - K_LD: maximal number of large drones
    - num_candidates_depots: number of candidate depots
    - num_candidates_launch_points: number of candidate launch points
    - num_gathering_points: number of gathering points
    - pi: Penalty for unmet demand
    - T: Time horizon in hour
    - e: maximal number of open depots
    - p: maximal number of open launch points
    - Q_T: capacity of the vehicle
    - c_SD: cost for using a small drone
    - c_LD: cost for using a large drone
    - v_T: speed of the vehicle: 45 km/hr
    - tau: setup time for small drone to deliver: 2 miniutes
    - R: list of optimal solutions
    - R_hat: list of optimal solutions
    - lbs: list of optimal solutions
    - smalldrones: small drone object
    - largedrones: large drone object
    - Q_SD: maximum payload of the small drone
    - Q_LD: maximum payload of the large drone
    - q: demand list
    """

   # Create a new model for scenario solving
    m1 = gp.Model("scenario_solving")
    m1.setParam('OutputFlag', 0)

    # Add first stage variables
    u = m1.addVars(num_candidates_depots, vtype=gp.GRB.BINARY, name="depot")
    v = m1.addVars(num_candidates_launch_points, vtype=gp.GRB.BINARY, name="launch_point")
    w = m1.addVars(num_candidates_depots, num_candidates_launch_points, vtype=gp.GRB.BINARY, name="LP_to_DP")

    # v.BranchPriority = 1
    # u.BranchPriority = 3
    # w.BranchPriority = 1

    # Add second stage variables
    g_SD = m1.addVars(K_SD, vtype=gp.GRB.BINARY, name="small_drone_usage")
    g_LD = m1.addVars(K_LD, vtype=gp.GRB.BINARY, name="large_drone_usage")
    x = m1.addVars(num_gathering_points, num_candidates_launch_points + num_candidates_depots, K_SD, vtype=gp.GRB.BINARY, name="small_drone_assignment")
    y = m1.addVars(num_gathering_points, num_candidates_depots, K_LD, vtype=gp.GRB.BINARY, name="large_drone_assignment")
    o = m1.addVars(num_gathering_points, lb=0, vtype=gp.GRB.CONTINUOUS, name="unmet_demand")

    # Add objective function
    m1.setObjective(gp.quicksum(pi * o[i] for i in range(num_gathering_points)) +
                    gp.quicksum(c_SD * g_SD[k] for k in range(K_SD)) +
                    gp.quicksum(c_LD * g_LD[k] for k in range(K_LD)),
                    gp.GRB.MINIMIZE)
    # Add constraints
    m1.addConstr(gp.quicksum(u[d] for d in range(num_candidates_depots)) <= e)
    m1.addConstr(gp.quicksum(v[l] for l in range(num_candidates_launch_points)) <= p)
    for l in range(num_candidates_launch_points):
        m1.addConstr(gp.quicksum(w[d,l] for d in range(num_candidates_depots)) == v[l])
        m1.addConstr(gp.quicksum( q[i][0] * x[i,l,k] for i in range(num_gathering_points) for k in range(K_SD)) <= Q_T * v[l] )

    for k in range(K_SD):
        m1.addConstr(gp.quicksum(x[i,l,k] for i in range(num_gathering_points) for l in range(num_candidates_launch_points + num_candidates_depots)) <= g_SD[k])
        if k < K_SD - 1:
            m1.addConstr(g_SD[k] >= g_SD[k+1]) # Remove symmetry
        for l in range(num_candidates_launch_points):
            total_airtime = 0
            for i in range(num_gathering_points):
                distance , angle = GP_LP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                if air_speed_SD <= wind_speed:
                    m1.addConstr(x[i,l,k] == 0)
                    total_airtime += 0
                else:
                    total_airtime += x[i,l,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
            m1.addConstr( gp.quicksum( w[d,l] * DP_LP_matrix[d,l]/v_T for d in range(num_candidates_depots)) + total_airtime <= T )
        for l in range(num_candidates_depots):
            total_airtime = 0
            for i in range(num_gathering_points):
                distance , angle = GP_DP_matrix[i,l] # retrieve the distance and angle between the current launch point and the gathering point
                air_speed_SD = airspeed_calculation( d = distance , m_pack= Q_SD, drone=smalldrones) # calculates the maximum airspeed (km/hr) of the small drone
                if air_speed_SD <= wind_speed:
                    m1.addConstr(x[i,l+num_candidates_launch_points,k] == 0)
                    total_airtime += 0
                else:
                    total_airtime += x[i,l+num_candidates_launch_points,k] * (calculate_time_to_deliver( air_speed_SD , angle , wind_speed, wind_angle , distance ) + tau)
            m1.addConstr( total_airtime <= T )

    for k in range(K_LD):
        m1.addConstr(gp.quicksum(y[i,d,k] for i in range(num_gathering_points) for d in range(num_candidates_depots)) <= g_LD[k])
        if k < K_LD - 1:
            m1.addConstr(g_LD[k] >= g_LD[k+1]) # Remove symmetry
        for d in range(num_candidates_depots):
            total_airtime = 0
            for i in range(num_gathering_points):
                distance , angle = GP_DP_matrix[i,d] # retrieve the distance and angle between the current launch point and the gathering point
                air_speed_LD = airspeed_calculation( d = distance , m_pack= Q_LD, drone=largedrones) # calculates the maximum airspeed (km/hr) of the small drone
                if air_speed_LD <= wind_speed:
                    m1.addConstr(y[i,d,k] == 0)
                    total_airtime += 0
                else:
                    total_airtime += y[i,d,k] * calculate_time_to_deliver( air_speed_LD , angle , wind_speed, wind_angle , distance )
            m1.addConstr( total_airtime <= T )

    for i in range(num_gathering_points):
        m1.addConstr( q[i][0] 
                    -  Q_SD * gp.quicksum( x[i,l,k] for l in range(num_candidates_depots+num_candidates_launch_points) for k in range(K_SD)) 
                    -  Q_LD * gp.quicksum( y[i,d,k] for d in range(num_candidates_depots) for k in range(K_LD))
                    <= o[i])
        for d in range(num_candidates_depots):
            for k in range(K_LD):
                m1.addConstr(y[i,d,k] <= u[d])
            for k in range( K_SD ):
                m1.addConstr(x[i,(num_candidates_launch_points+d),k] <= u[d])
        for l in range(num_candidates_launch_points):
            for k in range(K_SD):
                m1.addConstr(x[i,l,k] <= v[l])

    if len(R) > 0:
        for solution_set in R:
            # optimality constraint implementation
            u_bar = solution_set[0]
            v_bar = solution_set[1]
            w_bar = solution_set[2]
            m1.addConstr(
                gp.quicksum(u[d] for d in range(num_candidates_depots) if u_bar[d] == 0) +
                gp.quicksum((1 - u[d]) for d in range(num_candidates_depots) if u_bar[d] == 1) +
                gp.quicksum(v[l] for l in range(num_candidates_launch_points) if v_bar[l] == 0) +
                gp.quicksum((1 - v[l]) for l in range(num_candidates_launch_points) if v_bar[l] == 1) +
                gp.quicksum(w[d, l] for d in range(num_candidates_depots) for l in range(num_candidates_launch_points) if w_bar[d, l] == 0) +
                gp.quicksum((1 - w[d, l]) for d in range(num_candidates_depots) for l in range(num_candidates_launch_points) if w_bar[d, l] == 1)
                >= 1
            )

    # Optimize the model
    m1.optimize()
    # Retrieve and store optimal variable values
    if m1.status == gp.GRB.OPTIMAL:
        u_values = [u[d].X for d in range(num_candidates_depots)]
        v_values = [v[l].X for l in range(num_candidates_launch_points)]
        w_values = np.zeros((num_candidates_depots,num_candidates_launch_points))
        o_values = np.zeros(num_gathering_points)
        for d in range(num_candidates_depots):
            for l in range(num_candidates_launch_points):
                w_values[d,l] = w[d,l].X
        for i in range(num_gathering_points):
            o_values[i] = o[i].X
    
    return m1.objVal , (u_values,v_values,w_values,o_values)


#
    

    
