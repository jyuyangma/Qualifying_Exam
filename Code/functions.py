import numpy as np

def airspeed_calculation( d, m_pack, drone, rho = 1.225, f = 1.2):
    '''
    This function is designed to calculate the air speed of a delivery drone, from depot/launch site to the gathering point.
    The calculation is based on the range formula provided in the paper: 
    Dukkanti et al. (2021) "Minimizing energy and cost in range-limited drone deliveries with speed optimization".
    ===============================================================================================
    We have the input parameters as follows:
    Input:
    - d     : the distance between the depot/launch site and the gathering point (in meters)
    - rho   : the air density (in kg/m^3)
    - m_pack: the mass of the payload (in kg)
    - drone : a Python object with the following attributes:
                ~ delta : profile drag coefficient
                ~ m_uav : the mass of the UAV (in kg)
                ~ m_batt: the mass of the battery (in kg)
                ~ U_tip : tip speed of the rotor (in m/s)
                ~ s     : rotor solidity
                ~ A     : rotor disk area (in m^2)
                ~ omega : blade angular velocity (in rad/s) 
                ~ r     : radius of the rotor (in meters)
                ~ k     : incremental correction factor to induced power
                ~ v_0   : mean rotor induced velocity in hover (in m/s)
                ~ d_r   : fuselage drag ratio
                ~ B_mass: energy capacity (density) per mass of the battery (in J/kg)
                ~ theta : depth of discharge (between 0 and 1)
                ~ v_max : maximum speed of the delivery drone (in m/s)
    - f     : safety factor to reserve energy
    Output:
    - ground speed of the delivery drone (in km/hr).
    ===============================================================================================
    '''
    # Extract the attributes of the drone object
    delta  = drone.delta
    m_uav  = drone.m_uav
    m_batt = drone.m_batt
    U_tip  = drone.U_tip
    s      = drone.s
    A      = drone.A
    omega  = drone.omega
    r      = drone.r
    k      = drone.k
    v_0    = drone.v_0
    d_r    = drone.d_r
    B_mass = drone.B_mass
    theta  = drone.theta

    # Construct some intermediate variables
    p_0   = (delta/8) * rho * s * A * omega**3 * r**3
    p_f   = (1 + k) * (m_uav + m_pack + m_batt)**(1.5) / np.sqrt(2 * rho * A)
    p_b   = (1 + k) * (m_uav + m_batt)**(1.5) / np.sqrt(2 * rho * A)
    mu_1  = p_0
    mu_2  = 3*p_0/(U_tip**2)
    mu_3f = p_f * v_0
    mu_3b = p_b * v_0
    mu_4  = 0.5 * d_r * rho * s * A
    THETA = (m_batt * B_mass * theta) / f
    # Calculate the air speed of the delivery drone
    coeff = [2*mu_4*d*1000 , 2*mu_2*d*1000 , -2*THETA , 2*mu_1*d*1000, mu_3f + mu_3b]
    roots = np.roots(coeff)
    # Filter for positive real roots
    v_lists = [root for root in roots if np.isreal(root) and 0 < root.real]
    # Convert the list to a numpy array
    v_lists = np.array(v_lists).real
    
    if len(v_lists) == 0:
        air_speed = 0  # No positive roots found within the maximum speed
    else:
        air_speed = np.min(v_lists)  # Choose the smallest positive root within the maximum spee

    return 3.6 * air_speed

def generate_unique_positions(num_positions, size):
    """
    Generate a set of unique positions on the map.

    Parameters:
    - num_positions (int): Number of positions to generate.
    - size (tuple): The size of the map (width, height).

    Returns:
    - set: A set of unique positions (tuples).
    """
    positions = set()
    while len(positions) < num_positions:
        new_position = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
        positions.add(new_position)
    return positions

def generate_all_locations(S , num_GP, cand_D, cand_LP, size=(300, 300)):
    """
    Parameters:
    - num_GP (int) : Number of gathering points to generate.
    - cand_D (int) : Number of candidates for depots to generate.
    - cand_LP (int): Number of candidates for launch points to generate.
    - size (tuple): The size of the map (width, height).
    - mean_demand (float): Mean of the normal distribution for demand weights.
    - std_demand (float): Standard deviation of the normal distribution for demand weights.

    Returns:
    - demand_amount_and_location (list): A list of arrays, first element is the demand, the second element is the location.
    - depots_location (list): A list of depot locations.
    - launch_points_location (list): A list of launch point locations.
    P.S. Each unit is quantified as 1 kilometers.
    """
    
    # Generate unique positions for gathering points, depots, and launch points
    total_positions = num_GP + cand_D + cand_LP + S
    unique_positions = generate_unique_positions(total_positions, size)
    unique_positions = list(unique_positions)

    # Split positions into gathering points, depots, and launch points
    GP_positions = unique_positions[:num_GP]
    D_positions = unique_positions[num_GP:num_GP+cand_D]
    LP_positions = unique_positions[num_GP+cand_D:num_GP + cand_D + cand_LP]
    earthquake_positions = unique_positions[num_GP + cand_D + cand_LP:]

    demand_list = np.zeros(num_GP)
    # Set customer demands on the map
    for i in range(num_GP):
        # Ensure demand is positive
        demand = np.random.uniform(low=0.02, high=20)
        demand_list[i] = demand

    return [demand_list, GP_positions], D_positions, LP_positions , earthquake_positions

def calculate_distance_and_angle( DP_LP_loc, GP_loc ):
    '''
    This function calculates the distance and angle between two locations.
    Assuming that each unit on the map is equivalent to 5 kilometers.
    '''
    # Calculate distance using numpy.linalg.norm
    distance = np.linalg.norm(DP_LP_loc - GP_loc)
    
    # Calculate angle in radians using numpy.arctan2
    angle_radians = np.arctan2(GP_loc[1] - DP_LP_loc[1], GP_loc[0] - DP_LP_loc[0])
    
    # Convert angle to degrees
    # angle_degrees = np.degrees(angle_radians)
    
    return distance, angle_radians

def generate_wind_speed_and_angle(min_speed=0, max_speed=20):
    # Generate a random wind speed within the specified range
    wind_speed = np.random.uniform(min_speed, max_speed)
    
    # Generate a random angle between 0 and 360 degrees
    wind_angle_degrees = np.random.uniform(0, 360)
    
    # Convert the angle to radians
    wind_angle_radians = np.radians(wind_angle_degrees)
    
    return wind_speed, wind_angle_radians

def calculate_time_to_deliver( air_speed, drone_angle, wind_speed, wind_angle, distance ):
    '''
    This function calculates the time taken to deliver a package from the depot to the gathering point.
    The time is calculated based on the air speed of the drone, the wind speed and angle, and the distance to be covered.
    '''
    # Calculate the forward time of the delivery
    time_forward = distance / (np.sqrt(air_speed**2 - (wind_speed**2 * np.sin(wind_angle - drone_angle)**2)) + wind_speed * np.cos(wind_angle - drone_angle))
    time_backward = distance / (np.sqrt(air_speed**2 - (wind_speed**2 * np.sin(wind_angle - drone_angle)**2)) - wind_speed * np.cos(wind_angle - drone_angle))
    
    return time_forward + time_backward

def construct_distance_angle_matrix(list1, list2):

    n = len(list1)
    m = len(list2)
    
    # Initialize a matrix to store the distance and angle tuples
    matrix = np.empty((n, m), dtype=object)
    
    for i in range(n):
        for j in range(m):
            distance, angle = calculate_distance_and_angle(list1[i], list2[j])
            matrix[i, j] = (distance, angle)
    
    return matrix

def intensity_calculation( magnitude , depth , distance ):

    intensity1 = 7.023 + 0.703 * magnitude - 2.826 * np.log10(distance)
    intensity2 = 5.002 + 0.75 * magnitude - 0.0094 * distance - 1.454 * np.log10(distance)
    intensity3 = 7.494 + 0.744 * magnitude - 3.377 * np.log10((distance**3 + depth**3)**(1/3)) + 0.017 * depth
    intensity4 = 2.281 + 0.874 * magnitude - 0.618 * np.log10((1+distance**2/depth**2)**(1/2)) - 0.016 * ( (depth**2 + distance**2)**(1/2) - depth)

    return np.mean([intensity1, intensity2, intensity3, intensity4])

def affected_percentage( magnitude , depth , distance):

    intensity = intensity_calculation(magnitude, depth, distance)

    i_L = np.floor(intensity).astype(int)
    i_U = np.ceil(intensity).astype(int)

    # Constructing the dictionary based on the extracted data
    damage_data = {
        1: {
            "Definition": "I-Not felt",
            "No damage": 1,
            "Slight damage": 0,
            "Medium damage": 0,
            "Heavy damage and collapse": 0
        },
        2: {
            "Definition": "II-Weak",
            "No damage": 0.9999,
            "Slight damage": 0.0001,
            "Medium damage": 0,
            "Heavy damage and collapse": 0
        },
        3: {
            "Definition": "III-Weak",
            "No damage": 0.9998,
            "Slight damage": 0.0002,
            "Medium damage": 0,
            "Heavy damage and collapse": 0
        },
        4: {
            "Definition": "IV-Light",
            "No damage": 0.9995,
            "Slight damage": 0.0003,
            "Medium damage": 0.0002,
            "Heavy damage and collapse": 0.0001
        },
        5: {
            "Definition": "V-Moderate",
            "No damage": 0.999,
            "Slight damage": 0.0004,
            "Medium damage": 0.0002,
            "Heavy damage and collapse": 0.0004
        },
        6: {
            "Definition": "VI-Strong",
            "No damage": 0.995,
            "Slight damage": 0.0024,
            "Medium damage": 0.0022,
            "Heavy damage and collapse": 0.0004
        },
        7: {
            "Definition": "VII-Very strong",
            "No damage": 0.9383,
            "Slight damage": 0.0259,
            "Medium damage": 0.0267,
            "Heavy damage and collapse": 0.0091
        },
        8: {
            "Definition": "VIII-Damaging",
            "No damage": 0.8746,
            "Slight damage": 0.0531,
            "Medium damage": 0.0441,
            "Heavy damage and collapse": 0.0282
        },
        9: {
            "Definition": "IX-Destructive",
            "No damage": 0.4339,
            "Slight damage": 0.2275,
            "Medium damage": 0.1816,
            "Heavy damage and collapse": 0.1570
        },
        10: {
            "Definition": "X-Devastating",
            "No damage": 0.3251,
            "Slight damage": 0.1914,
            "Medium damage": 0.1529,
            "Heavy damage and collapse": 0.3306
        }
        ,
        11: {
            "Definition": "XI-Disastrous",
            "No damage": 0.2251,
            "Slight damage": 0.1914,
            "Medium damage": 0.2529,
            "Heavy damage and collapse": 0.3306
        }
        ,
        12: {
            "Definition": "XII-Catastrophic",
            "No damage": 0.1251,
            "Slight damage": 0.1914,
            "Medium damage": 0.2529,
            "Heavy damage and collapse": 0.4306
        }
    }

    # Extracting the data based on the intensity level
    data_L = damage_data[i_L]
    data_U = damage_data[i_U]

    L_lv1 = data_L["No damage"]
    L_lv2 = data_L["Slight damage"]
    L_lv3 = data_L["Medium damage"]
    L_lv4 = data_L["Heavy damage and collapse"]

    U_lv1 = data_U["No damage"]
    U_lv2 = data_U["Slight damage"]
    U_lv3 = data_U["Medium damage"]
    U_lv4 = data_U["Heavy damage and collapse"]

    # Calculate the probability of damage based on the distance
    increased_percentage = 0 * (L_lv1+U_lv1)/2 + 0.10 * (L_lv2+U_lv2)/2 + 0.50 * (L_lv3+U_lv3)/2 + 1 * (L_lv4+U_lv4)/2

    return increased_percentage