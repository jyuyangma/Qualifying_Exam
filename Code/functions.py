import numpy as np

def groundspeed_calculation( d, m_pack, drone, rho = 1.225, f = 1.2):
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
    - ground speed of the delivery drone (in m/s).
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
    coeff = [2*mu_4*d , 2*mu_2*d , -2*THETA , 2*mu_1*d, mu_3f + mu_3b]
    roots = np.roots(coeff)
    # Filter for positive real roots
    v_lists = [root for root in roots if np.isreal(root) and 0 < root.real]
    # Convert the list to a numpy array
    v_lists = np.array(v_lists).real
    if len(v_lists) == 0:
        air_speed = None  # No positive roots found within the maximum speed
    else:
        air_speed = np.min(v_lists)  # Choose the smallest positive root within the maximum spee

    return air_speed

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


def generate_all_locations(num_GP, cand_D, cand_LP, size=(1000, 1000)):
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
    """
    
    # Generate unique positions for gathering points, depots, and launch points
    total_positions = num_GP + cand_D + cand_LP
    unique_positions = generate_unique_positions(total_positions, size)
    unique_positions = list(unique_positions)

    # Split positions into gathering points, depots, and launch points
    GP_positions = unique_positions[:num_GP]
    D_positions = unique_positions[num_GP:num_GP+cand_D]
    LP_positions = unique_positions[num_GP+cand_D:]

    demand_list = np.zeros(num_GP)
    # Set customer demands on the map
    for i in range(num_GP):
        # Ensure demand is positive
        demand = np.random.uniform(low=0.02, high=20)
        demand_list[i] = demand

    return [demand_list, GP_positions], D_positions, LP_positions