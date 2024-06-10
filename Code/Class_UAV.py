class Drone:
# drone : a Python object with the following attributes:
#                 ~ delta : profile drag coefficient
#                 ~ m_uav : the mass of the UAV (in kg)
#                 ~ m_batt: the mass of the battery (in kg)
#                 ~ U_tip : tip speed of the rotor (in m/s)
#                 ~ s     : rotor solidity
#                 ~ A     : rotor disk area (in m^2)
#                 ~ omega : blade angular velocity (in rad/s) 
#                 ~ r     : radius of the rotor (in meters)
#                 ~ k     : incremental correction factor to induced power
#                 ~ v_0   : mean rotor induced velocity in hover (in m/s)
#                 ~ d_r   : fuselage drag ratio
#                 ~ B_mass: energy capacity (density) per mass of the battery (in J/kg)
#                 ~ theta : depth of discharge (between 0 and 1)
#                 ~ v_max : maximum speed of the delivery drone (in m/s)
    def __init__(self, model, delta, m_uav, m_batt, U_tip, s, A, omega, r, k, v_0, d_r, B_mass, theta,  max_payload):
        self.model  = model
        self.delta  = delta
        self.m_uav  = m_uav
        self.m_batt = m_batt
        self.U_tip  = U_tip
        self.s      = s
        self.A      = A
        self.omega  = omega
        self.r      = r
        self.k      = k
        self.v_0    = v_0
        self.d_r    = d_r
        self.B_mass = B_mass
        self.theta  = theta
        self.max_payload = max_payload

    def display_info(self):
        print(f"{'Variable':<40} {'Value':<20} {'Units':<10}")
        print(f"{'Model:':<40} {self.model:<20}")
        print(f"{'Profile Drag Coefficient:':<40} {self.delta:<20}")
        print(f"{'UAV Mass:':<40} {self.m_uav:<20} {'kg':<10}")
        print(f"{'Battery Mass:':<40} {self.m_batt:<20} {'kg':<10}")
        print(f"{'Tip Speed of the Rotor:':<40} {self.U_tip:<20} {'m/s':<10}")
        print(f"{'Rotor Solidity:':<40} {self.s:<20}")
        print(f"{'Rotor Disk Area:':<40} {self.A:<20} {'m^2':<10}")
        print(f"{'Blade Angular Velocity:':<40} {self.omega:<20} {'rad/s':<10}")
        print(f"{'Rotor Radius:':<40} {self.r:<20} {'meters':<10}")
        print(f"{'Correction Factor to Induced Power:':<40} {self.k:<20}")
        print(f"{'Mean Rotor Induced Velocity in Hover:':<40} {self.v_0:<20} {'m/s':<10}")
        print(f"{'Fuselage Drag Ratio:':<40} {self.d_r:<20}")
        print(f"{'Energy Capacity per Mass of the Battery:':<40} {self.B_mass:<20} {'J/kg':<10}")
        print(f"{'Depth of Discharge:':<40} {self.theta:<20}")
        print(f"{'Max Payload:':<40} {self.max_payload:<20} {'kilograms':<10} \n")

class SmallDrone(Drone):
    def __init__(self, model):
        super().__init__(model, delta=0.012, m_uav=2.04 , m_batt=0.89, U_tip=120,
                         s=0.05, A=0.503, omega=300, r=0.4, k=0.1, v_0=4.03, d_r=0.6, B_mass=540000,
                         theta=0.8, max_payload=2.0)

class LargeDrone(Drone):
    def __init__(self, model):
        super().__init__(model, delta=0.012, m_uav=10 , m_batt=5.0, U_tip=150,
                         s=0.08, A=1.0, omega=250, r=1.0, k=0.15, v_0=6.0, d_r=0.8, B_mass=540000,
                         theta=0.8, max_payload=200)
        
# # Example usage:
# small_drone = SmallDrone("SmallX100")
# large_drone = LargeDrone("LargeX200")

# small_drone.display_info()
# print()
# large_drone.display_info()
