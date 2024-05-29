import os
import pandas as pd
import numpy as np

# Ensure the 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

def save_to_csv(type, scenario_number, demand_array = None, location_array = None):
    demand_flag = False
    # Check if the map_array and location_array are provided
    if type == 'GP':
        location_type = 'gather_points'
        demand_flag = True
    elif type == 'DP':
        location_type = 'depots'
    elif type == 'LP':
        location_type = 'launch_points'
    else:
        print('Invalid location type. The function will not run.')
        return
    
    # Define the file paths
    if demand_flag:
        file1_path = './data/' +  f"scenario_{scenario_number}_demand_{location_type}.csv"
        # Save the map array to a CSV file
        df_map = pd.DataFrame(demand_array)
        df_map.to_csv(file1_path, index=False, header=False)

    file2_path = './data/' + f"scenario_{scenario_number}_locations_{location_type}.csv"

    # Check if the files already exist
    if os.path.exists(file2_path):
        print('Files already exist. The function will not run.')
        return

    df_location = pd.DataFrame(location_array)
    df_location.to_csv(file2_path, index=False, header=False)

    print(f'Data arrays have been saved')


def read_csv_files(type, S):
    
    location_list = []
    demand_flag = False

    if type == 'GP':
        demand_list = []
        location_type = 'gather_points'
        demand_flag = True
    elif type == 'DP':
        location_type = 'depots'
    elif type == 'LP':
        location_type = 'launch_points'
    else:
        print('Invalid location type. The function will not run.')
        return

    for s in range(S):

        if demand_flag:
            file1_path = './data/' +  f"scenario_{s+1}_demand_{location_type}.csv"
            if os.path.exists(file1_path):
                df1 = pd.read_csv(file1_path, header=None)
                # values1 = df1.values.flatten().tolist()
                # values1 = [float(val) if isinstance(val, str) else val for val in values1]
                demand_list.append(np.asarray(df1))
            else:
                print(f'{file1_path} does not exist.')

        file2_path = './data/' + f"scenario_{s+1}_locations_{location_type}.csv"

        if os.path.exists(file2_path):
            df2 = pd.read_csv(file2_path, header=None)
            # values2 = df2.values.flatten().tolist()
            # values2 = [float(val) if isinstance(val, str) else val for val in values2]
            location_list.append(np.asarray(df2))
        else:
            print(f'{file2_path} does not exist.')

    print(f'Data arrays read complete')
    if demand_flag:
        return demand_list, location_list
    else:
        return location_list