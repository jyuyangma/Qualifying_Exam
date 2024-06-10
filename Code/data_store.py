import os
import pandas as pd
import numpy as np

def save_to_csv(set_NO , type, scenario_number, demand_array = None, location_array = None , weather_array = None , printFlag = False):
    demand_flag = False
    weather_flag = False
    EQ_flag = False
    save_Flag = True
    # Check if the map_array and location_array are provided
    if type == 'GP':
        location_type = 'gather_points'
        demand_flag = True
    elif type == 'DP':
        location_type = 'depots'
    elif type == 'LP':
        location_type = 'launch_points'
    elif type == 'EQ':
        location_type = 'earthquake'
        EQ_flag = True
    else:
        location_type = 'weather'
        weather_flag = True

    # Ensure the 'data' directory exists
    if not os.path.exists('./set_' + f'{set_NO}' + '_data/'):
        os.makedirs('./set_' + f'{set_NO}' + '_data/')
        
    # Define the file paths
    if demand_flag:
        file1_path = './set_' + f'{set_NO}' + '_data/' +  f"scenario_{scenario_number}_demand_{location_type}.csv"
        if os.path.exists(file1_path):
            save_Flag = False
        else:
            # Save the map array to a CSV file
            df_map = pd.DataFrame(demand_array)
            df_map.to_csv(file1_path, index=False, header=False)

    if weather_flag:
        file2_path = './set_' + f'{set_NO}' + '_data/' +  f"scenario_{scenario_number}_{location_type}.csv"
        # Save the map array to a CSV file
        df_location = pd.DataFrame(weather_array).T
    elif EQ_flag:
        file2_path = './set_' + f'{set_NO}' + '_data/' + f"scenario_{scenario_number}_locations_{location_type}.csv"
        df_location = pd.DataFrame(location_array).T
    else:
        file2_path = './set_' + f'{set_NO}' + '_data/' + f"locations_{location_type}.csv"
        df_location = pd.DataFrame(location_array)

    # Check if the files already exist
    if os.path.exists(file2_path):
        save_Flag = False
    else:
        df_location.to_csv(file2_path, index=False, header=False)

    if printFlag:
        if save_Flag:
            print(f'{type} Data arrays have been saved')
            return
        else:
            print(f'{type} Data arrays already exist, new arrays not saved')
            return


def read_csv_files(set_NO , type, S , printFlag = False):
    
    location_list = []
    weather_list = []
    demand_flag = False
    weather_flag = False
    EQ_flag = False
    readFlag = True

    if type == 'GP':
        demand_list = []
        location_type = 'gather_points'
        demand_flag = True
    elif type == 'DP':
        location_type = 'depots'
    elif type == 'LP':
        location_type = 'launch_points'
    elif type == 'EQ':
        location_type = 'earthquake'
        EQ_flag = True
    else:
        location_type = 'weather'
        weather_flag = True

    for s in range(S):

        if demand_flag:
            file1_path = './set_' + f'{set_NO}' + '_data/' +  f"scenario_{s+1}_demand_{location_type}.csv"
            if os.path.exists(file1_path):
                df1 = pd.read_csv(file1_path, header=None)
                # values1 = df1.values.flatten().tolist()
                # values1 = [float(val) if isinstance(val, str) else val for val in values1]
                demand_list.append(np.asarray(df1))
            else:
                readFlag = False
        
        if weather_flag:
            file2_path = './set_' + f'{set_NO}' + '_data/' + f"scenario_{s+1}_{location_type}.csv"
            if os.path.exists(file2_path):
                df2 = pd.read_csv(file2_path, header=None)
                # values2 = df2.values.flatten().tolist()
                # values2 = [float(val) if isinstance(val, str) else val for val in values2]
                weather_list.append(np.asarray(df2))
            else:
                readFlag = False
        elif EQ_flag:
            file2_path = './set_' + f'{set_NO}' + '_data/' + f"scenario_{s+1}_locations_{location_type}.csv"
            if os.path.exists(file2_path):
                df2 = pd.read_csv(file2_path, header=None)
                # values2 = df2.values.flatten().tolist()
                # values2 = [float(val) if isinstance(val, str) else val for val in values2]
                location_list.append(np.asarray(df2))
            else:
                readFlag = False
        else:
            file2_path = './set_' + f'{set_NO}' + '_data/' + f"locations_{location_type}.csv"

    if os.path.exists(file2_path):
        df2 = pd.read_csv(file2_path, header=None)
        # values2 = df2.values.flatten().tolist()
        # values2 = [float(val) if isinstance(val, str) else val for val in values2]
        location_list.append(np.asarray(df2))
    else:
        readFlag = False

    if printFlag:
        if readFlag:
            print(f'{type} Data arrays have been read')
        else:
            print(f'{type} Data arrays do not exist')
            return

    if demand_flag:
        return demand_list, location_list
    elif weather_flag:
        return weather_list
    else:
        return location_list