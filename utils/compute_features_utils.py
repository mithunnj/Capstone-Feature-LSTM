'''
Description: Contains utils code for Capstone project changes to the compute_features.py 
    functionality.
'''
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple
import sys

import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle as pkl

# Set global FPs
CURR_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(CURR_DIR)
DATA_DIR = ROOT_DIR + "/data"
COMPUTE_FEATURES_SAVE_DIR = ROOT_DIR + "/computed_features"

def compute_physics_features(seq_path):
    '''
    Input: seq_path <str> - This is the filepath of the .csv data file that is parsed   
        by the Argoverse utils.
    Output: None

    Description: Given a .csv data file, this will store the following physics features into a .pkl file in the ~project_root/computed_features directory:
        - timestamp, track_id (agent_id), x, y, vel_x, vel_y, acc_x, acc_y, jerk_x, jerk_y
    '''

    # Helper function to calculate time varying parameters for agents
    def rate_of_change(data, param, index):
        param_change = data[index][param] - data[index-1][param]
        time_change = data[index]['TIMESTAMP'] - data[index-1]['TIMESTAMP']

        return param_change/time_change

    df = pd.read_csv(seq_path) # Load .csv data
    # Per agent information 
    per_agent_info_headings = ["TIMESTAMP", "TRACK_ID", "X", "Y", "VEL_X", "VEL_Y", "ACC_X", "ACC_Y", "JERK_X", "JERK_Y"] 

    # Get all unique track_ids (represents the agents in the scene - .csv file)
    agent_ids = df["TRACK_ID"].unique().tolist()

    # Data store for .csv file
    total_data = dict()

    for agent in agent_ids:

        agent_info = list() # List to contain the agent information over a series of timesteps
        agent_df = df[df['TRACK_ID'] == agent] # Agent specific data from the .csv dataframe
            
        for index, row in agent_df.iterrows(): # Each row represents a specific timestep of data for a specific agent (track_id)
            
            data = {key: None for key in per_agent_info_headings} # Initialize empty data structure to store row data

            # Load data
            data['TIMESTAMP'] = row['TIMESTAMP']
            data['TRACK_ID'] = row['TRACK_ID']
            data["X"] = row["X"]
            data["Y"] = row["Y"]

            # Store agent data
            agent_info.append(data)

            # Update time varying data
            index = 0 if len(agent_info)  == 0 else len(agent_info)-1

            # Update velocity after there are at least 2 position information
            if len(agent_info) >= 2:
                agent_info[index]["VEL_X"] = rate_of_change(agent_info, "X", index)
                agent_info[index]["VEL_Y"] = rate_of_change(agent_info, "Y", index)

            # Update acceleration after there are at least 2 velocity information
            if len(agent_info) >= 3:
                agent_info[index]["ACC_X"] = rate_of_change(agent_info, "VEL_X", index)
                agent_info[index]["ACC_Y"] = rate_of_change(agent_info, "VEL_Y", index)

            # Update jerk after there are at least 2 acceleration information
            if len(agent_info) >= 4:
                agent_info[index]["JERK_X"] = rate_of_change(agent_info, "ACC_X", index)
                agent_info[index]["JERK_Y"] = rate_of_change(agent_info, "ACC_Y", index)

        total_data[agent] = agent_info # Store all timestep'd data for this agent

    # Save data to .pkl file in the COMPUTED_FEATURES_DIR 
    file_name = "computed_physics_features_file_{}.pkl".format(seq_path.split('/')[-1].split('.')[0])
    save_dir = COMPUTE_FEATURES_SAVE_DIR + "/{}".format(file_name)
    pkl_file = open(save_dir, "wb+")
    pkl.dump(total_data, pkl_file)
    pkl_file.close()

    return