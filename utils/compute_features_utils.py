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
import math
import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle as pkl

# Set global FPs
CURR_DIR = os.getcwd() if os.getcwd().split('/')[-1] == "Capstone_Social_LSTM" else os.path.dirname(os.getcwd())
DATA_DIR = CURR_DIR + "/data"
COMPUTE_FEATURES_SAVE_DIR = CURR_DIR + "/computed_features"

def calc_magnitude(x, y):
    '''
    Input: x,y <Float> - x,y components of physics vectors (ex. vel_x, vel_y)
    Output: <Float>
    
    Description: Given the x and y components of physics vectors, this will return the normalized magnitude.
        ex. Given velocity in the x, y direction, the scalar speed will be the result.
    '''
    squared_terms = (x**2) + (y**2)

    return math.sqrt(squared_terms)

def compute_physics_features(seq_path):
    '''
    Input: seq_path <str> - This is the filepath of the .csv data file that is parsed   
        by the Argoverse utils.
    Output: None
    Description: Given a .csv data file, this will store the following physics features into a .pkl file in the ~project_root/computed_features directory:
        - timestamp, track_id (agent_id), x, y, vel_x, vel_y, acc_x, acc_y, jerk_x, jerk_y
    '''

    # Helper function to calculate time varying parameters for agents
    def rate_of_change(data, time, index):
        param_change = data[index] - data[index-1]
        time_change = time[index] - time[index-1]

        return param_change/time_change

    # Helper function calculate yaw with delta_x, delta_y
    def calc_yaw(x, y, index):
        dX = x[index] - x[index-1]
        dY = y[index] - y[index-1]

        yaw = math.atan2(dY, dX)

        return yaw

    df = pd.read_csv(seq_path) # Load .csv data

    # Per agent information 
    column_headings = ["SEQUENCE", "TRACK_ID", "TIMESTAMP", "X", "Y", "VEL_X", "VEL_Y", "ACC_X", "ACC_Y", "JERK_X", "JERK_Y", "YAW", "YAW_RATE"] 

    # Get all unique track_ids (represents the agents in the scene - .csv file)
    agent_ids = df["TRACK_ID"].unique().tolist()

    # Data store for .csv file
    total_data = list()

    for agent in agent_ids:

        agent_df = df[df['TRACK_ID'] == agent] # Agent specific data from the .csv dataframe

        # NEW DATA STRUCTURE
        agent_data = [seq_path, agent] # Initiate agent data structure with SEQUENCE_ID and TRACK_ID info
        timestamp, x, y, vel_x, vel_y, acc_x, acc_y, jerk_x, jerk_y, yaw, yaw_rate = list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()

        for index, row in agent_df.iterrows(): # Each row represents a specific timestep of data for a specific agent (track_id)


            # Load data
            timestamp.append(row['TIMESTAMP'])
            x.append(row["X"])
            y.append(row["Y"])

            index = len(x)-1 if len(x) == len(y) else sys.exit('ERROR: compute_physics_features, x and y data length not the same')

            # Update velocity, yaw after there are at least 2 position information
            if len(timestamp) >= 2:
                # Calculate velocity
                VEL_X = rate_of_change(x, timestamp, index)
                VEL_Y = rate_of_change(y, timestamp, index)
                YAW = calc_yaw(x, y, index) 

                if len(timestamp) == 2: # If this is the first calculaton of velocity, set the first index to the same vel value instead of NONE
                    vel_x.extend((VEL_X, VEL_X))
                    vel_y.extend((VEL_Y, VEL_Y))
                    yaw.extend((YAW, YAW))
                else:
                    vel_x.append(VEL_X)
                    vel_y.append(VEL_Y)
                    yaw.append(YAW)

            # Update acceleration after there are at least 2 velocity information
            # Update yaw rate after there are at least 2 yaw values
            if len(timestamp) >= 3:
                ACC_X = rate_of_change(vel_x, timestamp, index)
                ACC_Y = rate_of_change(vel_y, timestamp, index)
                YAW_RATE = rate_of_change(yaw, timestamp, index)

                if len(timestamp) == 3:
                    acc_x.extend((ACC_X, ACC_X, ACC_X))
                    acc_y.extend((ACC_Y, ACC_Y, ACC_Y))
                    yaw_rate.extend((YAW_RATE, YAW_RATE, YAW_RATE))
                else:
                    acc_x.append(ACC_X)
                    acc_y.append(ACC_Y)
                    yaw_rate.append(YAW_RATE)

            # Update jerk after there are at least 2 acceleration information
            if len(timestamp) >= 4:
                JERK_X = rate_of_change(acc_x, timestamp, index)
                JERK_Y = rate_of_change(acc_y, timestamp, index)

                if len(timestamp) == 4:
                    jerk_x.extend((JERK_X, JERK_X, JERK_X, JERK_X))
                    jerk_y.extend((JERK_Y, JERK_Y, JERK_Y, JERK_Y))
                else:
                    jerk_x.append(JERK_X)
                    jerk_y.append(JERK_Y)

        agent_data.extend((timestamp, x, y, vel_x, vel_y, acc_x, acc_y, jerk_x, jerk_y, yaw, yaw_rate))
        total_data.append(agent_data)

    return column_headings, total_data

def save_ml_physics_features(data, mode):
    '''
    data: This is an array from the compute_physics features saved in the format required for training
        Each row is a track_id, and all the physics features are can be passed in sub lists using the following indices:
    mode: Parsed from args.mode and can be any of the following (specified by user): train/val/test

        Given a sublist of the main data array (each row represents a single track id), you can parse it in the following way:

        - Index 0 <str>: "SEQUENCE"
        - Index 1 <str>: "TRACK_ID"
        - Index 2 <list>: "TIMESTAMP"
        - Index 3 <list>: "X"
        - Index 4 <list>: "Y"
        - Index 5 <list>: "VEL_X"
        - Index 6 <list>: "VEL_Y"
        - Index 7 <list>: "ACC_X"
        - Index 8 <list>: "ACC_Y"
        - Index 9 <list>: "JERK_X"
        - Index 10 <list>: "JERK_Y"
        - Index 11 <list>: "YAW"
        - Index 12 <list>: "YAW_RATE"
    '''
    fp = str(data[0][0])
    seq = fp.split('/')[-1].split('.')[0] # Parse the file name of the data file
    df = pd.read_csv(fp, dtype={"TIMESTAMP": str})
    agent_id = df[df['OBJECT_TYPE'] == "AGENT"]["TRACK_ID"].tolist()[0] # Identify the AGENT_ID in this data file
    
    # Save all the track_id information 
    with open(COMPUTE_FEATURES_SAVE_DIR + '/forecasting_features_{}_physics_all.pkl'.format(mode), 'wb') as filehandle_A:
        pkl.dump(data, filehandle_A)

    filehandle_A.close()

    # Save agent information only
    agent_data = [info for info in data if info[1] == agent_id]
    with open(COMPUTE_FEATURES_SAVE_DIR + '/forecasting_features_{}_physics.pkl'.format(mode), 'wb') as filehandle_B:
        pkl.dump(agent_data, filehandle_B)

    filehandle_B.close()

    return

'''
#NOTE: UNCOMMENT THIS IF YOU WANT TO KNOW HOW TO LOAD PHYSICS DATA
def sample_open_physics_ml_features():
    with open(ML_FEATURES_SAVE_DIR + '/2645_PHYSICS_ML_FEATURES.pkl', 'rb') as f:
        x = pkl.load(f)

        print("Track ID info: {}\n".format(x[0][1]))
        print("X for Track ID: {}".format(x[0][3]))

    return

sample_open_physics_ml_features()
'''
