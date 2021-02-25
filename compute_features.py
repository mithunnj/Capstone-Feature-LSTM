"""This module is used for computing social and map features for motion forecasting baselines.

Example usage:
    $ python compute_features.py --data_dir ~/val/data 
        --feature_dir ~/val/features --mode val
    $ python compute_features.py --data_dir ./data/forecasting_sample/data --feature_dir ./computed_features --mode train
"""

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

from argoverse.map_representation.map_api import ArgoverseMap

from utils.baseline_config import RAW_DATA_FORMAT, _FEATURES_SMALL_SIZE, FEATURE_TYPES
from utils.map_features_utils import MapFeaturesUtils
from utils.social_features_utils import SocialFeaturesUtils

# Set global FPs
CURR_DIR = os.getcwd()
DATA_DIR = CURR_DIR + "/data"
COMPUTE_FEATURES_SAVE_DIR = CURR_DIR + "/computed_features"


def parse_arguments() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="Directory where the sequences (csv files) are saved",
    )
    parser.add_argument(
        "--feature_dir",
        default="",
        type=str,
        help="Directory where the computed features are to be saved",
    )
    parser.add_argument("--mode",
                        required=True,
                        type=str,
                        help="train/val/test")
    parser.add_argument("--feature_type",
                        required=True,
                        type=str,
                        help="One of candidates_lanes/physics/semantic_map/lead_agent (stored in config).",
                        choices=FEATURE_TYPES.keys())
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument("--small",
                        action="store_true",
                        help="If true, a small subset of data is used.")
    parser.add_argument("--multi_agent",
                        default=False,
                        action="store_true",
                        help="If true, compute features will only compute the lane selection features.")
    return parser.parse_args()


def load_seq_save_features(
        start_idx: int,
        sequences: List[str],
        save_dir: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,
        argoverse_map_api_instance: ArgoverseMap
) -> None:
    """Load sequences, compute features, and save them.
    
    Args:
        start_idx : Starting index of the current batch
        sequences : Sequence file names
        save_dir: Directory where features for the current batch are to be saved
        map_features_utils_instance: MapFeaturesUtils instance
        social_features_utils_instance: SocialFeaturesUtils instance

    """
    count = 0
    args = parse_arguments()
    all_rows = []

    
    # Enumerate over the batch starting at start_idx
    for seq in sequences[start_idx:start_idx + args.batch_size]:

        if not seq.endswith(".csv"):
            continue
        
        seq_file_path = f"{args.data_dir}/{seq}"
        seq_id = int(seq.split(".")[0])

        # Compute physics features
        compute_physics_features(file_path)

        # Compute social and map features
        feature_columns, scene_rows = compute_features(
            seq_id, seq_file_path, map_features_utils_instance,
            social_features_utils_instance,
            argoverse_map_api_instance)
        count += 1

        # Merge the features for all agents and all scenes
        all_rows.extend(scene_rows)

        print(
            f"{args.mode}/{args.feature_type}:{count}/{args.batch_size} with start {start_idx} and end {start_idx + args.batch_size}"
        )
        
        break # DEBUG REMOVE - Wanted to break out of this process after a single file

    assert "SEQUENCE" in feature_columns, "Missing feature column: SEQUENCE"
    assert "TRACK_ID" in feature_columns, "Missing feature column: TRACK_ID"
    
    # Create dataframe for this batch
    data_df = pd.DataFrame(
        all_rows,
        columns=feature_columns,
    )

    # Save the computed features for all the sequences in the batch as a single file
    os.makedirs(save_dir, exist_ok=True)
    data_df.to_pickle(
        f"{save_dir}/forecasting_features_{args.mode}_{args.feature_type}_{start_idx}_{start_idx + args.batch_size}.pkl"
    )

def compute_physics_features(seq_path):

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


def compute_features(
        seq_id: int,
        seq_path: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,
        avm: ArgoverseMap
) -> Tuple[list, pd.DataFrame]:
    """Compute features for all.

    Args:
        seq_path (str): file path for the sequence whose features are to be computed.
        map_features_utils_instance: MapFeaturesUtils instance.
        social_features_utils_instance: SocialFeaturesUtils instance.
    Returns:
        columns (list of strings): ["SEQUENCE", "TRACK_ID", "FEATURE_1", ..., "FEATURE_N"]
        features_dataframe (pandas dataframe): Pandas dataframe where each cell is list of features or a list of features per centerline

    """
   
    args = parse_arguments()
    
    scene_df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

    columns = list()
    all_feature_rows = dict()

    # Compute agent list based on args.multi_agent
    agent_list = []
    if args.multi_agent:

        # Construct list of agents in the scene
        agent_list = scene_df["TRACK_ID"].unique().tolist()
    
    else:
        # Construct a list of only the Argo AGENT
        agent_list = scene_df[scene_df["OBJECT_TYPE"] == "AGENT"]["TRACK_ID"].unique().tolist()

    # Call function for the given feature type
    if args.feature_type == "testing": # Temp values for testing
        columns = ["SEQUENCE", "TRACK_ID", "MY_FEATURE"]
        all_feature_rows = [ [ seq_id, agent_list[0], 1.0 ] ]

    elif args.feature_type == "candidate_lanes": # SASHA add lane candidate function here
        columns, all_feature_rows = None, None

    elif args.feature_type == "physics": # MITHUN add physics function call here
        columns, all_feature_rows = None, None

    elif args.feature_type == "semantic_map": # FARID add semantic map function call here
        columns, all_feature_rows = None, None

    elif args.feature_type == "lead_agent": # DENIZ add semantic map function call her
        columns, all_feature_rows = None, None

    else:
        assert False, "Invalid feature type."
    
    return columns, all_feature_rows


def merge_saved_features(batch_save_dir: str) -> None:
    """Merge features saved by parallel jobs.

    Args:
        batch_save_dir: Directory where features for all the batches are saved.

    """
    args = parse_arguments()
    feature_files = os.listdir(batch_save_dir)
    all_features = []
    for feature_file in feature_files:
        if not feature_file.endswith(".pkl") or args.mode not in feature_file:
            continue
        file_path = f"{batch_save_dir}/{feature_file}"
        df = pd.read_pickle(file_path)
        all_features.append(df)

        # Remove the batch file
        os.remove(file_path)

    all_features_df = pd.concat(all_features, ignore_index=True)

    # Save the features for all the sequences into a single file
    all_features_df.to_pickle(
        f"{args.feature_dir}/forecasting_features_{args.mode}_{args.feature_type}.pkl")


if __name__ == "__main__":
    """Load sequences and save the computed features."""
    args = parse_arguments()

    start = time.time()

    # Warn if the data directory does not contain the mode
    if args.mode not in args.data_dir:
        print("WARNING: Mode does not match data directory name.")
    
    # Check if feature does not support multi-agent computation
    if args.multi_agent:
        # Check the feature supports multi agent
        assert FEATURE_TYPES[args.feature_type]["supports_multi_agent"], "This feature does not support computing for multiple agents in the scene."

    # Initialize Argoverse Map API and util functions
    map_features_utils_instance = MapFeaturesUtils()
    argoverse_map_api_instance = None # ArgoverseMap()
    social_features_utils_instance = None # SocialFeaturesUtils()

    # Get list of scenes and create temp directory
    sequences = os.listdir(args.data_dir)
    temp_save_dir = tempfile.mkdtemp()

    # If the small flag is set, restrict the number os sequences (for testing)
    num_sequences = _FEATURES_SMALL_SIZE if args.small else len(sequences)

    # Compute features in parallel batches
    Parallel(n_jobs=-2)(delayed(load_seq_save_features)(
        i,
        sequences,
        temp_save_dir,
        map_features_utils_instance,
        social_features_utils_instance,
        argoverse_map_api_instance
    ) for i in range(0, num_sequences, args.batch_size))

    # Switch the above parrallel call to this if visualizing with cProfile
    # load_seq_save_features(
    #     0,
    #     sequences,
    #     temp_save_dir,
    #     map_features_utils_instance,
    #     social_features_utils_instance,
    #     argoverse_map_api_instance
    # )

    # Merge the batched features and clean up
    merge_saved_features(temp_save_dir)
    shutil.rmtree(temp_save_dir)

    print(
        f"Feature computation for {args.mode} set completed in {(time.time()-start)/60.0} mins"
    )
