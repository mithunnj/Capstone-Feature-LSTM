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

from utils.baseline_config import RAW_DATA_FORMAT, _FEATURES_SMALL_SIZE
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
    parser.add_argument("--extended_map",
                        default=False,
                        type=bool,
                        help="If true, compute features returns an extended map.")
    return parser.parse_args()


def load_seq_save_features(
        start_idx: int,
        sequences: List[str],
        save_dir: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,
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
    data = []

    
    # Enumerate over the batch starting at start_idx
    for seq in sequences[start_idx:start_idx + args.batch_size]:

        if not seq.endswith(".csv"):
            continue

        file_path = f"{args.data_dir}/{seq}"
        seq_id = int(seq.split(".")[0])

        # Compute physics features
        compute_physics_features(file_path)

        # Compute social and map features
        features, map_feature_helpers = compute_features(
            file_path, map_features_utils_instance,
            social_features_utils_instance)
        count += 1
        data.append([
            seq_id,
            features,
            map_feature_helpers["CANDIDATE_CENTERLINES"],
            map_feature_helpers["ORACLE_CENTERLINE"],
            map_feature_helpers["CANDIDATE_NT_DISTANCES"],
            map_feature_helpers["CANDIDATE_LANE_SEGMENTS"],
            map_feature_helpers["LANE_SEGMENTS_IN_BUBBLE"],
            map_feature_helpers["LANE_SEGMENTS_IN_FRONT"],
            map_feature_helpers["LANE_SEGMENTS_IN_BACK"],
        ])

        print(
            f"{args.mode}:{count}/{args.batch_size} with start {start_idx} and end {start_idx + args.batch_size}"
        )
        
        break # DEBUG REMOVE - Wanted to break out of this process after a single file

    data_df = pd.DataFrame(
        data,
        columns=[
            "SEQUENCE",
            "FEATURES",
            "CANDIDATE_CENTERLINES",
            "ORACLE_CENTERLINE",
            "CANDIDATE_NT_DISTANCES",
            "CANDIDATE_LANE_SEGMENTS",
            "LANE_SEGMENTS_IN_BUBBLE",
            "LANE_SEGMENTS_IN_FRONT",
            "LANE_SEGMENTS_IN_BACK",
        ],
    )

    # Save the computed features for all the sequences in the batch as a single file
    os.makedirs(save_dir, exist_ok=True)
    data_df.to_pickle(
        f"{save_dir}/forecasting_features_{args.mode}_{start_idx}_{start_idx + args.batch_size}.pkl"
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
        seq_path: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute social and map features for the sequence.

    Args:
        seq_path (str): file path for the sequence whose features are to be computed.
        map_features_utils_instance: MapFeaturesUtils instance.
        social_features_utils_instance: SocialFeaturesUtils instance.
    Returns:
        merged_features (numpy array): SEQ_LEN x NUM_FEATURES
        map_feature_helpers (dict): Dictionary containing helpers for map features

    """
   
    args = parse_arguments()
    df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

    # Get social and map features for the agent
    agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

    # Social features are computed using only the observed trajectory
    social_features = social_features_utils_instance.compute_social_features(
        df, agent_track, args.obs_len, args.obs_len + args.pred_len,
        RAW_DATA_FORMAT)

    # agent_track will be used to compute n-t distances for future trajectory,
    # using centerlines obtained from observed trajectory
    map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
        agent_track,
        args.obs_len,
        args.obs_len + args.pred_len,
        RAW_DATA_FORMAT,
        args.mode,
    )

    # Combine social and map features

    # If track is of OBS_LEN (i.e., if it's in test mode), use agent_track of full SEQ_LEN,
    # But keep (OBS_LEN+1) to (SEQ_LEN) indexes having None values
    if agent_track.shape[0] == args.obs_len:
        agent_track_seq = np.full(
            (args.obs_len + args.pred_len, agent_track.shape[1]), None)
        agent_track_seq[:args.obs_len] = agent_track
        merged_features = np.concatenate(
            (agent_track_seq, social_features, map_features), axis=1)
    else:
        merged_features = np.concatenate(
            (agent_track, social_features, map_features), axis=1)

    return merged_features, map_feature_helpers


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
        f"{args.feature_dir}/forecasting_features_{args.mode}.pkl")


if __name__ == "__main__":
    """Load sequences and save the computed features."""
    args = parse_arguments()

    start = time.time()

    map_features_utils_instance = MapFeaturesUtils()
    social_features_utils_instance = SocialFeaturesUtils()

    sequences = os.listdir(args.data_dir)
    temp_save_dir = tempfile.mkdtemp()

    num_sequences = _FEATURES_SMALL_SIZE if args.small else len(sequences)

    Parallel(n_jobs=-2)(delayed(load_seq_save_features)(
        i,
        sequences,
        temp_save_dir,
        map_features_utils_instance,
        social_features_utils_instance,
    ) for i in range(0, num_sequences, args.batch_size))
    merge_saved_features(temp_save_dir)
    shutil.rmtree(temp_save_dir)

    print(
        f"Feature computation for {args.mode} set completed in {(time.time()-start)/60.0} mins"
    )
