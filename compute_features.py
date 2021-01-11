"""This module is used for computing social and map features for motion forecasting baselines.
Example usage:
    $ python compute_features.py --data_dir ~/val/data 
        --feature_dir ~/val/features --mode val
"""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple

import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # remove .loc warning print
import pickle as pkl

from utils.baseline_config import RAW_DATA_FORMAT, _FEATURES_SMALL_SIZE
from utils.map_features_utils import MapFeaturesUtils
from utils.social_features_utils import SocialFeaturesUtils
import math
import matplotlib.pyplot as plt


def parse_arguments() -> Any:
    """Parse command line arguments."""
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument("--social_test", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate test mode for social features compute.")
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

        # Compute social and map features
        #features, map_feature_helpers = compute_features(
        compute_features(
            file_path, map_features_utils_instance,
            social_features_utils_instance)
        count += 1
    #     data.append([
    #         seq_id,
    #         features,
    #         map_feature_helpers["CANDIDATE_CENTERLINES"],
    #         map_feature_helpers["ORACLE_CENTERLINE"],
    #         map_feature_helpers["CANDIDATE_NT_DISTANCES"],
    #     ])

    #     print(
    #         f"{args.mode}:{count}/{args.batch_size} with start {start_idx} and end {start_idx + args.batch_size}"
    #     )

    # data_df = pd.DataFrame(
    #     data,
    #     columns=[
    #         "SEQUENCE",
    #         "FEATURES",
    #         "CANDIDATE_CENTERLINES",
    #         "ORACLE_CENTERLINE",
    #         "CANDIDATE_NT_DISTANCES",
    #     ],
    # )

    # Save the computed features for all the sequences in the batch as a single file
    # os.makedirs(save_dir, exist_ok=True)
    # data_df.to_pickle(
    #     f"{save_dir}/forecasting_features_{args.mode}_{start_idx}_{start_idx + args.batch_size}.pkl"
    # )



###########find_velocities of a given track,  returns an array of velocities. should be optimized for python via numpy or sth else, 
########### or we can move to C entirely. for now, we are only concerned with finding how many agents we are dropping
def compute_vel_track(            
            av_track: np.ndarray,
            obs_len: int) -> List[float]:
    
    av_params = list()

    for i in range(obs_len):
        if (i < (obs_len-1)):
            # Agent coordinates, to find dx, dy
            ## Based on a single element of av_track np.ndarray in the form: ['315968204.20525765' '00000000-0000-0000-0000-000000000000' 'AV' 419.354293323178 1125.927382712982 'MIA']
            av_x_t1, av_y_t1 = av_track[i, -3], av_track[i, -2]

            av_x_t2, av_y_t2 = av_track[i+1, -3], av_track[i+1, -2]

            # timestamps to find dt
            time_t1 = float(av_track[i, 0])
            time_t2 = float(av_track[i+1, 0])

            # find velocities
            vel_x = (av_x_t2 - av_x_t1)/(time_t2 - time_t1)
            vel_y = (av_y_t2 - av_y_t1)/(time_t2 - time_t1)

            av_inst_velo = math.sqrt(vel_x**2 + vel_y**2)
           
            # Store AV's x, y, vel for each timestep
            info = {
                    "x": av_x_t1,
                    "y": av_y_t1,
                    "vel": av_inst_velo
                }

            av_params.append(info)
        else:
            break


    return av_params

def check_safezone_overlap(av_info, agent_info, time_diff, default_radius=1, viz_regions=False):
    '''
    Input:
        - av_info, agent_info <dict>: Contains the x,y coordinates and velocity magnitude of AV and agent.
        - default_radius <int>: This will set the default radius of the safezone that is constructred around each actor.
        - viz_region <bool>: This is for debugging purposes. If TRUE, this will safe a plot of the two actors with their safezone

    Output: 
        - <bool>: If the two safezones intersect will return True otherwise False.
    '''

    # Option to visualize regions
    if viz_regions:

        #define circles
        av_actual_circle = plt.Circle((av_info["x"], av_info["y"]), radius=av_info['vel'], color='r', fill=False)
        av_min_circle = plt.Circle((av_info["x"], av_info["y"]), radius=default_radius, color='b', fill=False)
        agent_actual_circle = plt.Circle((agent_info["x"], agent_info["y"]), radius=agent_info['vel'], color='r', fill=False)
        agent_min_circle = plt.Circle((agent_info["x"], agent_info["y"]), radius=default_radius, color='b', fill=False)

        fig, ax = plt.subplots() 
        ax.cla() # clear things for fresh plot
        ax.set_xlim( (av_info["x"] - 15 if av_info["x"] < agent_info["x"] else agent_info["x"] - 15), (av_info["x"] + 15 if av_info["x"] > agent_info["x"] else agent_info["x"] + 15) )
        ax.set_ylim( (av_info["y"] - 15 if av_info["y"] < agent_info["y"] else agent_info["y"] - 15), (av_info["y"] + 15 if av_info["y"] > agent_info["y"] else agent_info["y"] + 15) )

        # add actor coordinates to plot
        ax.plot(av_info["x"], av_info["y"], "o", color="black")
        ax.plot(agent_info["x"], agent_info["y"], "o", color="black")

        #add circles to plot
        plt.gca().add_artist(av_actual_circle)
        plt.gca().add_artist(av_min_circle)
        plt.gca().add_artist(agent_actual_circle)
        plt.gca().add_artist(agent_min_circle)

        fig.savefig('check_safezone_overlap.png')

    # Set min radius based on velocity of AV and agent
    av_vel = default_radius if av_info["vel"] < default_radius else av_info["vel"]
    agent_vel = default_radius if agent_info["vel"] < default_radius else agent_info["vel"]

    # Calculate interesection: Inspired by: https://stackoverflow.com/questions/8367512/how-do-i-detect-intersections-between-a-circle-and-any-other-circle-in-the-same
    intersection_result = math.sqrt( ((av_info["x"] - agent_info["x"])**2) + ((av_info["y"] - agent_info["y"])**2) ) <= ( (av_vel + agent_vel)*time_diff)

    return intersection_result

def compute_features(
        seq_path: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,
#) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
) -> None:
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

    if args.social_test:
        
        agents_social_features = dict() # Store of social feature computations for each 

        initial_agent_ids = df["TRACK_ID"].values # All the track_ids (unique actor identifier in .csv dataset)
        initial_agent_ids = list(dict.fromkeys(initial_agent_ids)) # Remove duplicates from IDs
        final_agent_ids = initial_agent_ids.copy() # This is the list of agents that will get removed from the filtering process below.
        initial_count, final_count = len(initial_agent_ids), 0

        df_copy = df.copy() # Copy dataframe to avoid changing the original data structure

        # Identify TRACK_ID for AV from dataframe information
        av_id = df_copy.loc[df['OBJECT_TYPE'] == 'AV', 'TRACK_ID'].iloc[0]
        # here first find the AV track array
        av_track = df_copy[df_copy["OBJECT_TYPE"] == "AV"].values

        # get velocities
        av_params = compute_vel_track(av_track, args.obs_len)
        
        # Remove all irrelevant agents that do not have overlapping bounding regions with AV
        initial_agent_ids.remove(av_id)
        final_agent_ids.remove(av_id)

        for agent in initial_agent_ids:
            # here loop thru the agent track id's and get their tracks, then loop thru these x,y values to first compute velocities, 
            # then from the velocities, construct the boxes at every timestep for the current agent and then the AV, to compare overlap.
            # if any overlap exists, continue
            # make a counter to show how many we dropped, tune it to drop enough if we have very few dropped.


            # make sure agent track is same as the observed time length, if not pad it with zeros
            current_agent_track = df_copy[df_copy["TRACK_ID"] == agent].values
            agent_ts = np.sort(np.unique(df_copy["TIMESTAMP"].values))

            if agent_ts.shape[0] == args.obs_len:
                df_obs = df_copy
                current_agent_track_obs = current_agent_track
            else:
                # Get obs dataframe and agent track
                df_obs = df_copy[df_copy["TIMESTAMP"] < agent_ts[args.obs_len]]
                assert (np.unique(df_obs["TIMESTAMP"].values).shape[0] == args.obs_len
                        ), "Obs len mismatch"
                current_agent_track_obs = current_agent_track[:args.obs_len]


            # Remove agent_id if there is not enough time stamped data, means that real time data does not have enough length for this actor,
            # so lstm cannot make a prediction.
            if (len(current_agent_track_obs) < args.obs_len):
                final_agent_ids.remove(agent)
                
                continue

            # i from 0 to 18 because i+1 increment for vel calculations will result in going beyond 0-19 range of tracks
            agent_seq_overlap = list()

            i = args.obs_len - 2

            # Agent coordinates for two timesteps
            agent_x_t1, agent_y_t1 = current_agent_track_obs[i][-3], current_agent_track_obs[i][-2]
            agent_x_t2, agent_y_t2 = current_agent_track_obs[i+1][-3], current_agent_track_obs[i+1][-2]
            av_x_t2, av_y_t2 = av_track[i+1][-3], av_track[i+1][-2]

            # timesteps to find dt
            time_t1 = float(current_agent_track_obs[i,0])
            time_t2 = float(current_agent_track_obs[i+1,0])

            # calculate velocity for agent over two timesteps
            vel_x = (agent_x_t2 - agent_x_t1)/(time_t2 - time_t1)
            vel_y = (agent_y_t2 - agent_y_t1)/(time_t2 - time_t1)

            # calculate instantaneous velocity.
            agent_inst_velo = math.sqrt(vel_x**2 + vel_y**2)

            agent_params = {
                "x": agent_x_t1,
                "y": agent_y_t1,
                "vel": agent_inst_velo
            }

            # Check overlapping safezone regions between the AV and the agent for a timestep
            # [True, False, False] -> [False ]
            overlap = check_safezone_overlap(av_params[i], agent_params, time_diff=(args.pred_len*(time_t2-time_t1)))
            agent_seq_overlap.append(overlap) # For a single timestep in the sequence store the overlap result between the AV and the agent.

            # If there was overlap between the AV and the agent at least once in the sequence, keep. Otherwise disregard agent.
            if agent_seq_overlap[-1] == False:
                final_agent_ids.remove(agent)
                
            final_count = len(final_agent_ids)


        print("\nRESULTS: Filtering agents based on intersection safezones between AV and AGENT: \n")
        print("Data filepath: {}".format(seq_path))
        print("# of agents dropped: {}".format(initial_count - final_count))
        print("# of agents total: {}".format(initial_count))
        print("(dropped #)/(total): {}".format( (initial_count - final_count) / initial_count))

        return 

        ## NOTE: Social features calculation was omitted for the Design Critique phase of the Capstone deliverable. 
        # for agent in initial_agent_ids:
        #     # Setup pd datastructure to set agent as the OBJECT_TYPE = AGENT

        #     # Indexing a DataFrame returns a reference to the initial DataFrame. Thus, changing the subset will change the initial DataFrame. Thus, you'd want to use the copy if you want to make sure the initial DataFrame shouldn't change.
        #     df_copy = df.copy() 

        #     # Change the OBJECT_TYPE for all IDs matching the agent to the AGENT type, all others will be considered OTHER
        #     ### BIG ISSUE HERE, need to make sure we have enough sequence length while defining a TRACK_ID as AGENT 
        #     ### sth like number of track_id associated values == (obs_len)


        #     df_copy["OBJECT_TYPE"].loc[(df_copy["TRACK_ID"] == agent)] = "AGENT"
        #     df_copy["OBJECT_TYPE"].loc[(df_copy["TRACK_ID"] != agent)] = "OTHER"

        #     # Compute social features

        #     # here we need to calculate the agent track first
        #     agent_track = df_copy[df_copy["OBJECT_TYPE"] == "AGENT"].values

        #     # we also need a data structure we pass, so that we don't recalculate the distances here to the compute_social_features
        #     # for now, lets say it is a dictionary with keys being the track_ids of two actors, use string keys here because tuple keys are slow in search
        #     # mydict[ str(1)+" "+str(2) ] = 'some_float' here the value is the distance, hence float

        #     social_features = social_features_utils_instance.compute_social_features(
        #         df_copy, agent_track, args.obs_len, args.obs_len + args.pred_len,
        #         RAW_DATA_FORMAT)

        #     # Store social features for that agent
        #     agents_social_features[agent] = social_features

        # print("Data fp: {} \n Computed social feature: {}\n".format(seq_path, agents_social_features))

    else:

        # Get social and map features for the agent
        agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

        # Social features are computed using only the observed trajectory
        social_features = social_features_utils_instance.compute_social_features(
            df, agent_track, args.obs_len, args.obs_len + args.pred_len,
            RAW_DATA_FORMAT)

    # # agent_track will be used to compute n-t distances for future trajectory,
    # # using centerlines obtained from observed trajectory
    # map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
    #     agent_track,
    #     args.obs_len,
    #     args.obs_len + args.pred_len,
    #     RAW_DATA_FORMAT,
    #     args.mode,
    # )

    # # Combine social and map features

    # # If track is of OBS_LEN (i.e., if it's in test mode), use agent_track of full SEQ_LEN,
    # # But keep (OBS_LEN+1) to (SEQ_LEN) indexes having None values
    # if agent_track.shape[0] == args.obs_len:
    #     agent_track_seq = np.full(
    #         (args.obs_len + args.pred_len, agent_track.shape[1]), None)
    #     agent_track_seq[:args.obs_len] = agent_track
    #     merged_features = np.concatenate(
    #         (agent_track_seq, social_features, map_features), axis=1)
    # else:
    #     merged_features = np.concatenate(
    #         (agent_track, social_features, map_features), axis=1)

    # return merged_features, map_feature_helpers
    


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

    if args.social_test: # NOTE: For testing purposes skip the merge_saved_features step.
        # Example command line input to test social feature changes: python compute_features.py --data_dir /Users/mithunjothiravi/Repos/Capstone_Social_LSTM/data --mode test --social_test True
        # NOTE: Once map code reflects the changes to the social computations, integrate it with the map features. 
        #   Look at compute_features func above to see what needs to get done.
        # how can we share the social feature dictionary amongs all? maybe this is just taken care of automatically. Sth to think abt while debugging, might cause issues
        Parallel(n_jobs=-2)(delayed(load_seq_save_features)(
            i,
            sequences,
            temp_save_dir,
            map_features_utils_instance,
            social_features_utils_instance,
        ) for i in range(0, num_sequences, args.batch_size))
    else: # Perform regular script behaviour
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
