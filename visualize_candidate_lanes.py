"""
Visualize the map argoverse data.

Args:
    arg1: "--mode" should be "train", "test", "val" or "compute_all" (capstone one)
    arg2: "--single_figure" optional flag which will combine all the centerlines onto one figure for the scene.
    others: see parse_arguments()

Returns:
    Nothing

"""
import os
import pickle as pkl
import pandas as pd
import sys
import argparse
import time
import matplotlib.pyplot as plt
from utils.baseline_config import RAW_DATA_FORMAT

from argoverse.utils.mpl_plotting_utils import visualize_centerline

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obs_len",
        default=20,
        type=int,
        help="Directory where the sequences (csv files) are saved",
    )
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
        help="Directory where the computed features are saved",
    )
    parser.add_argument("--mode",
                        required=True,
                        type=str,
                        help="train/val/test/compute_all/lanes_only")
    parser.add_argument(
        "--sequence_num",
        default=-1,
        type=int,
        help="Specify a specific sequence to visualize.",
    )
    parser.add_argument(
        "--batch_start",
        default=0,
        type=int,
        help="Specify the starting row of features to visualize.",
    )
    parser.add_argument(
        "--batch_end",
        default=-1,
        type=int,
        help="Specify the last row to visualize, -1 to visualize till end.",
    )
    parser.add_argument(
        "--single_figure",
        default=False,
        action="store_true",
        help="Plot all candidates for a scenein one figure.",
    )
    return parser.parse_args()

def plot_agent_track(track_id, seq_agents_df, colour, line_width, alpha):
    """Plot the track for a given agent"""

    agent_track = seq_agents_df[seq_agents_df["TRACK_ID"] == track_id].values

    agent_xy = agent_track[:, [RAW_DATA_FORMAT["X"], RAW_DATA_FORMAT["Y"]
                                   ]].astype("float")

    plt.plot(
        agent_xy[:, 0],
        agent_xy[:, 1],
        "-",
        color=colour,
        alpha=alpha,
        linewidth=line_width,
        zorder=15,
    )

    return agent_xy

def plot_scene(args, seq, seq_agents_df, map_feature_row):
    """Function that plots the centerlines and all agent trajectories."""

    candidate_centerlines = map_feature_row["CANDIDATE_CENTERLINES"]

    # Visualize the map centerlines
    for centerline_coords in candidate_centerlines:
        visualize_centerline(centerline_coords)

    # Plot the all the other agents
    for other_id in seq_agents_df["TRACK_ID"].unique():
        plot_agent_track(track_id=other_id, seq_agents_df=seq_agents_df, colour="#289BD4", line_width=3, alpha=0.5)

    # Plot the trajectory of the real "AGENT"
    agent_track_id = seq_agents_df[seq_agents_df["OBJECT_TYPE"] == "AGENT"]["TRACK_ID"].unique()[0]
    agent_xy = plot_agent_track(track_id=agent_track_id, seq_agents_df=seq_agents_df, colour="#D4CF53", line_width=3, alpha=1)

    # Plot the trajectory of this agent
    track_id = map_feature_row["TRACK_ID"]
    agent_track = seq_agents_df[seq_agents_df["TRACK_ID"] == track_id].values
    plot_agent_track(track_id=track_id, seq_agents_df=seq_agents_df, colour="#d33e4c", line_width=3, alpha=1)


    return agent_xy

def visualize_map_features_row_separate(args, seq, seq_agents_df, map_feature_row):
    """Visualize a row of map features and the scene."""

    print("Visualizing sequence {}, agent {}, with {} candidates.".format(map_feature_row["SEQUENCE"], map_feature_row["TRACK_ID"], len(map_feature_row["CANDIDATE_CENTERLINES"])))   
    
    candidate_centerlines = map_feature_row["CANDIDATE_CENTERLINES"]

    plt.figure(figsize=(8, 7))

    agent_xy = plot_scene(args, seq, seq_agents_df, map_feature_row) 

    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    plt.axis("off")
    plt.title(f"Num candidates = {len(candidate_centerlines)}, Track Len = {len(agent_xy)}")
    plt.savefig(f"{args.feature_dir}/{seq}_{map_feature_row['TRACK_ID']}.png")

def visualize_map_features_row_single_figure(args, seq_id, seq_agents_df, seq_features_df):
    """Visualize all rows of map features of map features and the scene."""

    print("Visualizing sequence {}.".format(seq_id))   
    
    plt.figure(figsize=(8, 7))

    for index, row in seq_features_df.iterrows():
        plot_scene(args, seq_id, seq_agents_df, row) 
    
    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    plt.axis("off")
    plt.title(f"Sequence {seq_id}")
    plt.savefig(f"{args.feature_dir}/{seq_id}.png")


def visualize_computed_centerlines(args, sequences):
    """Load the computed features and scene and then visualize."""

    # Load the computed features
    feature_data = pkl.load( open( "{}/forecasting_features_{}.pkl".format(args.feature_dir, args.mode), "rb" ) )
    all_features_dataframe = pd.DataFrame(feature_data)

    # If batch limits are set, slice dataframe from dataframe
    if args.batch_start != 0 or args.batch_end != -1:
        all_features_dataframe = all_features_dataframe.loc[args.batch_start:args.batch_end]

    # Get a list of sequences
    sequence_list = sequences
    if args.sequence_num != -1:
        sequence_list = ["{}.csv".format(args.sequence_num)]

    # Loop over the sequences, computing each sequence, visualizing the rows in that sequence
    for seq in sequence_list:

        # Load the sequence file
        if not seq.endswith(".csv"):
            continue
        file_path = f"{args.data_dir}/{seq}"
        seq_agents_df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})

        # Loop over the features rows in this sequence
        seq_id = int(seq.split(".")[0])
        seq_features_df = all_features_dataframe[all_features_dataframe["SEQUENCE"] == seq_id]

        if not args.single_figure:
            # Loop over all the feature rows, visualizing the centerline
            for index, row in seq_features_df.iterrows():
                visualize_map_features_row_separate(args, seq_id, seq_agents_df, row)
        else:
            # Visualize the all agents onto a single figure
            visualize_map_features_row_single_figure(args, seq_id, seq_agents_df, seq_features_df)



if __name__ == "__main__":
    """Load sequences and save the computed features."""
    args = parse_arguments()

    start = time.time()

    sequences = os.listdir(args.data_dir)

    visualize_computed_centerlines(args, sequences)

    print(
        f"Vizualization for {args.mode} set completed in {(time.time()-start)/60.0} mins."
    )
    print(
        f"Visualized scene {args.sequence_num} from agent {args.batch_start} to {args.batch_end}."
    )
