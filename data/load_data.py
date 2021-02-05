'''
Great documentation on Lyft untils (L5Kit)
    - https://lyft.github.io/l5kit/data_format.html#scenes
'''

import numpy as np
import pandas as pd 
import time

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

import os
import sys

CUR_DIR = os.getcwd() 
PROJ_DIR = os.path.dirname(CUR_DIR)
LYFT_GENERATED_DATA_DIR = PROJ_DIR + "/lyft_generated_data"

TIME_CORRECTION_FACTOR = 1000000000 # ns -> s, Lyft data uses GPS time

# Column names for .csv file export
COLUMN_NAMES = ["TIMESTAMP",
                "TRACK_ID",
                "OBJECT_TYPE",
                "X",
                "Y",
                "CITY_NAME", 
                ]

def load_lyft_data():
    
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = CUR_DIR
    # get config
    cfg = load_config_data("./visualisation_config.yaml")

    dm = LocalDataManager()
    dataset_path = dm.require(cfg["val_data_loader"]["key"])
    zarr_dataset = ChunkedDataset(dataset_path).open()

    return zarr_dataset

def write_csv(data, file_count, file_timestamp):

    # Add column names to the data file
    np_data = np.array(data)
    df = pd.DataFrame(data = np_data, columns = COLUMN_NAMES)

    # Generate save directory for the files
    save_dir = LYFT_GENERATED_DATA_DIR + "/{}".format(file_timestamp)
    filename = save_dir + "/{}_lyft_{}.csv".format(file_timestamp, file_count)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    df.to_csv(filename, sep='\t', encoding='utf-8')

    print("\nSuccessfully converted Lyft scene to: {}".format(filename))

    return


def main():

    data = load_lyft_data()
    scene_count = 0

    for scene in data.scenes: # Each Lyft scene is 25 seconds long

        fp_time = int(time.time()) # Timestamp will be used to store all the Lyft data relevant to this scene to a common directory (see write_csv function)

        # Contents for .csv file
        content = list()
        
        # Fetch relevant frames for this scene
        frame_start_ind, frame_end_ind = scene['frame_index_interval'][0], scene['frame_index_interval'][-1]

        # Loop through each frame, and fetch timestamp plus agent information
        for frame_ind in range(frame_start_ind, frame_end_ind):
            timestamp = data.frames[frame_ind]['timestamp']/TIME_CORRECTION_FACTOR
            agent_start_ind, agent_end_ind = data.frames[frame_ind]['agent_index_interval'][0], data.frames[frame_ind]['agent_index_interval'][-1]

            for agent_ind in range(agent_start_ind, agent_end_ind):
                # Content for a row in .csv file
                row_content = list()

                track_id = data.agents[agent_ind]['track_id']

                object_type_arr = data.agents[agent_ind]['label_probabilities'] # label_probabilities are listed like this: [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.], which refers to the PERCEPTION_LABELS defined in l5kit
                object_type_ind = int(np.argmax(object_type_arr)) # Get the index of the prob. array with the heighest probability of class detection
                object_type = PERCEPTION_LABELS[object_type_ind].split("_")[-1] # All labels come in the form: PERCEPTION_LABEL_CAR, only keep the last part of the string

                x, y = data.agents[agent_ind]['centroid'][0], data.agents[agent_ind]['centroid'][-1]

                city_name = "None"

                row_content = [timestamp, track_id, object_type, x, y, city_name] # Same structure as Argoverse .csv files
                content.append(row_content)
                
        # Write data to .csv for scene
        scene_count += 1
        write_csv(content, scene_count, fp_time)

    return

main()







