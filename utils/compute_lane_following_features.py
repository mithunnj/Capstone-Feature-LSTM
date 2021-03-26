import pandas as pd
import numpy as np
from typing import Tuple, List

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import get_normal_and_tangential_distance_point
from utils.baseline_utils import get_xy_from_nt

def get_agent_track_from_df(scene_df, track_id, raw_data_format):
    """Get the agent track in (x, y) from a dataframe of the scene."""

    agent_track = scene_df[scene_df["TRACK_ID"] == track_id].values
    agent_xy = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]]].astype(np.float16)

    return agent_xy

def compute_n_points_for_point_by_lane(centerline, point, n, spacing, speed_distance):
    """Compute n closest points for this lane."""

    # Find the initial norm/tang distance (corresponding to the closest point on the lane)
    _, t_0 = get_normal_and_tangential_distance_point(x=point[0], y=point[1], centerline=centerline)

    # Taking spacing to be the max between spacing and speed distance
    spacing = max(spacing, speed_distance)

    # Loop over n points, incrementing tan then converting norm and tan back to x, y
    t_current = t_0
    n_current = 0
    points = list()
    for i in range(n):

        # Add the next point
        next_point = get_xy_from_nt(n=n_current, t=t_current, centerline=centerline)
        points.append(next_point)

        # If this is not the first point, increment with exponentially increasing "look-ahead" points
        if i < 3:
            t_current += speed_distance
        else:
            t_current += spacing * (2 ** (i - 2)) # Will add 2x, 4x, 8x, 16x, etc.

    return points

def compute_n_points_for_lanes(
        agent_xy: np.ndarray,
        centerlines: np.ndarray,
        num_points: int,
        spacing: int,
        speed: np.ndarray
    ) -> np.ndarray:
    """Find closest n points for each candidate centerline."""

    # Loop over centerlines and store closest points in a numpy array
    computed_points = np.zeros((len(centerlines), len(agent_xy), num_points, 2), dtype=np.float16)
    for lane in centerlines:

        # Loop over points in the agent trajectory
        for idx, point in enumerate(agent_xy):

            # Adjust the spacing to match the speed (speed * time)
            speed_distance = speed[idx] * 0.01 # Assume 0.01s or 1Hz

            # Compute n closest points from the lane to this point and store
            closest_points = compute_n_points_for_point_by_lane(centerline=lane[1], point=point,\
                n=num_points, spacing=spacing, speed_distance=speed_distance)
            computed_points[lane[0], idx, :] = np.array(closest_points)

    return computed_points

def compute_lane_following_features(
    scene_df: pd.DataFrame,
    agent_list: list,
    precomputed_lanes: pd.DataFrame,
    raw_data_format: list,
    map_inst: ArgoverseMap,
    seq_id: int,
    obs_len: int,
    precomputed_physics: pd.DataFrame,
    # Configurable constants
    num_points: int = 7, # Number of points to return
    spacing: int = 2 # Distance between points (in metres)
) -> Tuple[List, List]:
    """Compute lane following features (n closest points)."""

    track_id = agent_list[0]

    # Get agent track and limit to obs_len
    agent_xy = get_agent_track_from_df(scene_df, track_id, raw_data_format)
    agent_xy = agent_xy[0:obs_len, :]

    # Get candidate centerlines
    precomputed_scene  = precomputed_lanes[precomputed_lanes["SEQUENCE"] == seq_id]
    centerlines = precomputed_scene["CENTERLINES"].values[0]

    # Get speed from precomputed velocities
    precomputed_agent_physics = precomputed_physics[(precomputed_physics["SEQUENCE"] == seq_id) \
        & (precomputed_physics["TRACK_ID"] == track_id)]
    vel_x = precomputed_agent_physics["VEL_X"].values[0]
    vel_y = precomputed_agent_physics["VEL_Y"].values[0]
    vel_xy = np.column_stack((vel_x, vel_y))
    speed = np.linalg.norm(vel_xy[0:20, :], axis=1)

    # Compute n closest points
    n_computed_points = compute_n_points_for_lanes(
        agent_xy=agent_xy,
        centerlines=centerlines,
        num_points=num_points,
        spacing=spacing,
        speed=speed
    )

    # Convert to feature format
    column_names = [ "SEQUENCE", "TRACK_ID" ]
    items = [ seq_id, track_id ]
    for i in range(num_points):

        # Add column names for this point (point in the future)
        column_names.append(f"POINT{i}_X")
        column_names.append(f"POINT{i}_Y")

        # Construct centerline list
        lanes_x = list() # list follows format [ (centerline_id, [x1, x2, ...]) ]
        lanes_y = list()

        for lane in centerlines:
            lanes_x.append((lane[0], n_computed_points[lane[0], :, i, 0])) # append a tuple of format ()
            lanes_y.append((lane[0], n_computed_points[lane[0], :, i, 1]))
        
        items.append(lanes_x)
        items.append(lanes_y)

    return column_names, [ items ]