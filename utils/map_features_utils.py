"""This module is used for computing map features for motion forecasting baselines."""

from typing import Any, Dict, List, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import cascaded_union

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    get_nt_distance,
    remove_overlapping_lane_seq,
)
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from argoverse.utils.line_projection import project_to_line_seq
from utils.baseline_config import (
    _MANHATTAN_THRESHOLD,
    _DFS_THRESHOLD_FRONT_SCALE,
    _DFS_THRESHOLD_BACK_SCALE,
    _MAX_SEARCH_RADIUS_CENTERLINES,
    _MAX_CENTERLINE_CANDIDATES_TEST,
)


class MapFeaturesUtils:
    """Utils for computation of map-based features."""
    def __init__(self):
        """Initialize class."""
        self._MANHATTAN_THRESHOLD = _MANHATTAN_THRESHOLD
        self._DFS_THRESHOLD_FRONT_SCALE = _DFS_THRESHOLD_FRONT_SCALE
        self._DFS_THRESHOLD_BACK_SCALE = _DFS_THRESHOLD_BACK_SCALE
        self._MAX_SEARCH_RADIUS_CENTERLINES = _MAX_SEARCH_RADIUS_CENTERLINES
        self._MAX_CENTERLINE_CANDIDATES_TEST = _MAX_CENTERLINE_CANDIDATES_TEST

    def get_point_in_polygon_score(self, lane_seq: List[int],
                                   xy_seq: np.ndarray, city_name: str,
                                   avm: ArgoverseMap) -> int:
        """Get the number of coordinates that lie insde the lane seq polygon.

        Args:
            lane_seq: Sequence of lane ids
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
        Returns:
            point_in_polygon_score: Number of coordinates in the trajectory that lie within the lane sequence

        """
        lane_seq_polygon = cascaded_union([
            Polygon(avm.get_lane_segment_polygon(lane, city_name)).buffer(0)
            for lane in lane_seq
        ])
        point_in_polygon_score = 0
        for xy in xy_seq:
            point_in_polygon_score += lane_seq_polygon.contains(Point(xy))
        return point_in_polygon_score

    def sort_lanes_based_on_point_in_polygon_score(
            self,
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
    ) -> List[List[int]]:
        """Filter lane_seqs based on the number of coordinates inside the bounding polygon of lanes.

        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
        Returns:
            sorted_lane_seqs: Sequences of lane sequences sorted based on the point_in_polygon score

        """
        point_in_polygon_scores = []
        for lane_seq in lane_seqs:
            point_in_polygon_scores.append(
                self.get_point_in_polygon_score(lane_seq, xy_seq, city_name,
                                                avm))
        randomized_tiebreaker = np.random.random(len(point_in_polygon_scores))
        sorted_point_in_polygon_scores_idx = np.lexsort(
            (randomized_tiebreaker, np.array(point_in_polygon_scores)))[::-1]
        sorted_lane_seqs = [
            lane_seqs[i] for i in sorted_point_in_polygon_scores_idx
        ]
        sorted_scores = [
            point_in_polygon_scores[i]
            for i in sorted_point_in_polygon_scores_idx
        ]
        return sorted_lane_seqs, sorted_scores

    def get_heuristic_centerlines_for_test_set(
            self,
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            max_candidates: int,
            scores: List[int],
    ) -> List[np.ndarray]:
        """Sort based on distance along centerline and return the centerlines.
        
        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
            max_candidates: Maximum number of centerlines to return
        Return:
            sorted_candidate_centerlines: Centerlines in the order of their score 

        """
        aligned_centerlines = []
        aligned_lane_seq = []
        diverse_centerlines = []
        diverse_lane_seq = []
        diverse_scores = []
        num_candidates = 0

        # Get first half as aligned centerlines
        aligned_cl_count = 0
        for i in range(len(lane_seqs)):
            lane_seq = lane_seqs[i]
            score = scores[i]
            diverse = True
            centerline = avm.get_cl_from_lane_seq([lane_seq], city_name)[0]
            if aligned_cl_count < int(max_candidates / 2):
                start_dist = LineString(centerline).project(Point(xy_seq[0]))
                end_dist = LineString(centerline).project(Point(xy_seq[-1]))
                if end_dist > start_dist:
                    aligned_cl_count += 1
                    aligned_centerlines.append(centerline)
                    aligned_lane_seq.append(lane_seq)
                    diverse = False
            if diverse:
                diverse_centerlines.append(centerline)
                diverse_lane_seq.append(lane_seq)
                diverse_scores.append(score)

        num_diverse_centerlines = min(len(diverse_centerlines),
                                      max_candidates - aligned_cl_count)
        test_centerlines = aligned_centerlines
        test_lane_seq = aligned_lane_seq
        if num_diverse_centerlines > 0:
            probabilities = ([
                float(score + 1) / (sum(diverse_scores) + len(diverse_scores))
                for score in diverse_scores
            ] if sum(diverse_scores) > 0 else [1.0 / len(diverse_scores)] *
                             len(diverse_scores))
            diverse_centerlines_idx = np.random.choice(
                range(len(probabilities)),
                num_diverse_centerlines,
                replace=False,
                p=probabilities,
            )
            diverse_centerlines = [
                diverse_centerlines[i] for i in diverse_centerlines_idx
            ]
            diverse_lane_seq = [
                diverse_lane_seq[i] for i in diverse_centerlines_idx
            ]
            test_centerlines += diverse_centerlines
            test_lane_seq += diverse_lane_seq

        return test_centerlines, test_lane_seq

    def get_candidate_centerlines_for_trajectory(
            self,
            xy: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            viz: bool = False,
            max_search_radius: float = 50.0,
            seq_len: int = 50,
            max_candidates: int = 10,
            mode: str = "test",
    ) -> List[np.ndarray]:
        """Get centerline candidates upto a threshold.

        Algorithm:
        1. Take the lanes in the bubble of last observed coordinate
        2. Extend before and after considering all possible candidates
        3. Get centerlines based on point in polygon score.

        Args:
            xy: Trajectory coordinates, 
            city_name: City name, 
            avm: Argoverse map_api instance, 
            viz: Visualize candidate centerlines, 
            max_search_radius: Max search radius for finding nearby lanes in meters,
            seq_len: Sequence length, 
            max_candidates: Maximum number of centerlines to return, 
            mode: train/val/test mode

        Returns:
            candidate_centerlines: List of candidate centerlines

        """

        candidate_lane_segments = None
        candidate_centerlines = None


        # Get all lane candidates within a bubble
        curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
            xy[-1, 0], xy[-1, 1], city_name, self._MANHATTAN_THRESHOLD)

        # Keep expanding the bubble until at least 1 lane is found
        while (len(curr_lane_candidates) < 1
               and self._MANHATTAN_THRESHOLD < max_search_radius):
            self._MANHATTAN_THRESHOLD *= 2
            curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
                xy[-1, 0], xy[-1, 1], city_name, self._MANHATTAN_THRESHOLD)

        assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

        # Set dfs threshold
        traj_len = xy.shape[0]

        # Assuming a speed of 50 mps, set threshold for traversing in the front and back
        dfs_threshold_front = (self._DFS_THRESHOLD_FRONT_SCALE *
                               (seq_len + 1 - traj_len) / 10)
        dfs_threshold_back = self._DFS_THRESHOLD_BACK_SCALE * (traj_len +
                                                               1) / 10

        # DFS to get all successor and predecessor candidates
        obs_pred_lanes: List[Sequence[int]] = []
        for lane in curr_lane_candidates:
            candidates_future = avm.dfs(lane, city_name, 0,
                                        dfs_threshold_front)
            candidates_past = avm.dfs(lane, city_name, 0, dfs_threshold_back,
                                      True)
                
            # Merge past and future
            for past_lane_seq in candidates_past:
                for future_lane_seq in candidates_future:
                    assert (
                        past_lane_seq[-1] == future_lane_seq[0]
                    ), "Incorrect DFS for candidate lanes past and future"
                    obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

        # Removing overlapping lanes
        obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

        # Sort lanes based on point in polygon score
        obs_pred_lanes, scores = self.sort_lanes_based_on_point_in_polygon_score(
            obs_pred_lanes, xy, city_name, avm)

        # If the best centerline is not along the direction of travel, re-sort
        if mode == "test" or mode == "lanes_only":
            # Sort based on alignment with candidate lane
            candidate_centerlines, candidate_lane_segments = self.get_heuristic_centerlines_for_test_set(
                obs_pred_lanes, xy, city_name, avm, max_candidates, scores)
        else:
            # Pick oracle centerline
            candidate_centerlines = avm.get_cl_from_lane_seq(
                [obs_pred_lanes[0]], city_name)

        if viz:
            plt.figure(0, figsize=(8, 7))
            for centerline_coords in candidate_centerlines:
                visualize_centerline(centerline_coords)
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "-",
                color="#d33e4c",
                alpha=1,
                linewidth=3,
                zorder=15,
            )

            final_x = xy[-1, 0]
            final_y = xy[-1, 1]

            plt.plot(
                final_x,
                final_y,
                "o",
                color="#d33e4c",
                alpha=1,
                markersize=10,
                zorder=15,
            )
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title(f"Number of candidates = {len(candidate_centerlines)}")
            plt.show()

        if mode == "lanes_only":
            return candidate_centerlines, candidate_lane_segments

        return candidate_centerlines

    def compute_map_features(
            self,
            agent_track: np.ndarray,
            obs_len: int,
            seq_len: int,
            raw_data_format: Dict[str, int],
            mode: str,
            avm: ArgoverseMap
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute map based features for the given sequence.

        If the mode is test, oracle_nt_dist will be empty, candidate_nt_dist will be populated.
        If the mode is train/val, oracle_nt_dist will be populated, candidate_nt_dist will be empty.

        Args:
            agent_track : Data for the agent track
            obs_len : Length of observed trajectory
            seq_len : Length of the sequence
            raw_data_format : Format of the sequence
            mode: train/val/test mode
            
        Returns:
            oracle_nt_dist (numpy array): normal and tangential distances for oracle centerline
                map_feature_helpers (dict): Dictionary containing helpers for map features

        """
        obs_pred_lanes = []
        unique_segments_future = []
        unique_segments_past = []
        curr_lane_candidates = [] 


        # Get observed 2 secs of the agent
        agent_xy = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]
                                   ]].astype("float")
        agent_track_obs = agent_track[:obs_len]
        agent_xy_obs = agent_track_obs[:, [
            raw_data_format["X"], raw_data_format["Y"]
        ]].astype("float")

        # Get API for Argo Dataset map
        city_name = agent_track[0, raw_data_format["CITY_NAME"]]

        # Get candidate centerlines using observed trajectory
        if mode == "test":
            oracle_centerline = np.full((seq_len, 2), None)
            oracle_nt_dist = np.full((seq_len, 2), None)
            candidate_centerlines = self.get_candidate_centerlines_for_trajectory(
                agent_xy_obs,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                max_candidates=self._MAX_CENTERLINE_CANDIDATES_TEST,
            )

            # Get nt distance for the entire trajectory using candidate centerlines
            candidate_nt_distances = []
            for candidate_centerline in candidate_centerlines:
                candidate_nt_distance = np.full((seq_len, 2), None)
                candidate_nt_distance[:obs_len] = get_nt_distance(
                    agent_xy_obs, candidate_centerline)
                candidate_nt_distances.append(candidate_nt_distance)

        elif mode == "compute_all":
            # Get oracle centerline
            oracle_centerline = self.get_candidate_centerlines_for_trajectory(
                agent_xy,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                mode="train",
            )[0]
            # Get NT distance for oracle centerline
            oracle_nt_dist = get_nt_distance(agent_xy,
                                             oracle_centerline,
                                             viz=False)
            
            # Get candidate centerl = []ines
            candidate_centerlines, obs_pred_lanes, unique_segments_future, unique_segments_past, curr_lane_candidates = self.get_candidate_centerlines_for_trajectory(
                agent_xy_obs,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                max_candidates=self._MAX_CENTERLINE_CANDIDATES_TEST,
                mode=mode
            )

            # Get nt distance for the entire trajectory using candidate centerlines
            candidate_nt_distances = []
            for candidate_centerline in candidate_centerlines:
                candidate_nt_distance = np.full((seq_len, 2), None)
                candidate_nt_distance[:obs_len] = get_nt_distance(
                    agent_xy_obs, candidate_centerline)
                candidate_nt_distances.append(candidate_nt_distance)

        elif mode == "lanes_only":
            # Get oracle centerline
            oracle_centerline = self.get_candidate_centerlines_for_trajectory(
                agent_xy,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                mode="train",
            )[0]
            # Not computing oracle nt_distances
            oracle_nt_dist = np.full((seq_len, 2), None) 
            
            # Get candidate centerl = []ines
            candidate_centerlines, obs_pred_lanes, unique_segments_future, unique_segments_past, curr_lane_candidates = self.get_candidate_centerlines_for_trajectory(
                agent_xy_obs,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                max_candidates=self._MAX_CENTERLINE_CANDIDATES_TEST,
                mode=mode
            )

            # Not computing candidate nt_distances
            candidate_nt_distances = []

        else:
            oracle_centerline = self.get_candidate_centerlines_for_trajectory(
                agent_xy,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                mode=mode,
            )[0]
            candidate_centerlines = [np.full((seq_len, 2), None)]
            candidate_nt_distances = [np.full((seq_len, 2), None)]

            # Get NT distance for oracle centerline
            oracle_nt_dist = get_nt_distance(agent_xy,
                                             oracle_centerline,
                                             viz=False)

        map_feature_helpers = {
            "ORACLE_CENTERLINE": oracle_centerline,
            "CANDIDATE_CENTERLINES": candidate_centerlines,
            "CANDIDATE_NT_DISTANCES": candidate_nt_distances,
            "CANDIDATE_LANE_SEGMENTS": obs_pred_lanes,
            "LANE_SEGMENTS_IN_BUBBLE": curr_lane_candidates,
            "LANE_SEGMENTS_IN_FRONT": unique_segments_future,
            "LANE_SEGMENTS_IN_BACK": unique_segments_past 
        }

        return oracle_nt_dist, map_feature_helpers


    def compute_lane_candidates(
            self,
            seq_id: int,
            scene_df: pd.DataFrame,
            agent_list: list,
            obs_len: int,
            seq_len: int,
            raw_data_format: Dict[str, int],
            mode: str,
            multi_agent: bool,
            avm: ArgoverseMap
    ) -> Tuple[list, list]:
        """Compute candidate centerlines for a given sequence.

        Args:
            scene_df : Dataframe for the scene where each row is an agent and timestep
            agent_list: A list of track ids of agents that should be computed
            obs_len : Length of observed trajectory
            seq_len : Length of the sequence
            raw_data_format : Format of the sequence
            mode: train/val/test mode
            multi_agent: Whether or not this should be computed for multiple agents (must be False)
            
        Returns:
            colums (list of strings): column names for the return dataframe
            dataframe (pd dataframe): Dataframe containing the centerlines for each agent

        """

        assert avm != None, "Invalid argoverse map api instance passed to compute_lane_candidates."
        assert not multi_agent and len(agent_list) == 1, "Candidate centerlines not supported for multiple agents"

        # Get agent track
        track_id = agent_list[0]
        agent_track = scene_df[scene_df["TRACK_ID"] == track_id].values

        # Get observed 2 secs of the agent
        agent_xy = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]
                                   ]].astype("float")
        agent_track_obs = agent_track[:obs_len]
        agent_xy_obs = agent_track_obs[:, [
            raw_data_format["X"], raw_data_format["Y"]
        ]].astype("float")

        
        # Get API for Argo Dataset map
        city_name = agent_track[0, raw_data_format["CITY_NAME"]]

        # Find the oracle centerline for training
        oracle_centerline = self.get_candidate_centerlines_for_trajectory(
            agent_xy,
            city_name,
            avm,
            viz=False,
            max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
            seq_len=seq_len,
            mode="train",
        )[0]

        # Find the candidate centerlines for testing
        candidate_centerlines, candidate_lane_segments = self.get_candidate_centerlines_for_trajectory(
            agent_xy_obs,
            city_name,
            avm,
            viz=False,
            max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
            seq_len=seq_len,
            max_candidates=self._MAX_CENTERLINE_CANDIDATES_TEST,
            mode="lanes_only"
        )

        # Set return columns
        columns = ["SEQUENCE", "TRACK_ID", "CENTERLINES", "LANE_SEGMENTS", "ORACLE_CENTERLINE" ]

        # Simple check to see if the lane segments and centerlines are the same length
        assert len(candidate_centerlines) == len(candidate_lane_segments), "Lane candidates do not match lane segments."

        # Convert list of centerlines/segments to tuple format (centerline_id, data)
        centerline_tuples = []
        lane_segment_tuples = []
        for i in range(len(candidate_centerlines)):
            centerline_tuples.append((i, candidate_centerlines[i]))
            lane_segment_tuples.append((i, candidate_lane_segments[i]))

        # Construct the return row
        rows = [ [ seq_id, track_id, centerline_tuples, lane_segment_tuples, oracle_centerline ] ]

        return columns, rows
