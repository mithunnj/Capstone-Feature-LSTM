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
    get_normal_and_tangential_distance_point,
)
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from argoverse.utils.line_projection import project_to_line_seq
from utils.baseline_config import (
    _MANHATTAN_THRESHOLD,
    _DFS_THRESHOLD_FRONT_SCALE,
    _DFS_THRESHOLD_BACK_SCALE,
    _MAX_SEARCH_RADIUS_CENTERLINES,
    _MAX_CENTERLINE_CANDIDATES_TEST,
    NEARBY_DISTANCE_THRESHOLD, # this already exists so following the convention in social utils
    FRONT_OR_BACK_OFFSET_THRESHOLD,
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
        self.NEARBY_DISTANCE_THRESHOLD = NEARBY_DISTANCE_THRESHOLD


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

        # ADD A LOOP HERE TO GO OVER THE CANDIDATE CENTERLINES. CURRENTLY WE DO IT ONLY FOR ONE LINE.

        # here, do the lead and following agent assignments
        initial_agent_ids = df["TRACK_ID"].values # All the track_ids (unique actor identifier in .csv dataset)
        initial_agent_ids = list(dict.fromkeys(initial_agent_ids)) # Remove duplicates from IDs
        removed_agent_ids = initial_agent_ids
        # Remove the current agent we are calculating the features for
        removed_agent_ids.remove(track_id)


        # here need to loop thru candidate centerlines to return for every candidate centerline.
        # need to add normal dist. threshold bc the lead vehicle might be returned even tho it is far from the lane
        # since algorthm returns the closest currently
        # need to add threshold for distance to the agent.
        # need to convert to desired standard format that is (centerline, feature)
        leading_actor = []
        following_actor = []
        leading_actor_dist = []
        following_actor_dist = []
        #print(candidate_centerlines)
        c_line_idx = 0
        # for c_line in candidate_centerlines:
        #     leading_actor_idx, following_actor_idx, \
        #         leading_actor_dist_idx, following_actor_dist_idx = \
        #         self.get_lead_and_follow(df, obs_len, agent_track, removed_agent_ids, \
        #         c_line, raw_data_format)
        #     leading_actor.append(leading_actor_idx)
        #     following_actor.append(following_actor_idx)
        #     leading_actor_dist.append(leading_actor_dist_idx)
        #     following_actor_dist.append(following_actor_dist_idx)
        #     c_line_idx += 1

        map_feature_helpers = {
            "LEADING_VEHICLES": leading_actor,
            "FOLLOWING_VEHICLES": following_actor,
            "LEADING_DISTANCES": leading_actor_dist,
            "FOLLOWING_DISTANCES": following_actor_dist,
            "ORACLE_CENTERLINE": oracle_centerline,
            "CANDIDATE_CENTERLINES": candidate_centerlines,
            "CANDIDATE_NT_DISTANCES": candidate_nt_distances,
            "CANDIDATE_LANE_SEGMENTS": obs_pred_lanes,
            "LANE_SEGMENTS_IN_BUBBLE": curr_lane_candidates,
            "LANE_SEGMENTS_IN_FRONT": unique_segments_future,
            "LANE_SEGMENTS_IN_BACK": unique_segments_past 
        }

        return oracle_nt_dist, map_feature_helpers

    def compute_lead(
            self,
            seq_id: int,
            scene_df: pd.DataFrame,
            agent_list: list,
            obs_len: int,
            seq_len: int,
            raw_data_format: Dict[str, int],
            mode: str,
            multi_agent: bool,
            avm: ArgoverseMap,
            precomputed_physics: pd.DataFrame
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

        # here, do the lead and following agent assignments
        initial_agent_ids = scene_df["TRACK_ID"].values # All the track_ids (unique actor identifier in .csv dataset)
        initial_agent_ids = list(dict.fromkeys(initial_agent_ids)) # Remove duplicates from IDs
        removed_agent_ids = initial_agent_ids
        # Remove the current agent we are calculating the features for
        removed_agent_ids.remove(track_id)

        # here need to loop thru candidate centerlines to return for every candidate centerline.
        # need to add normal dist. threshold bc the lead vehicle might be returned even tho it is far from the lane
        # since algorthm returns the closest currently
        # need to add threshold for distance to the agent.
        # need to convert to desired standard format that is (centerline, feature)
        leading_actor = []
        following_actor = []
        leading_actor_dist = []
        following_actor_dist = []

        following_x = []
        following_y = []
        following_vel_x = []
        following_vel_y = []
        following_acc_x = []
        following_acc_y = []
        following_jerk_x = []
        following_jerk_y = []
        following_yaw = []
        following_yaw_rate = []

        leading_x = []
        leading_y = []
        leading_vel_x = []
        leading_vel_y = []
        leading_acc_x = []
        leading_acc_y = []
        leading_jerk_x = []
        leading_jerk_y = []
        leading_yaw = []
        leading_yaw_rate = []

        #print(candidate_centerlines)
        c_line_idx = 0
        for c_line in candidate_centerlines:
            leading_actor_idx, following_actor_idx, \
                leading_actor_dist_idx, following_actor_dist_idx, \
                following_x_idx, following_y_idx, following_vel_x_idx, following_vel_y_idx, following_acc_x_idx, \
                following_acc_y_idx, following_jerk_x_idx, following_jerk_y_idx, following_yaw_idx, following_yaw_rate_idx, \
                leading_x_idx, leading_y_idx, leading_vel_x_idx, leading_vel_y_idx, leading_acc_x_idx, \
                leading_acc_y_idx, leading_jerk_x_idx, leading_jerk_y_idx, leading_yaw_idx, leading_yaw_rate_idx = \
                self.get_lead_and_follow(seq_id, track_id, scene_df, obs_len, agent_track, removed_agent_ids, \
                c_line, raw_data_format, precomputed_physics)

            leading_actor.append((c_line_idx, leading_actor_idx))
            following_actor.append((c_line_idx, following_actor_idx))
            leading_actor_dist.append((c_line_idx, leading_actor_dist_idx))
            following_actor_dist.append((c_line_idx, following_actor_dist_idx))

            following_x.append((c_line_idx, following_x_idx))
            following_y.append((c_line_idx, following_y_idx))
            following_vel_x.append((c_line_idx, following_vel_x_idx))
            following_vel_y.append((c_line_idx, following_vel_y_idx))
            following_acc_x.append((c_line_idx, following_acc_x_idx))
            following_acc_y.append((c_line_idx, following_acc_y_idx))
            following_jerk_x.append((c_line_idx, following_jerk_x_idx))
            following_jerk_y.append((c_line_idx, following_jerk_y_idx))
            following_yaw.append((c_line_idx, following_yaw_idx))
            following_yaw_rate.append((c_line_idx, following_yaw_rate_idx))

            leading_x.append((c_line_idx, leading_x_idx))
            leading_y.append((c_line_idx, leading_y_idx))
            leading_vel_x.append((c_line_idx, leading_vel_x_idx))
            leading_vel_y.append((c_line_idx, leading_vel_y_idx))
            leading_acc_x.append((c_line_idx, leading_acc_x_idx))
            leading_acc_y.append((c_line_idx, leading_acc_y_idx))
            leading_jerk_x.append((c_line_idx, leading_jerk_x_idx))
            leading_jerk_y.append((c_line_idx, leading_jerk_y_idx))
            leading_yaw.append((c_line_idx, leading_yaw_idx))
            leading_yaw_rate.append((c_line_idx, leading_yaw_rate_idx))

            c_line_idx += 1

        # Convert list of centerlines/segments to tuple format (centerline_id, data)
        centerline_tuples = []
        for i in range(len(candidate_centerlines)):
            centerline_tuples.append((i, candidate_centerlines[i]))

        # Set return columns
        columns = ["SEQUENCE", "TRACK_ID", "CENTERLINES", "LEAD_VEHICLES", "FOLLOWING_VEHICLES", "LEAD_DISTANCES", "FOLLOWING_DISTANCES", \
            "FOLLOWING_X", "FOLLOWING_Y", "FOLLOWING_VEL_X", "FOLLOWING_VEL_Y", "FOLLOWING_ACC_X", "FOLLOWING_ACC_Y", "FOLLOWING_JERK_X", "FOLLOWING_JERK_Y", "FOLLOWING_YAW", "FOLLOWING_YAW_RATE", \
            "LEADING_X", "LEADING_Y", "LEADING_VEL_X", "LEADING_VEL_Y", "LEADING_ACC_X", "LEADING_ACC_Y", "LEADING_JERK_X", "LEADING_JERK_Y", "LEADING_YAW", "LEADING_YAW_RATE"]

        # Construct the return row
        rows = [ [ seq_id, track_id, centerline_tuples, leading_actor, following_actor, leading_actor_dist, following_actor_dist, \
                following_x, following_y, following_vel_x, following_vel_y, following_acc_x, \
                following_acc_y, following_jerk_x, following_jerk_y, following_yaw, following_yaw_rate, \
                leading_x, leading_y, leading_vel_x, leading_vel_y, leading_acc_x, \
                leading_acc_y, leading_jerk_x, leading_jerk_y, leading_yaw, leading_yaw_rate ] ]
        # print(rows)

        return columns, rows

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

        # Simple check to see if the lane segments and centerlines are the same length
        assert len(candidate_centerlines) == len(candidate_lane_segments), "Lane candidates do not match lane segments."

        # Convert list of centerlines/segments to tuple format (centerline_id, data)
        centerline_tuples = []
        lane_segment_tuples = []
        for i in range(len(candidate_centerlines)):
            centerline_tuples.append((i, candidate_centerlines[i]))
            lane_segment_tuples.append((i, candidate_lane_segments[i]))

        # Set return columns
        columns = ["SEQUENCE", "TRACK_ID", "CENTERLINES", "LANE_SEGMENTS", "ORACLE_CENTERLINE" ]

        # Construct the return row
        rows = [ [ seq_id, track_id, centerline_tuples, lane_segment_tuples, oracle_centerline ] ]

        return columns, rows


    def get_lead_and_follow(
            self,
            seq_id,
            track_id,
            df,
            obs_len,
            agent_track,
            removed_agent_ids,
            c_line,
            raw_data_format,
            physics_df,
    ):
        # need to add normal dist. threshold bc the lead vehicle might be returned even tho it is far from the lane
        # since algorthm returns the closest currently
        # need to add threshold for distance to the agent. --> check instant_distance values.
        # need to convert to desired standard format that is (centerline, feature)
        """Given a list of centerline coordinates, agent ids in a scene, and the agent tracks,
        finds the lead and following vehicles associated with the centerlines, along with the 
        distances between agents.

        Args:
            
        Returns:

        """
        leading_actor = list()
        following_actor = list()
        leading_actor_dist = list()
        following_actor_dist = list()
        actor_info_front = dict()
        actor_info_back = dict()

        following_x = list()
        following_y = list()
        following_vel_x = list()
        following_vel_y = list()
        following_acc_x = list()
        following_acc_y = list()
        following_jerk_x = list()
        following_jerk_y = list()
        following_yaw = list()
        following_yaw_rate = list()

        leading_x = list()
        leading_y = list()
        leading_vel_x = list()
        leading_vel_y = list()
        leading_acc_x = list()
        leading_acc_y = list()
        leading_jerk_x = list()
        leading_jerk_y = list()
        leading_yaw = list()
        leading_yaw_rate = list()

        for i in range(obs_len):
            lead_score = 0
            lead_id = 0
            following_score = 0
            following_id = 0

            for actor_id in removed_agent_ids:
                actor_track = df[df["TRACK_ID"] == actor_id].values

                if len(actor_track) <= i:
                    continue

                actor_x, actor_y = (
                    actor_track[i, raw_data_format["X"]],
                    actor_track[i, raw_data_format["Y"]],
                )

                agent_x, agent_y = (
                    agent_track[i, raw_data_format["X"]],
                    agent_track[i, raw_data_format["Y"]],
                )

                instant_distance = np.sqrt((agent_x - actor_x)**2 + (agent_y - actor_y)**2)

                # remove anything further than 5 meters here.
                if (instant_distance >= self.NEARBY_DISTANCE_THRESHOLD):
                    continue

                is_front_or_back = self.get_is_front_or_back(
                    agent_track[:2, :] if i == 0 else agent_track[:i + 1, :],
                    actor_x,
                    actor_y,
                    raw_data_format,
                )
                
                # here implementing only for the first centerline, we can change this later accordingly
                tang_dist, normal_dist = get_normal_and_tangential_distance_point(
                    actor_x,
                    actor_y,
                    c_line,
                )
                # print("SHORTEST:" , shortest_distance_to_candidate)
                if is_front_or_back == "front":
                    actor_info_front[actor_id] = abs(normal_dist)

                elif is_front_or_back == "back":
                    actor_info_back[actor_id] = abs(normal_dist)

            # now append to the list of lead and following agents
            # need to handle the case where there is no items in these dictionaries
            sort_distances_back = sorted(actor_info_back.items(), key=lambda x: x[1])
            if sort_distances_back:
                # print("REACHED back")
                # print(sort_distances_back[0][0])
                following_actor.append(sort_distances_back[0][0])

                actor_track = df[df["TRACK_ID"] == sort_distances_back[0][0]].values

                # this shows the physics feature access works
                # row = df[df["TRACK_ID"] == sort_distances_back[0][0]]
                # print(row['X'])
                actor_x, actor_y = (
                    actor_track[i, raw_data_format["X"]],
                    actor_track[i, raw_data_format["Y"]],
                )
                agent_x, agent_y = (
                    agent_track[i, raw_data_format["X"]],
                    agent_track[i, raw_data_format["Y"]],
                )
                instant_distance = np.sqrt((agent_x - actor_x)**2 + (agent_y - actor_y)**2)
                # here you can compare instant distance with a threshold to decide
                following_actor_dist.append(instant_distance)

                # HERE APPEND PHYSICS FEATURES
                # print("FOLLOWING")
                following_physics = physics_df.loc[(physics_df['SEQUENCE'] == seq_id) & (physics_df['TRACK_ID'] == sort_distances_back[0][0])]
                following_x.append(following_physics['X'].values[0][i])
                following_y.append(following_physics['Y'].values[0][i])
                following_vel_x.append(following_physics['VEL_X'].values[0][i])
                following_vel_y.append(following_physics['VEL_Y'].values[0][i])
                following_acc_x.append(following_physics['ACC_X'].values[0][i])
                following_acc_y.append(following_physics['ACC_Y'].values[0][i])
                following_jerk_x.append(following_physics['JERK_X'].values[0][i])
                following_jerk_y.append(following_physics['JERK_Y'].values[0][i])
                following_yaw.append(following_physics['YAW'].values[0][i])
                following_yaw_rate.append(following_physics['YAW_RATE'].values[0][i])
            else:
                following_actor.append(-1)
                following_actor_dist.append(-1)
                following_x.append(0)
                following_y.append(0)
                following_vel_x.append(0)
                following_vel_y.append(0)
                following_acc_x.append(0)
                following_acc_y.append(0)
                following_jerk_x.append(0)
                following_jerk_y.append(0)
                following_yaw.append(0)
                following_yaw_rate.append(0)


    #column_headings = ["SEQUENCE", "TRACK_ID", "TIMESTAMP", "X", "Y", "VEL_X", "VEL_Y", "ACC_X", "ACC_Y", "JERK_X", "JERK_Y", "YAW", "YAW_RATE"] 


            sort_distances_front = sorted(actor_info_front.items(), key=lambda x: x[1])
            if sort_distances_front:
                # print("REACHED front")
                # print(sort_distances_front[0][0])
                leading_actor.append(sort_distances_front[0][0])
                
                actor_track = df[df["TRACK_ID"] == sort_distances_front[0][0]].values
                actor_x, actor_y = (
                    actor_track[i, raw_data_format["X"]],
                    actor_track[i, raw_data_format["Y"]],
                )
                agent_x, agent_y = (
                    agent_track[i, raw_data_format["X"]],
                    agent_track[i, raw_data_format["Y"]],
                )
                instant_distance = np.sqrt((agent_x - actor_x)**2 + (agent_y - actor_y)**2)
                # here you can compare instant distance with a threshold to decide
                leading_actor_dist.append(instant_distance)

                # APPEND PHYISCS FEATURES HERE
                leading_physics = physics_df.loc[(physics_df['SEQUENCE'] == seq_id) & (physics_df['TRACK_ID'] == sort_distances_front[0][0])]
                # # Uncomment to debug
                # print("SEQUENCE_ID", seq_id)
                # print(physics_df)
                # print("LEADING")
                # print(leading_physics)
                # print(leading_physics['X'])
                # print(leading_physics['X'].values[0][0])
                # print(leading_physics['X'].values[0][1])
                leading_x.append(leading_physics['X'].values[0][i])
                leading_y.append(leading_physics['Y'].values[0][i])
                leading_vel_x.append(leading_physics['VEL_X'].values[0][i])
                leading_vel_y.append(leading_physics['VEL_Y'].values[0][i])
                leading_acc_x.append(leading_physics['ACC_X'].values[0][i])
                leading_acc_y.append(leading_physics['ACC_Y'].values[0][i])
                leading_jerk_x.append(leading_physics['JERK_X'].values[0][i])
                leading_jerk_y.append(leading_physics['JERK_Y'].values[0][i])
                leading_yaw.append(leading_physics['YAW'].values[0][i])
                leading_yaw_rate.append(leading_physics['YAW_RATE'].values[0][i])

            else:
                leading_actor.append(-1)
                leading_actor_dist.append(-1)
                leading_x.append(0)
                leading_y.append(0)
                leading_vel_x.append(0)
                leading_vel_y.append(0)
                leading_acc_x.append(0)
                leading_acc_y.append(0)
                leading_jerk_x.append(0)
                leading_jerk_y.append(0)
                leading_yaw.append(0)
                leading_yaw_rate.append(0)

            # Need to clean these up for the next timestep
            actor_info_front.clear()
            actor_info_back.clear()
        return tuple(leading_actor), tuple(following_actor), tuple(leading_actor_dist), tuple(following_actor_dist), \
                tuple(following_x), tuple(following_y), tuple(following_vel_x), tuple(following_vel_y), tuple(following_acc_x), \
                tuple(following_acc_y), tuple(following_jerk_x), tuple(following_jerk_y), tuple(following_yaw), tuple(following_yaw_rate), \
                tuple(leading_x), tuple(leading_y), tuple(leading_vel_x), tuple(leading_vel_y), tuple(leading_acc_x), \
                tuple(leading_acc_y), tuple(leading_jerk_x), tuple(leading_jerk_y), tuple(leading_yaw), tuple(leading_yaw_rate)




    def get_is_front_or_back(
            self,
            track: np.ndarray,
            neigh_x: float,
            neigh_y: float,
            raw_data_format: Dict[str, int],
    ):
        """Check if the neighbor is in front or back of the track.

        Args:
            track (numpy array): Track data
            neigh_x (float): Neighbor x coordinate
            neigh_y (float): Neighbor y coordinate
        Returns:
            _ (str): 'front' if in front, 'back' if in back

        """
        # We don't have heading information. So we need at least 2 coordinates to determine that.
        # Here, front and back is determined wrt to last 2 coordinates of the track
        x2 = track[-1, raw_data_format["X"]]
        y2 = track[-1, raw_data_format["Y"]]

        # Keep taking previous coordinate until first distinct coordinate is found.
        idx1 = track.shape[0] - 2
        while idx1 > -1:
            x1 = track[idx1, raw_data_format["X"]]
            y1 = track[idx1, raw_data_format["Y"]]
            if x1 != x2 or y1 != y2:
                break
            idx1 -= 1

        # If all the coordinates in the track are the same, there's no way to find front/back
        if idx1 < 0:
            return None

        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([neigh_x, neigh_y])
        proj_dist = np.abs(np.cross(p2 - p1,
                                    p1 - p3)) / np.linalg.norm(p2 - p1)

        # Interested in only those neighbors who are not far away from the direction of travel
        if proj_dist < FRONT_OR_BACK_OFFSET_THRESHOLD:

            dist_from_end_of_track = np.sqrt(
                (track[-1, raw_data_format["X"]] - neigh_x)**2 +
                (track[-1, raw_data_format["Y"]] - neigh_y)**2)
            dist_from_start_of_track = np.sqrt(
                (track[0, raw_data_format["X"]] - neigh_x)**2 +
                (track[0, raw_data_format["Y"]] - neigh_y)**2)
            dist_start_end = np.sqrt((track[-1, raw_data_format["X"]] -
                                      track[0, raw_data_format["X"]])**2 +
                                     (track[-1, raw_data_format["Y"]] -
                                      track[0, raw_data_format["Y"]])**2)

            return ("front"
                    if dist_from_end_of_track < dist_from_start_of_track
                    and dist_from_start_of_track > dist_start_end else "back")

        else:
            return None