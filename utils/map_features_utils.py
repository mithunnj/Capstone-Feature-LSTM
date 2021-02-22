"""This module is used for computing map features for motion forecasting baselines."""

from typing import Any, Dict, List, Tuple
import pandas as pd                     
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import cascaded_union

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    get_nt_distance,
    remove_overlapping_lane_seq,
    get_normal_and_tangential_distance_point,
)
from argoverse.utils.mpl_plotting_utils import visualize_centerline
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
        diverse_centerlines = []
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
                    diverse = False
            if diverse:
                diverse_centerlines.append(centerline)
                diverse_scores.append(score)

        num_diverse_centerlines = min(len(diverse_centerlines),
                                      max_candidates - aligned_cl_count)
        test_centerlines = aligned_centerlines
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
            test_centerlines += diverse_centerlines

        return test_centerlines

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

        # Use a set to keep track of a unique list of future and past nodes
        unique_segments_future = set()
        unique_segments_past = set()

        # DFS to get all successor and predecessor candidates
        obs_pred_lanes: List[Sequence[int]] = []
        for lane in curr_lane_candidates:
            candidates_future = avm.dfs(lane, city_name, 0,
                                        dfs_threshold_front)
            candidates_past = avm.dfs(lane, city_name, 0, dfs_threshold_back,
                                      True)

            # Add items to the sets
            if mode == "compute_all":
                for future_lane_seq in candidates_future:
                    unique_segments_future.update(future_lane_seq)
                
                for past_lane_seq in candidates_past:
                    unique_segments_past.update(past_lane_seq)
                
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
        if mode == "test" or mode == "compute_all":
            candidate_centerlines = self.get_heuristic_centerlines_for_test_set(
                obs_pred_lanes, xy, city_name, avm, max_candidates, scores)
        else:
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

        if mode == "compute_all":
            return candidate_centerlines, obs_pred_lanes, list(unique_segments_future), list(unique_segments_past), curr_lane_candidates

        return candidate_centerlines

    def compute_map_features(
            self,
            df: pd.DataFrame,
            track_id,
            agent_track: np.ndarray,
            obs_len: int,
            seq_len: int,
            raw_data_format: Dict[str, int],
            mode: str,
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
        avm = ArgoverseMap()

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


        # loop thru all the actor tracks, and calculate
        # the following scoring, and leading scoring for each timestep for all the actors. 
        # then, keep the actor id of the highest scores for each timestep, and return them
        # as an array.
        #
        # an intial algorithm checks for distance to the centerline at every timestep and 
        # the closest vehicles (within a threshold) to the centerline will be returned.
        leading_actor = list()
        following_actor = list()
        leading_actor_dist = list()
        following_actor_dist = list()
        actor_info_front = dict()
        actor_info_back = dict()
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

                if (instant_distance >= self.NEARBY_DISTANCE_THRESHOLD):
                    continue

                is_front_or_back = self.get_is_front_or_back(
                    agent_track[:2, :] if i == 0 else agent_track[:i + 1, :],
                    actor_x,
                    actor_y,
                    raw_data_format,
                )
                
                # here implementing only for the first centerline, we can change this later accordingly
                shortest_distance_to_candidate = get_normal_and_tangential_distance_point(
                    actor_x,
                    actor_y,
                    candidate_centerlines[0],
                )

                if is_front_or_back == "front":
                    actor_info_front[actor_id] = shortest_distance_to_candidate

                elif is_front_or_back == "back":                
                    actor_info_back[actor_id] = shortest_distance_to_candidate

            # now append to the list of lead and following agents
            # need to handle the case where there is no items in these dictionaries
            sort_distances_back = sorted(actor_info_back.items(), key=lambda x: x[1])
            if sort_distances_back:
                # print("REACHED back")
                # print(sort_distances_back[0][0])
                following_actor.append(sort_distances_back[0][0])

                actor_track = df[df["TRACK_ID"] == sort_distances_back[0][0]].values
                actor_x, actor_y = (
                    actor_track[i, raw_data_format["X"]],
                    actor_track[i, raw_data_format["Y"]],
                )
                agent_x, agent_y = (
                    agent_track[i, raw_data_format["X"]],
                    agent_track[i, raw_data_format["Y"]],
                )
                instant_distance = np.sqrt((agent_x - actor_x)**2 + (agent_y - actor_y)**2)
                following_actor_dist.append(instant_distance)
            else:
                following_actor.append(-1)
                following_actor_dist.append(-1)


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
                leading_actor_dist.append(instant_distance)
            else:
                leading_actor.append(-1)
                leading_actor_dist.append(-1)

            # Need to clean these up for the next timestep
            actor_info_front.clear()
            actor_info_back.clear()

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