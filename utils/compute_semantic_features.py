import numpy as np
import pandas as pd
import math
from argoverse.utils.centerline_utils import get_nt_distance

from argoverse.map_representation.map_api import ArgoverseMap

def compute_semantic_features(scene_df, agent_list, precomp_lanes,raw_data_format, map_inst,seq_id):  

	def get_semantic_index(segments,lane_dict):
		num_seg = len(segments)

		stop_sign_index    = [-1] * num_seg
		intersection_index = [-1] * num_seg
		three_plus_index   = [-1] * num_seg
		directions         = ['NONE'] * num_seg

		cur_stop_sign = -1
		cur_intersection = -1
		cur_three_plus = -1


		for ind in reversed(range(num_seg)):
			lane_seg = lane_dict[segments[ind]]
			if lane_seg.has_traffic_control:
				cur_stop_sign = ind

			if lane_seg.is_intersection:
				cur_intersection = ind

			if lane_seg.successors is not None:
				if len(lane_seg.successors) > 1:
					cur_three_plus = ind
			else:
				cur_three_plus = -1

			stop_sign_index[ind] = cur_stop_sign
			intersection_index[ind] = cur_intersection
			three_plus_index[ind] = cur_three_plus
			directions[ind] = lane_seg.turn_direction


		return stop_sign_index, intersection_index, three_plus_index, directions



	def get_distances(start_coor,end_coor,centerlines):
		#print("Start: ",start_coor)
		#print("End: ",end_coor)
		dist = get_nt_distance(np.array([start_coor, end_coor]), np.array(centerlines))[1][1]
		#print("Dist: ",dist)

		return dist

	def get_segment_ids(coors,segments,lane_dict):
		shortest_dist = -1
		index = -1
		found = 0
		segment_ids = [-1] * len(coors)

		for c, coor in enumerate(coors):
			for ind, segment in enumerate(segments):
				if found == 0:
					lane_seg = lane_dict[segment]
					for centerlines in lane_seg.centerline:
						dist =  math.hypot(coor[0] - centerlines[0], coor[1] - centerlines[1])
						if(shortest_dist == -1):
							shortest_dist = dist
						elif(dist <= shortest_dist):
							shortest_dist = dist
						else:
							segment_ids[c] = ind
							found = 1
							break
				else:
					found = 0
					break
		return segment_ids


	dist_to_intersect_feature  = []
	dist_to_stop_feature 	   = []
	dist_to_three_plus_feature = []
	turning_direction_feature  = []


	track_id = agent_list[0]

	agent_df    = scene_df[scene_df["TRACK_ID"] == track_id]
	agent_track = scene_df[scene_df["TRACK_ID"] == track_id].values
	agent_xy    = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]]].astype("float")

	precomp_df  = precomp_lanes[precomp_lanes["TRACK_ID"] == track_id]
	centerlines = precomp_df["CENTERLINES"].values[0]
	segments    = precomp_df["LANE_SEGMENTS"].values[0]

	city_name = agent_df["CITY_NAME"].values[0]

	try:
		tmp = map_inst.city_lane_centerlines_dict[city_name][segments[0][1][0]]

	except:
		if city_name == 'MIA' :
			city_name = 'PIT'
		else:
			city_name = 'MIA'

	#print("City name: ", city_name, "Track ID: ",track_id)

	lane_dict = map_inst.city_lane_centerlines_dict[city_name]
	inf_far = 1000

	dist_to_intersect_per_track = []
	dist_to_stop_per_track = []
	dist_to_three_plus_per_track = []
	turning_direction_per_track = []
	centerline_list = []

	for s,centerline in enumerate(centerlines):
		dist_to_intersect_per_centerline  = []
		dist_to_stop_per_centerline = []
		dist_to_three_plus_per_centerline = []
		turning_direction_per_centerline = []

		xy = agent_xy	
		segment_index = get_segment_ids(xy,segments[s][1],lane_dict)
		stop_sign_index, intersection_index, three_plus_index, directions = get_semantic_index(segments[s][1],lane_dict)

		for c,coor in enumerate(xy):
			stop_sign_seg = stop_sign_index[segment_index[c]]
			if(stop_sign_seg != -1):
				stop_sign_coor = lane_dict[segments[s][1][stop_sign_seg]].centerline[-1]
				dist_to_stop_per_centerline.append(get_distances(coor,stop_sign_coor,centerline[1]))

			else:
				dist_to_stop_per_centerline.append(inf_far)


			intersection_seg = intersection_index[segment_index[c]]
			if(intersection_seg != -1):
				intersection_coor = lane_dict[segments[s][1][intersection_seg]].centerline[-1]
				dist_to_intersect_per_centerline.append(get_distances(coor,intersection_coor,centerline[1]))
			else:
				dist_to_intersect_per_centerline.append(inf_far)


			three_plus_seg = three_plus_index[segment_index[c]]
			if(three_plus_seg != -1):
				three_plus_coor = lane_dict[segments[s][1][three_plus_seg]].centerline[-1]
				dist_to_three_plus_per_centerline.append(get_distances(coor,three_plus_coor,centerline[1]))
			else:
				dist_to_three_plus_per_centerline.append(inf_far)

			turning_direction = directions[segment_index[c]]
			turning_direction_per_centerline.append(turning_direction)

		centerline_list.append((s, centerline))


		dist_to_intersect_per_track.append((s, dist_to_intersect_per_centerline))
		dist_to_stop_per_track.append((s, dist_to_stop_per_centerline))
		dist_to_three_plus_per_track.append((s, dist_to_three_plus_per_centerline))
		turning_direction_per_track.append((s, turning_direction_per_centerline))

	#dist_to_intersect_feature.append(dist_to_intersect_per_track)
	#dist_to_stop_feature.append(dist_to_stop_per_track)
	#dist_to_three_plus_feature.append(dist_to_three_plus_per_track)
	#turning_direction_feature.append(turning_direction_per_track)


	columns = ["SEQUENCE", "TRACK_ID", "CENTERLINES", "DIST_TO_INTERSECTION", "DIST_TO_STOP", "DIST_TO_THREE_PLUS", "TURNING_DIRECTION"]
	rows = [ [ seq_id, track_id, centerline_list, dist_to_intersect_per_track, dist_to_stop_per_track, dist_to_three_plus_per_track, turning_direction_per_track ] ]


	return columns, rows






