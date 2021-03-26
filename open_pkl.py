"""
Prints out the argo map feature data.

Args:
    arg1: "mode" should be "train", "test", "val" or "compute_all" (capstone one)
    arg2: "verbose" optional flag which will print out each column separately.

Returns:
    Nothing

"""

import pickle
import pandas as pd
import sys

# Load and print the stored features

mode = sys.argv[1]

should_print_all = len(sys.argv) >= 3 and sys.argv[2] == "-verbose"

feature_data = pickle.load( open( "testing_map_features/forecasting_features_{}.pkl".format(mode), "rb" ) )

all_features_dataframe = pd.DataFrame(feature_data)

# Print out each datatype
if should_print_all: 

    for index, row in all_features_dataframe.iterrows():

        print("Row: ", index)

        for column in all_features_dataframe:
            print("{}:".format(column))
            print(row[column])

# Print out the columns
print("Columns: ")
print(all_features_dataframe.columns)

# Print out the number of candidate centerlines in each scene
for index, row in all_features_dataframe.iterrows():
    if "CANDIDATE_CENTERLINES" in row.keys():
        print("Sequence {} has {} candidates.".format(row["SEQUENCE"], len(row["CANDIDATE_CENTERLINES"])))

# Print out the head of the dataframe
print(all_features_dataframe.head())
