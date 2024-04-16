import os
import csv
import pandas as pd
import numpy as np
import shutil #new import
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from statistics import mean
os.environ["DLClight"]="True"
import deeplabcut
import dlc2kinematics
import argparse

from lib.primary_analysis import deeplabcut_analyze_video, process_csv_to_dataframe_filter, add_distance_columns, add_rod_columns
from lib.generate_scores import add_extrema_columns, compute_and_add_joint_extrema, calculate_and_append_data_means

parser = argparse.ArgumentParser()
parser.add_argument("--video_folder", 
                    help="path to load rotarod run videos")
parser.add_argument("--comparison_csv", 
                    help="path to .csv containing list of rotarod run comparisons")
parser.add_argument("--output_folder", 
                    help="path to save gait analysis computation results")

args = parser.parse_args()

trial_name = "Testing" # [UPDATE]

data_folder_path = os.path.join("/content/drive/MyDrive/DLC Analyses", trial_name)
dlc_analyze_path = os.path.join(data_folder_path, "deeplabcut.analyze")

if os.path.exists(data_folder_path):
    shutil.rmtree(data_folder_path)
    os.mkdir(data_folder_path)
    os.mkdir(dlc_analyze_path)


# Column names
column_names = [
    "Mouse ID",
    "Induction",
    "Baseline Week",
    "Sample Week",
    "CoG Score",
    "Left Leg Drag Score",
    "Right Leg Drag Score",
    "Left Decrease in Angle Flexion",
    "Right Decrease in Angle Flexion",
    "Left Decrease in Angle FOM",
    "Right Decrease in Angle FOM"
]

# Path to save the CSV file
stats_path = os.path.join(data_folder_path, 'secondary_stats.csv')

# Create the CSV file and write the header row
with open(stats_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(column_names)

# Specify the directory path
raw_videos_path = "/content/drive/MyDrive/Fall 2023 (MARC)/DLC/Experiment1" # [UPDATE]

# Initialize an empty list to store the file names
mp4_files = []

# Recursively search for .mp4 files in the root_directory and its subdirectories
for root, dirs, files in os.walk(raw_videos_path):
    for filename in files:
        if filename.lower().endswith(".mp4") and "resnet" not in filename.lower() and "shuffle" not in filename.lower():
            # Append the file path to the list
            mp4_files.append(filename)

compare_csv = "/content/drive/MyDrive/Fall 2023 (MARC)/DLC/Experiment1/comparison_id.csv"
compare_id_df = pd.read_csv(compare_csv)

for index, row in compare_id_df.iterrows():
    base = row[0]
    samp = row[1]

    # Find matching file paths for the base and samp values
    matching_paths_base = []
    matching_paths_samp = []

    for file_path in mp4_files:
        if all(part.strip() in file_path for part in base.split(',')):
            matching_paths_base.append(file_path)
        if all(part.strip() in file_path for part in samp.split(',')):
            matching_paths_samp.append(file_path)

    # Set conditions to choose the best-matching video paths
    def is_desired_path(path):
        keywords = ["resnet", "shuffle", "filtered"]
        if not any(keyword in path for keyword in keywords):
            return True
        return False

    matching_paths_base = [path.replace(".mp4", "") for path in matching_paths_base if is_desired_path(path)]
    matching_paths_samp = [path.replace(".mp4", "") for path in matching_paths_samp if is_desired_path(path)]

    if matching_paths_base:
        base_vid_path = os.path.join(raw_videos_path, matching_paths_base[0] + ".mp4")
    else:
        base_vid_path = None

    if matching_paths_samp:
        samp_vid_path = os.path.join(raw_videos_path, matching_paths_samp[0] + ".mp4")
    else:
        samp_vid_path = None

    # Now, base_vid_path and samp_vid_path contain the best-matching video paths

# deeplabcut.analyze videos
base_dlc_analyze_path = deeplabcut_analyze_video(dlc_analyze_path, base_vid_path)
samp_dlc_analyze_path = deeplabcut_analyze_video(dlc_analyze_path, samp_vid_path)

def find_file_by_keywords(file_list, keywords):

    for file_path in file_list:
        # Check if all keywords are present in the file path
        if all(keyword.lower() in file_path.lower() for keyword in keywords):
            matching_files = file_path.replace(".csv", "")

    return matching_files

file_list = os.listdir(base_dlc_analyze_path)
base_keywords = [matching_paths_base[0], "filtered.csv"]
samp_keywords = [matching_paths_samp[0], "filtered.csv"]

base_filtered_csv_file = find_file_by_keywords(file_list, base_keywords)
samp_filtered_csv_file = find_file_by_keywords(file_list, samp_keywords)

base_filtered_csv_path = os.path.join(base_dlc_analyze_path, base_filtered_csv_file)
samp_filtered_csv_path = os.path.join(samp_dlc_analyze_path, samp_filtered_csv_file)

# remove outliers
base_filtered_df = process_csv_to_dataframe_filter(base_filtered_csv_path)
samp_filtered_df = process_csv_to_dataframe_filter(samp_filtered_csv_path)

# # base_filtered_df = process_csv_to_dataframe_filter(a2_base)
# # samp_filtered_df = process_csv_to_dataframe_filter(a2_samp)

add_distance_columns(base_filtered_df)
add_rod_columns(base_filtered_df)

add_distance_columns(samp_filtered_df)
add_rod_columns(samp_filtered_df)

n = 6

# Add extrema columns for various data columns
columns_to_process = [
    'leftpaw_x_d',
    'rightpaw_x_d',
    'rightpaw_y',
    'leftpaw_y',
    'tailbase_rod_y_dist',
]

for column in columns_to_process:
    add_extrema_columns(base_filtered_df, column, n)
    add_extrema_columns(samp_filtered_df, column, n)

primary_stats_df_base, base_joint_angles_df = compute_and_add_joint_extrema(base_filtered_df, base_filtered_csv_path, n)
primary_stats_df_samp, samp_joint_angles_df = compute_and_add_joint_extrema(samp_filtered_df, samp_filtered_csv_path, n)

data_means_new_base = []
data_means_new_samp = []

calculate_and_append_data_means(primary_stats_df_base, data_means_new_base)
calculate_and_append_data_means(primary_stats_df_samp, data_means_new_samp)

Mouse_Data_means_new_base = pd.DataFrame(data_means_new_base)
Mouse_Data_means_new_samp = pd.DataFrame(data_means_new_samp)

def pcnt_change(val1, val2):
    return ((val2 - val1) / val1) * 100

variables = ["H_L_TP Mean", "H_R_TP Mean", "JA_L Mean", "JA_R Mean"]

changes = [pcnt_change(Mouse_Data_means_new_base[var], Mouse_Data_means_new_samp[var]) for var in variables]

center_grav = float(changes[1] - changes[0] - changes[2] + changes[3])

def pcnt_change_flex(val1, val2):
    return ((val1 - val2) / val1) * 100  # Convert to percentage

def id_isolate(csv_path):
    return csv_path.split("/")[-1][:csv_path.split("/")[-1].find("DLC_dlc")]

variables = {
    'JA_L': ('left_freedom_movement', 'left_flexion', 'left_drag_factor', 'left_drag_score'),
    'JA_R': ('right_freedom_movement', 'right_flexion', 'right_drag_factor', 'right_drag_score')
}

joint_results = {}

for joint, (freedom_var, flexion_var, drag_factor_var, drag_score_var) in variables.items():
    base_max = Mouse_Data_means_new_base[joint + " Max"]
    samp_max = Mouse_Data_means_new_samp[joint + " Max"]
    base_min = Mouse_Data_means_new_base[joint + ' Min']
    samp_min = Mouse_Data_means_new_samp[joint + ' Min']
    base_dif = base_max - base_min
    samp_dif = samp_max - samp_min

    print(f"{joint} base_max: {base_max[0]}",
          f"{joint} samp_max: {samp_max[0]}",
          f"{joint} base_min: {base_min[0]}",
          f"{joint} samp_min: {samp_min[0]}",
          f"{joint} base_dif: {base_dif[0]}",
          f"{joint} samp_dif: {samp_dif[0]}",
          sep = "\n")

    flexion = pcnt_change_flex(base_min, samp_min)
    freedom_movement = pcnt_change(base_dif, samp_dif)
    drag_factor = pcnt_change(base_max, samp_max)
    drag_score = drag_factor - flexion

    joint_results[flexion_var] = f"{round(float(flexion), 1)}%"
    joint_results[freedom_var] = f"{round(float(freedom_movement), 1)}%"
    joint_results[drag_factor_var] = f"{round(float(drag_factor), 1)}%"
    joint_results[drag_score_var] = f"{round(float(drag_score), 1)}%"

# Add additional information to the joint_results dictionary
joint_results['Base ID'] = id_isolate(base_filtered_csv_path)
joint_results['Sample ID'] = id_isolate(samp_filtered_csv_path)
joint_results['center_gravity'] = f"{round(float(center_grav), 1)}%"  # Assuming center_grav is in percentage

joint_results['Baseline Week'] = int(str(joint_results['Base ID']).split("_")[1].replace("D", ""))//7
joint_results['Sample Week'] = int(str(joint_results['Sample ID']).split("_")[1].replace("D", ""))//7
joint_results['Mouse ID'] = str(joint_results['Base ID']).split("_")[2]

# Convert the joint_results dictionary to a Pandas DataFrame
result_df = pd.DataFrame([joint_results])

# Reorder columns to have 'Base ID', 'Sample ID', 'Center of Gravity' as the first three columns
result_df = result_df[['Mouse ID', 'Base ID', 'Sample ID', 'Baseline Week', 'Sample Week', 'center_gravity', ] + list(variables['JA_L']) + list(variables['JA_R'])]
