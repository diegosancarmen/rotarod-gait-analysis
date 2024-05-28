"""
cd /home/coeguest/hdelacruz/DeepLabCut
conda activate DEEPLABCUT
python rotarod-gait-analysis/experiments/compare_runs.py --video_folder /home/coeguest/hdelacruz/DeepLabCut/Experiment2
"""

import os
import csv
import pandas as pd
import numpy as np
import shutil #new import
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from statistics import mean
os.environ["DLClight"]="True"

# import deeplabcut
# import dlc2kinematics

import argparse
import sys

sys.path.append("/home/coeguest/hdelacruz/DeepLabCut/rotarod-gait-analysis")

from lib.gait_signatures.primary_analysis import deeplabcut_analyze_video, process_csv_to_dataframe_filter, add_distance_columns, add_rod_columns
from lib.gait_signatures.generate_scores import add_extrema_columns, compute_and_add_joint_extrema, calculate_and_append_data_means, calculate_joint_results, pcnt_change

parser = argparse.ArgumentParser()
parser.add_argument("--video_folder", 
                    help="path to load rotarod run videos")
parser.add_argument("--output_folder", 
                    help="path to load rotarod run videos", default = None)

args = parser.parse_args()

trial_name = args.video_folder.split("/")[-1]
if args.output_folder is None:
    output_folder = args.video_folder + "_output"
else:
    output_folder = args.output_folder
    
comparison_csv = os.path.join(args.video_folder, "comparison_id.csv")

config_yaml = "/home/coeguest/hdelacruz/DeepLabCut/automated_analysis/config.yaml"
dlc_analyze_path = os.path.join(output_folder, "deeplabcut.analyze")

os.makedirs(output_folder, exist_ok = True)
os.makedirs(dlc_analyze_path, exist_ok = True)

# if os.path.exists(output_folder):
#     shutil.rmtree(output_folder)
#     os.mkdir(output_folder)
#     os.mkdir(dlc_analyze_path)

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
stats_path = os.path.join(output_folder, 'gait_metrics.csv')

# Create the CSV file and write the header row
with open(stats_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(column_names)

# Recursively search for .mp4 files in the root_directory and its subdirectories
mp4_files = list(filter(lambda f: f.endswith('.mp4'), os.listdir(args.video_folder)))

compare_id_df = pd.read_csv(comparison_csv)

dataframe_list = []

for index, row in compare_id_df.iterrows():
    base = row[0]
    samp = row[1]

    base_vid_path = os.path.join(args.video_folder, base + ".mp4")
    samp_vid_path = os.path.join(args.video_folder, samp + ".mp4")      

    # deeplabcut.analyze videos
    base_dlc_analyze_path = deeplabcut_analyze_video(dlc_analyze_path, base_vid_path, config_yaml)
    samp_dlc_analyze_path = deeplabcut_analyze_video(dlc_analyze_path, samp_vid_path, config_yaml)
    
    base_df = os.path.join(base_dlc_analyze_path, f"{base}DLC_dlcrnetms5_Trial9May23shuffle1_150000_el_filtered.csv")
    samp_df = os.path.join(samp_dlc_analyze_path, f"{samp}DLC_dlcrnetms5_Trial9May23shuffle1_150000_el_filtered.csv")

    # remove outliers
    base_filtered_df = process_csv_to_dataframe_filter(base_df)
    samp_filtered_df = process_csv_to_dataframe_filter(samp_df)
    
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

    base_filtered_csv_path = os.path.join(base_dlc_analyze_path, [file for file in os.listdir(base_dlc_analyze_path) if base in file and 'filtered.h5' in file][0])
    samp_filtered_csv_path = os.path.join(samp_dlc_analyze_path, [file for file in os.listdir(samp_dlc_analyze_path) if samp in file and 'filtered.h5' in file][0])

    base_idx = base_filtered_df[
        (base_filtered_df['rightpaw_y']>(base_filtered_df['rightpaw_y_max'].mean() + 170)) | 
        (base_filtered_df['rightpaw_y']<(base_filtered_df['rightpaw_y_min'].mean() - 170)) |
        (base_filtered_df['leftpaw_y']>(base_filtered_df['leftpaw_y_max'].mean() + 170)) | 
        (base_filtered_df['leftpaw_y']<(base_filtered_df['leftpaw_y_min'].mean() - 170))  
        ].index.to_list()
    base_filtered_df = base_filtered_df.drop(base_idx)

    samp_idx = samp_filtered_df[
        (samp_filtered_df['rightpaw_y']>(samp_filtered_df['rightpaw_y_max'].mean() + 170)) | 
        (samp_filtered_df['rightpaw_y']<(samp_filtered_df['rightpaw_y_min'].mean() - 170)) |
        (samp_filtered_df['leftpaw_y']>(samp_filtered_df['leftpaw_y_max'].mean() + 170)) | 
        (samp_filtered_df['leftpaw_y']<(samp_filtered_df['leftpaw_y_min'].mean() - 170))  
        ].index.to_list()
    samp_filtered_df = samp_filtered_df.drop(samp_idx)


    primary_stats_df_base, base_joint_angles_df = compute_and_add_joint_extrema(base_filtered_df, base_filtered_csv_path, n)
    primary_stats_df_samp, samp_joint_angles_df = compute_and_add_joint_extrema(samp_filtered_df, samp_filtered_csv_path, n)

    data_means_new_base = []
    data_means_new_samp = []

    calculate_and_append_data_means(primary_stats_df_base, data_means_new_base)
    calculate_and_append_data_means(primary_stats_df_samp, data_means_new_samp)

    Mouse_Data_means_new_base = pd.DataFrame(data_means_new_base)
    Mouse_Data_means_new_samp = pd.DataFrame(data_means_new_samp)

    cog_variables = ["H_L_TP Mean", "H_R_TP Mean", "JA_L Mean", "JA_R Mean"]
    
    changes = [pcnt_change(Mouse_Data_means_new_base[var], Mouse_Data_means_new_samp[var]) for var in cog_variables]

    # Initialize the dictionary with base and sample values
    cog_dict = {
        **{f"Base {var}": Mouse_Data_means_new_base[var].iloc[0] for var in cog_variables},
        **{f"Samp {var}": Mouse_Data_means_new_samp[var].iloc[0] for var in cog_variables},
        **{f"{cog_variables[idx]} % Diff": round(float(changes[idx].iloc[0]), 2) for idx in range(len(cog_variables))}
    }

    center_grav = float(changes[1] - changes[0] - changes[2] + changes[3])

    joint_variables = {
        'JA_L': ('left_freedom_movement', 'left_flexion', 'left_drag_factor', 'left_drag_score'),
        'JA_R': ('right_freedom_movement', 'right_flexion', 'right_drag_factor', 'right_drag_score')
    }
        
    Mouse_Data_means_new_base, Mouse_Data_means_new_samp, joint_results = calculate_joint_results(joint_variables, Mouse_Data_means_new_base, Mouse_Data_means_new_samp, base, samp, center_grav, cog_dict)

    # Convert the joint_results dictionary to a Pandas DataFrame
    result_df = pd.DataFrame([joint_results] + [cog_dict])
    
    # # # Reorder columns to have 'Base ID', 'Sample ID', 'Center of Gravity' as the first three columns
    # result_df = result_df[['Mouse ID', 'Base ID', 'Sample ID', 'Baseline Week', 'Sample Week', 'center_gravity', ] + list(joint_variables['JA_L']) + list(joint_variables['JA_R'])]
    
    dataframe_list.append(result_df)

# Concatenate all dataframes in the list
combined_df = pd.concat(dataframe_list, ignore_index=True)

# Save the combined dataframe to a CSV file
combined_df.to_csv(os.path.join(f"{args.video_folder}_output", "combined_data.csv"), index=False)
