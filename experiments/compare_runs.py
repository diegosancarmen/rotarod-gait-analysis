"""
cd /home/coeguest/hdelacruz/DeepLabCut
conda activate DLC_gait
python rotarod-gait-analysis/experiments/compare_runs.py --video_folder /home/coeguest/hdelacruz/DeepLabCut/sample --output_folder /home/coeguest/hdelacruz/DeepLabCut/test/
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

import argparse
import sys
import cv2

root_path = os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)))

sys.path.append(root_path)

from lib.gait_signatures.primary_analysis import deeplabcut_analyze_video, video_array, process_csv_to_dataframe_filter, add_distance_columns, add_rod_columns
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

config_yaml = os.path.join(root_path, "automated_analysis/config.yaml")
dlc_analyze_path = os.path.join(output_folder, "deeplabcut.analyze")
vid_collage_path = os.path.join(output_folder, "video_array")

os.makedirs(output_folder, exist_ok = True)
os.makedirs(dlc_analyze_path, exist_ok = True)
os.makedirs(vid_collage_path, exist_ok = True)

# Column names
row_headers = [
    "Mouse ID,,,,,Paw-Tail Distance,, Ankle Joint Angle,, CoG Pre-Score,, FOM Ankle Joint Angle,,,,,,Left Leg,,,,Right Leg,,,,", 
    "Name, Induction Week, File ID#, Frame #,, H_L_TP Mean, H_R_TP Mean, JA_L Mean, JA_R Mean,, , JA_L Max, JA_L Min, JA_L Diff, JA_R Max, JA_R Min, JA_R Diff, % Change (FOM), Flexion Decline, Drag Factor, Drag Score, % Change (FOM), Flexion Decline, Drag Factor, Drag Score,"
    ]

os.makedirs(output_folder, exist_ok = True)

# Path to save the CSV file
stats_path = os.path.join(output_folder, 'gait_metrics.csv')
    
# Create the CSV file and write the header rows
with open(stats_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for header_row in row_headers:
        csv_row = header_row.split(",")
        writer.writerow(csv_row)
    
comparison_csv = os.path.join(args.video_folder, "comparison_id.csv")

config_yaml = os.path.join(root_path, "automated_analysis/config.yaml")
dlc_analyze_path = os.path.join(output_folder, "deeplabcut.analyze")

os.makedirs(output_folder, exist_ok = True)
os.makedirs(dlc_analyze_path, exist_ok = True)

# Recursively search for .mp4 files in the root_directory and its subdirectories
mp4_files = list(filter(lambda f: f.endswith('.mp4'), os.listdir(args.video_folder)))

compare_id = pd.read_csv(comparison_csv)
compare_id.columns = ["base", "base_start", "base_end", "samp", "samp_start", "samp_end"]
compare_id[['base_start', 'samp_start']] = compare_id[['base_start', 'samp_start']].fillna(0)

dataframe_list = []

for index, row in compare_id.iterrows():
    base, base_start, base_end = row["base"], row["base_start"], row["base_end"]
    samp, samp_start, samp_end = row["samp"], row["samp_start"], row["samp_end"]
 
    # deeplabcut.analyze videos
    base_dlc_analyze_path, base_trim = deeplabcut_analyze_video(dlc_analyze_path, args.video_folder, base, base_start, base_end, config_yaml)
    samp_dlc_analyze_path, samp_trim = deeplabcut_analyze_video(dlc_analyze_path, args.video_folder, samp, samp_start, samp_end, config_yaml)
        
    base_df = os.path.join(base_dlc_analyze_path, f"{base_trim}DLC_dlcrnetms5_Trial9May23shuffle1_150000_el_filtered.csv")
    samp_df = os.path.join(samp_dlc_analyze_path, f"{samp_trim}DLC_dlcrnetms5_Trial9May23shuffle1_150000_el_filtered.csv")

    # remove outliers
    base_filtered_df, base_df_shape = process_csv_to_dataframe_filter(base_df)
    samp_filtered_df, samp_df_shape = process_csv_to_dataframe_filter(samp_df)
    
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
    
    samp_idx = samp_filtered_df[
        (samp_filtered_df['rightpaw_y']>(samp_filtered_df['rightpaw_y_max'].mean() + 170)) | 
        (samp_filtered_df['rightpaw_y']<(samp_filtered_df['rightpaw_y_min'].mean() - 170)) |
        (samp_filtered_df['leftpaw_y']>(samp_filtered_df['leftpaw_y_max'].mean() + 170)) | 
        (samp_filtered_df['leftpaw_y']<(samp_filtered_df['leftpaw_y_min'].mean() - 170))  
        ].index.to_list()
    
    base_filtered_df = base_filtered_df.drop(base_idx)
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
        
    joint_results = calculate_joint_results(joint_variables, Mouse_Data_means_new_base, Mouse_Data_means_new_samp, base, samp, center_grav, cog_dict)
        
    # print(json.dumps(joint_results, sort_keys=True, indent=4))
    
    
    gait_analysis_rows = [
        f"{joint_results['Mouse ID']}, {joint_results['Baseline Week']}, {joint_results['Base ID']}, {base_df_shape[0]},, {joint_results['Base H_L_TP Mean']}, {joint_results['Base H_R_TP Mean']}, {joint_results['Base JA_L Mean']}, {joint_results['Base JA_R Mean']}, {joint_results['center_gravity']},, {joint_results['JA_L Base Max']}, {joint_results['JA_L Base Min']}, {joint_results['JA_L Base Diff']}, {joint_results['JA_R Base Max']}, {joint_results['JA_R Base Min']}, {joint_results['JA_R Base Diff']}, {joint_results['left_freedom_movement']}, {joint_results['left_flexion']}, {joint_results['left_drag_factor']}, {joint_results['left_drag_score']}, {joint_results['right_freedom_movement']}, {joint_results['right_flexion']}, {joint_results['right_drag_factor']}, {joint_results['right_drag_score']},",
        f"{joint_results['Mouse ID']}, {joint_results['Sample Week']}, {joint_results['Sample ID']}, {samp_df_shape[0]},, {joint_results['Samp H_L_TP Mean']}, {joint_results['Samp H_R_TP Mean']}, {joint_results['Samp JA_L Mean']}, {joint_results['Samp JA_R Mean']},,, {joint_results['JA_L Samp Max']}, {joint_results['JA_L Samp Min']}, {joint_results['JA_L Samp Diff']}, {joint_results['JA_R Samp Max']}, {joint_results['JA_R Samp Min']}, {joint_results['JA_R Samp Diff']},,,,,,,,,",
    ]

    # Create the CSV file and write the header rows
    with open(stats_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for header_row in gait_analysis_rows:
            csv_row = header_row.split(",")
            writer.writerow(csv_row)

    base_samp_collage = video_array(base_dlc_analyze_path, base_trim, samp_dlc_analyze_path, samp_trim, vid_collage_path)
