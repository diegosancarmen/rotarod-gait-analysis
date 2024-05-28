# !pip install DLC2Kinematics

import dlc2kinematics
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from datetime import datetime


def pcnt_change(val1, val2):
    return ((round(val2, 2) - round(val1, 2)) / round(val1, 2)) * 100
def pcnt_change_flex(val1, val2):
    return ((round(val1, 2) - round(val2, 2)) / round(val1, 2)) * 100  # Convert to percentage

# def pcnt_change(val1, val2):
#     return ((val2 - val1) / val1) * 100
# def pcnt_change_flex(val1, val2):
#     return ((val1 - val2) / val1) * 100  # Convert to percentage

def id_isolate(csv_path):
    return csv_path.split("/")[-1][:csv_path.split("/")[-1].find("DLC_dlc")]


def add_extrema_columns(df, column, n):
    min_column = f'{column}_min'
    max_column = f'{column}_max'

    min_indices = argrelextrema(df[column].values, np.less_equal, order=n)[0]
    max_indices = argrelextrema(df[column].values, np.greater_equal, order=n)[0]

    df[min_column] = df.iloc[min_indices][column]
    df[max_column] = df.iloc[max_indices][column]
    
def compute_and_add_joint_extrema(primary_stats_df, path, n):

    df, bodyparts, scorer = dlc2kinematics.load_data(path)

    joints_dict = {}
    joints_dict['R-Ankle'] = ['rightknee', 'rightankle', 'rightpaw']
    joints_dict['L-Ankle'] = ['leftknee', 'leftankle', 'leftpaw']

    joint_angles = dlc2kinematics.compute_joint_angles(df, joints_dict, save=False)
    
    column_names = ['rightankle_angle', 'leftankle_angle']
    joint_angles.columns = column_names
    
    primary_stats_df['leftankle_angle'] = joint_angles['leftankle_angle']
    primary_stats_df['leftankle_angle_min'] = joint_angles.iloc[argrelextrema(joint_angles.leftankle_angle.values, np.less_equal, order=n)[0]]['leftankle_angle']
    primary_stats_df['leftankle_angle_max'] = joint_angles.iloc[argrelextrema(joint_angles.leftankle_angle.values, np.greater_equal, order=n)[0]]['leftankle_angle']

    primary_stats_df['rightankle_angle'] = joint_angles['rightankle_angle']
    primary_stats_df['rightankle_angle_min'] = joint_angles.iloc[argrelextrema(joint_angles.rightankle_angle.values, np.less_equal, order=n)[0]]['rightankle_angle']
    primary_stats_df['rightankle_angle_max'] = joint_angles.iloc[argrelextrema(joint_angles.rightankle_angle.values, np.greater_equal, order=n)[0]]['rightankle_angle']


    # # Plotting
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    # # Plot left ankle angle
    # axes[0].plot(primary_stats_df.index, primary_stats_df['leftankle_angle'], label='Left Ankle Angle', color='blue')
    # axes[0].scatter(primary_stats_df.index, primary_stats_df['leftankle_angle_min'], label='Min Points', color='red')
    # axes[0].scatter(primary_stats_df.index, primary_stats_df['leftankle_angle_max'], label='Max Points', color='green')
    # axes[0].set_title('Left Ankle Angle with Min and Max Points')
    # axes[0].set_xlabel('Index')
    # axes[0].set_ylabel('Angle')
    # axes[0].legend()

    # # Plot right ankle angle
    # axes[1].plot(primary_stats_df.index, primary_stats_df['rightankle_angle'], label='Right Ankle Angle', color='blue')
    # axes[1].scatter(primary_stats_df.index, primary_stats_df['rightankle_angle_min'], label='Min Points', color='red')
    # axes[1].scatter(primary_stats_df.index, primary_stats_df['rightankle_angle_max'], label='Max Points', color='green')
    # axes[1].set_title('Right Ankle Angle with Min and Max Points')
    # axes[1].set_xlabel('Index')
    # axes[1].set_ylabel('Angle')
    # axes[1].legend()

    # plt.tight_layout()
    
    # # Save the plot with a timestamp to avoid overwriting
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"/home/coeguest/hdelacruz/DeepLabCut/Experiment2_test_output/max_min_plots_{timestamp}.png"
    # plt.savefig(filename)

    return primary_stats_df, joint_angles

def calculate_and_append_data_means(df, data_means_list):
    data_dict_means_new = {
        "V_L_PS Dist": round(df['leftpaw_y_max'].mean() - df['leftpaw_y_min'].mean(), 2),
        "V_R_PS Dist": round(df['rightpaw_y_max'].mean() - df['rightpaw_y_min'].mean(), 2),

        "Step_L_Avg": round(len(df[~df['leftpaw_y_max'].isnull() | ~df['leftpaw_y_min'].isnull()]) / 2, 0),
        "Step_L_Max": round(len(df[~df['leftpaw_y_max'].isnull()]), 2),
        "Step_L_Min": round(len(df[~df['leftpaw_y_min'].isnull()]), 2),

        "Step_R_Avg": round(len(df[~df['rightpaw_y_max'].isnull() | ~df['rightpaw_y_min'].isnull()]) / 2, 0),
        "Step_R_Max": round(len(df[~df['rightpaw_y_max'].isnull()]), 2),
        "Step_R_Min": round(len(df[~df['rightpaw_y_min'].isnull()]), 2),

        "H_L_TP Mean": round(df['leftpaw_x_d'].mean(), 2),
        "H_R_TP Mean": round(df['rightpaw_x_d'].mean(), 2),

        "JA_L Mean": round(df['leftankle_angle'].mean(), 2),
        "JA_L Max": round(df['leftankle_angle_max'].mean(), 2),
        "JA_L Min": round(df['leftankle_angle_min'].mean(), 2),

        "JA_R Mean": round(df['rightankle_angle'].mean(), 2),
        "JA_R Max": round(df['rightankle_angle_max'].mean(), 2),
        "JA_R Min": round(df['rightankle_angle_min'].mean(), 2),

        "R_TB_V Mean": round(df['tailbase_rod_y_dist'].mean(), 2)
    }

    data_means_list.append(data_dict_means_new)

def calculate_joint_results(variables, Mouse_Data_means_new_base, Mouse_Data_means_new_samp, base, samp, center_grav, cog_dict):
    joint_results = {}
    
    # Add additional information to the joint_results dictionary
    joint_results['Mouse ID'] = base.split("_")[2]
    joint_results['Base ID'] = base
    joint_results['Sample ID'] = samp
    joint_results['Baseline Week'] = int(base.split("_")[1].replace("D", "")) // 7
    joint_results['Sample Week'] = int(samp.split("_")[1].replace("D", "")) // 7

    for joint, (freedom_var, flexion_var, drag_factor_var, drag_score_var) in variables.items():
        base_max = Mouse_Data_means_new_base[joint + " Max"]
        samp_max = Mouse_Data_means_new_samp[joint + " Max"]
        base_min = Mouse_Data_means_new_base[joint + ' Min']
        samp_min = Mouse_Data_means_new_samp[joint + ' Min']
        base_dif = base_max - base_min
        samp_dif = samp_max - samp_min
        
        Mouse_Data_means_new_base[joint + " Diff"] = base_dif
        Mouse_Data_means_new_samp[joint + " Diff"] = samp_dif

        flexion = pcnt_change_flex(base_min, samp_min)
        freedom_movement = pcnt_change(base_dif, samp_dif)
        drag_factor = pcnt_change(base_max, samp_max)
        drag_score = drag_factor - flexion

        joint_results[freedom_var] = f"{round(float(freedom_movement.iloc[0]), 1)}%"
        joint_results[flexion_var] = f"{round(float(flexion.iloc[0]), 1)}%"
        joint_results[drag_factor_var] = f"{round(float(drag_factor.iloc[0]), 1)}%"
        joint_results[drag_score_var] = f"{round(float(drag_score.iloc[0]), 1)}%"

    joint_results.update(cog_dict)
    joint_results['center_gravity'] = f"{round(float(center_grav), 1)}%"  # Assuming center_grav is in percentage

    return Mouse_Data_means_new_base, Mouse_Data_means_new_samp, joint_results
