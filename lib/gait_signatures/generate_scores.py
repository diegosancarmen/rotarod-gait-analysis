# !pip install DLC2Kinematics

import dlc2kinematics
import numpy as np
from scipy.signal import argrelextrema

def add_extrema_columns(df, column, n):
    min_column = f'{column}_min'
    max_column = f'{column}_max'

    min_indices = argrelextrema(df[column].values, np.less_equal, order=n)[0]
    max_indices = argrelextrema(df[column].values, np.greater_equal, order=n)[0]

    df[min_column] = df.iloc[min_indices][column]
    df[max_column] = df.iloc[max_indices][column]

def compute_and_add_joint_extrema(primary_stats_df, path, n):
    path_h5 = str(path + ".h5")
    df, bodyparts, scorer = dlc2kinematics.load_data(path_h5)

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
