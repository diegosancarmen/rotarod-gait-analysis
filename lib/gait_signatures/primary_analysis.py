import numpy as np
import pandas as pd
import deeplabcut
import os
import subprocess

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip

# https://stackoverflow.com/a/6403077
def to_seconds(timestr):
    seconds= 0
    for part in timestr.split(':'):
        seconds= seconds*60 + int(part, 10)
    return seconds

def video_trim(video_folder, vid, start, end, dst_directory):

    video_path = os.path.join(video_folder, vid + ".mp4")

    with VideoFileClip(video_path) as clip:
        duration = clip.duration
    
    trim_start = float(start if start == 0 else to_seconds(start))
    trim_end = float(duration if pd.isna(end) else to_seconds(end))
    video_name = f"{vid}_trim_{trim_start}_{trim_end}"
    output_path = os.path.join(dst_directory, f"{video_name}.mp4")
    video_extract_path = os.path.join(dst_directory, f"{vid}.mp4")
    
    # FFmpeg command to extract video only
    cmd = ['ffmpeg', '-i', video_path, '-map', '0:v', '-c:v', 'copy', video_extract_path]

    try:
        subprocess.run(cmd, check=True)
        print(f"Video extracted: {video_extract_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

    ffmpeg_extract_subclip(video_extract_path, trim_start, trim_end, targetname=output_path)

    os.remove(video_extract_path)
    
    return video_name

def deeplabcut_analyze_video(dlc_analyze_path, video_folder, vid, start, end, config):
    config_path = config
    dst_directory = os.path.join(dlc_analyze_path, vid)
    if not os.path.exists(dst_directory):
        os.mkdir(dst_directory)

    video_name = video_trim(video_folder, vid, start, end, dst_directory)
    video_path = os.path.join(dst_directory, f"{video_name}.mp4")
    
    deeplabcut.analyze_videos(config_path, [video_path], videotype='mp4', save_as_csv=True, destfolder = dst_directory, cropping= [700, 1250, 300, 900], dynamic=(True, .5, 10))
    deeplabcut.filterpredictions(config_path, [video_path], destfolder = dst_directory, filtertype='arima')
    deeplabcut.analyze_videos_converth5_to_csv(dlc_analyze_path, videotype='mp4')

    return dst_directory, video_name

def process_csv_to_dataframe_filter(csv_path):
    dataframe = pd.read_csv(csv_path)
    df = dataframe.copy()
        
    col_list = []
    for i in range(0, df.shape[1]):
        col_list.append(f"{df.iloc[1][i]}_{df.iloc[2][i][0]}")
    
    df = df.drop([0, 1, 2, ])
    df.reset_index(drop=True, inplace=True)

    df.columns = col_list
    df = df.astype(float, errors = 'raise')
    
    df1 = df[
          [
            'leftpaw_x',
            'rightpaw_x',
            'tailbase_x',
            'leftpaw_y',
            'rightpaw_y',
            'tailbase_y',
            'leftknee_x',
            'rightknee_x',
            'leftknee_y',
            'rightknee_y',
            'rodleft_y',
            'rodright_y'
          ]
        ]   
    df = df.fillna(df1.mean())

    # Extract a list of all columns containing "_l"
    l_columns = [col for col in df.columns if col.endswith("_l")]

    # Iterate through the _l columns and apply the condition to corresponding _x and _y columns
    for l_column in l_columns:
        x_column = l_column.replace("_l", "_x")
        y_column = l_column.replace("_l", "_y")

        # Convert the columns to numeric values and replace non-numeric values with NaN
        df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
        df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
        df[l_column] = pd.to_numeric(df[l_column], errors='coerce')

        # # Apply the condition
        # condition = df[l_column] > 0.1
        # df[x_column] = np.where(condition, df[x_column], np.nan)
        # df[y_column] = np.where(condition, df[y_column], np.nan)

    return df, df.shape

def add_distance_columns(df):
    df['leftpaw_x_d'] = (df['tailbase_x'] - df['leftpaw_x']).abs()
    df['rightpaw_x_d'] = (df['tailbase_x'] - df['rightpaw_x']).abs()

def add_rod_columns(df):
    df['rod_y_avg'] = ((df['rodleft_y'] + df['rodright_y']) / 2).mean()
    df['tailbase_rod_y_dist'] = df['rod_y_avg'] - df['tailbase_y']
