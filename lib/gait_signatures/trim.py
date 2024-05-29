from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd

video_folder = "/home/coeguest/hdelacruz/DeepLabCut/Experiment2"
output_folder = "/home/coeguest/hdelacruz/DeepLabCut/test"
compare_id = pd.read_csv("/home/coeguest/hdelacruz/DeepLabCut/test/comparison_id.csv")

import os
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip

video_folder = "/home/coeguest/hdelacruz/DeepLabCut/Experiment2"
output_folder = "/home/coeguest/hdelacruz/DeepLabCut/test"
compare_id = pd.read_csv("/home/coeguest/hdelacruz/DeepLabCut/test/comparison_id.csv")

# https://stackoverflow.com/a/6403077
def to_seconds(timestr):
    seconds= 0
    for part in timestr.split(':'):
        seconds= seconds*60 + int(part, 10)
    return seconds

compare_id.columns = ["base", "base_start", "base_end", "samp", "samp_start", "samp_end"]

# Replace NaN with 0 for start times
compare_id[['base_start', 'samp_start']] = compare_id[['base_start', 'samp_start']].fillna(0)

# Iterate over the rows of the DataFrame
for index, row in compare_id.iterrows():
    base, base_start, base_end = row["base"], row["base_start"], row["base_end"]
    samp, samp_start, samp_end = row["samp"], row["samp_start"], row["samp_end"]

    # Construct file paths
    base_vid_path = os.path.join(video_folder, f"{base}.mp4")
    samp_vid_path = os.path.join(video_folder, f"{samp}.mp4")

    # Get video durations for base and sample videos
    with VideoFileClip(base_vid_path) as base_clip:
        base_duration = base_clip.duration
    with VideoFileClip(samp_vid_path) as samp_clip:
        samp_duration = samp_clip.duration
        
    # Convert start and end times to float and handle NaN values
    base_trim_start = float(base_start if base_start == 0 else to_seconds(base_start))
    base_trim_end = float(base_duration if pd.isna(base_end) else to_seconds(base_end))
    samp_trim_start = float(samp_start if samp_start == 0 else to_seconds(samp_start))
    samp_trim_end = float(samp_duration if pd.isna(samp_end) else to_seconds(samp_end))

    # Construct output paths
    base_output_path = os.path.join(output_folder, f"{base}_trim_{base_trim_start}_{base_trim_end}.mp4")
    samp_output_path = os.path.join(output_folder, f"{samp}_trim_{samp_trim_start}_{samp_trim_end}.mp4")

    # Extract subclips
    ffmpeg_extract_subclip(base_vid_path, base_trim_start, base_trim_end, targetname=base_output_path)
    ffmpeg_extract_subclip(samp_vid_path, samp_trim_start, samp_trim_end, targetname=samp_output_path)


# process_videos(compare_id, video_folder, output_folder)
