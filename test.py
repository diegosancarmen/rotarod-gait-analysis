import os
import csv

output_folder = "/home/coeguest/hdelacruz/DeepLabCut/Experiment2_output"

row_headers = ["Mouse ID,,,,Paw-Tail Distance,,,Ankle Joint Angle,,, CoG Pre-Score, FOM Ankle Joint Angle,,,,,,Left,,,,Right,,,,", "Name, Induction Week, File ID#, Frame #, H_L_TP Mean, H_R_TP Mean,, JA_L Mean, JA_R Mean,, , JA_L Max, JA_L Min, JA_L Diff, JA_R Max, JA_R Min, JA_R Diff, % Change (FOM), Flexion Decline, Drag Factor, Drag Score, % Change (FOM), Flexion Decline, Drag Factor, Drag Score"]

# Path to save the CSV file
stats_path = os.path.join(output_folder, 'gait_metrics.csv')
    
# Create the CSV file and write the header rows
with open(stats_path, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for header_row in row_headers:
        csv_row = header_row.split(",")
        writer.writerow(csv_row)

