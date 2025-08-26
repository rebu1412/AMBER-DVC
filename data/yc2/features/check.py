import os
import numpy as np

feature_folder = "/workspace/baseline/data/yc2/features/resnet_bn"
video_files = os.listdir(feature_folder)
error_files = []

for idx, video_file in enumerate(video_files):
    try:
        feats = np.load(os.path.join(feature_folder, video_file))
        # { }
        if feats.size == 0:
            print(f"File {video_file} is empty (no data).")
            error_files.append(video_file)
        # NaN or Inf
        elif np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
            print(f"File {video_file} contains invalid values (NaN or Inf).")
            error_files.append(video_file)
        # all zeros
        elif np.all(feats == 0):
            print(f"File {video_file} contains all zeros.")
            error_files.append(video_file)
        else:
            print(f"File {video_file} is valid.")
    except Exception as e:
        print(f"Error reading file {video_file}: {str(e)}")
        error_files.append(video_file)

if error_files:
    print("List of problematic files:", error_files)
else:
    print("No problematic files found.")

with open("error_files.txt", "w") as f:
    for file in error_files:
        f.write(file + "\n")