import os
import shutil
import subprocess
from scipy.ndimage import convolve1d
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

# replace the dataset path
work_path = 'D:\\MotionCapture'
model_name = "zeqian"
data_path = os.path.join(work_path, f"8.20_{model_name}_multiple")

# Step 1
# Put the matched rigid-body files and human-body files in one folder
# This step has already been completed in the provided dataset
folders = sorted([f for f in os.listdir(data_path) if f.startswith('Data_')])
txt_files = sorted([f for f in os.listdir(data_path) if f.startswith('Rigidbody_') and f.endswith('.txt')])
print(folders)
print(txt_files)

# Traverse through the .txt files and move them to their respective folders.
for folder, txt_file in zip(folders, txt_files):
    src_path = os.path.join(data_path, txt_file)
    dst_path = os.path.join(data_path, folder, txt_file)
    shutil.move(src_path, dst_path)
    print(f"Moved {txt_file} to {folder}")


# Step 2
# Replace the first frame for subsequent retargeting
def replace_amc_lines(amc_path, new_lines):
    with open(amc_path, 'r') as f:
        lines = f.readlines()
    lines[:63] = new_lines

    with open(amc_path, 'w') as f:
        f.writelines(lines)


frame_1_pose_path = os.path.join(work_path, f'frame_1_pose_{model_name}.txt')

# Read the first 63 lines of 'frame_1_pose.txt'
with open(frame_1_pose_path, 'r') as f:
    new_lines = f.readlines()[:63]

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    amc_path = os.path.join(folder_path, 'AMC', 'hand.amc')

    if os.path.exists(amc_path):
        replace_amc_lines(amc_path, new_lines)
        print(f"Replaced lines in {amc_path}")
    else:
        print(f"{amc_path} not found")

print("Done")


# Step 3
# Generate the BVH skeleton supported by Blender
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    amc_dir = os.path.join(folder_path, 'AMC')
    if os.path.exists(amc_dir):
        # Run a shell command in each AMC directory
        cmd = ['amc2bvh', f'../../../model_{model_name}.asf', 'Hand.amc', '-o', 'Hand.bvh']
        # cmd = ['amc2bvh', 'Hand.asf', 'Hand.amc', '-o', 'Hand.bvh']
        subprocess.run(cmd, cwd=amc_dir)
        print(f"Ran script in {amc_dir}")
    else:
        print(f"{amc_dir} not found")

print("Done")


# Step 4
# 120 -> 60 FPS, for the rigidbody_info is 60 FPS
# smooth bvh skeleton
def rolling_average(data, window_size):
    left_padding = data[:window_size-1]
    right_padding = data[-(window_size-1):]
    padded_data = np.vstack((left_padding, data, right_padding))

    window = np.ones(window_size) / window_size
    smoothed_data = convolve1d(padded_data, window, axis=0, mode='constant')[window_size-1:-window_size+1]
    return smoothed_data


def moving_average(mat, window_size):
    kernel = np.ones(window_size)/window_size
    padding = len(kernel) // 2
    padded_mat = np.pad(mat, ((padding, padding), (0, 0)), mode='edge')
    smoothed = np.convolve(padded_mat.flatten(), kernel, mode='valid')

    return smoothed.reshape(mat.shape[0], 1)


def read_and_write_bvh_file(filename):
    frames = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[417:]:
            frame_data = line.strip().split()
            if frame_data:
                frame = [float(value) for value in frame_data]
                frames.append(np.array(frame))

    frames = np.array(frames[::2])

    frames[1:, 0:3] = rolling_average(frames[1:, 0:3], 8)
    frames[1:, 6:] = rolling_average(frames[1:, 6:], 8)
    # frames[1:, 3:6] = savgol_filter(frames[1:, 3:6], 53, 3, axis=0)


    # # frames[0:3] is root transl; frames[3:6] is root orient
    #
    # for i in range(3, frames.shape[0]):
    #     # if np.linalg.norm(get_position(frames[i]) - get_position(frames[i-1])) > 0.1:
    #     #     print(i)
    #     #     print(np.linalg.norm(get_position(frames[i]) - get_position(frames[i-1])))
    #     #     # frames[i] = frames[i-1]
    #     if np.linalg.norm(frames[i][3:6] - frames[i-1][3:6]) > 18.0:
    #         frames[i+1] = frames[i]


    new_contents = lines[:417]
    for frame in frames:
        frame_line = " ".join(str(value) for value in frame)
        new_contents.append(frame_line + "\n")

    with open(filename, 'w') as file:
        file.writelines(new_contents)


for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    bvh_file = os.path.join(folder_path, 'AMC', 'Hand.bvh')
    if os.path.exists(bvh_file):
        read_and_write_bvh_file(bvh_file)
        print(f"120 FPS -> 60 FPS in {bvh_file}")
    else:
        print(f"{bvh_file} not found")
print("Done")


