import os
from datetime import datetime
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve1d


def axis_transform(transl_frame, orient_frame, markers_frame):
    for rot in orient_frame:
        y = rot[1]
        rot[1] = rot[2]
        rot[2] = y
        rot[1] = -rot[1]

    for pos in transl_frame:
        y = pos[1]
        pos[1] = pos[2]
        pos[2] = y
        pos[1] = -pos[1]

    for markers in markers_frame:
        for pos in markers:
            y = pos[1]
            pos[1] = pos[2]
            pos[2] = y
            pos[1] = -pos[1]

    return transl_frame, orient_frame, markers_frame


def get_timestamp(line):
    timestamp_parts = line.split("_")
    year = int(timestamp_parts[0])
    month = int(timestamp_parts[1])
    day = int(timestamp_parts[2])
    hour = int(timestamp_parts[3])
    minute = int(timestamp_parts[4])
    second = int(timestamp_parts[5])
    microsecond = int(timestamp_parts[6]) * 1000
    line_time = datetime(year, month, day, hour, minute, second, microsecond)

    return line_time


def get_rigidbody_info(file_path, start_time, frames):

    transl = []
    orient = []
    markers = []

    with open(file_path, 'r') as f_in:
        lines = f_in.readlines()

    start_line = 0
    for i, line in enumerate(lines):
        if line.startswith("2023_"):
            line_time = get_timestamp(line)
            time_difference = (line_time - start_time).total_seconds()

            if abs(time_difference) < 0.02:    # less than 20 ms
                start_line = i
                break

    num_chairs = int(lines[start_line+1])
    print('chairs_num: ' + str(num_chairs))
    start_timestamp = lines[start_line]
    end_timestamp = lines[start_line + (frames - 1) * (2 + 2 * num_chairs)]
    total_time = get_timestamp(end_timestamp) - get_timestamp(start_timestamp)
    FPS = frames // total_time.total_seconds()
    print('FPS: ' + str(FPS))

    for i in range(start_line, start_line+(frames-1)*(2+2*num_chairs) + 1, 2+2*num_chairs):

        # timestamp = lines[i].strip()

        orient_frame = []
        transl_frame = []
        markers_frame = []
        for j in range(num_chairs):
            body_data = lines[i + 2 + 2*j].strip().split('--')
            name = body_data[0].split(':')[1]
            exist = int(body_data[1].split(':')[1])
            pos = body_data[2].split(':')[1].replace('(', '').replace(')', '').split(',')
            pos = np.array([float(p) for p in pos])
            rot = body_data[3].split(':')[1].replace('(', '').replace(')', '').split(',')
            rot = np.array([float(r) for r in rot])
            orient_frame.append(np.array(rot))
            transl_frame.append(np.array(pos))

        for k in range(num_chairs):
            markers_info = lines[i+3+2*k].split()
            markers_coordinates = []
            for marker in markers_info:
                marker_coords = marker.split(':')[-1].split(',')
                marker_coords = np.array([float(coord) for coord in marker_coords])
                markers_coordinates.append(marker_coords)

            markers_frame.append(np.array(markers_coordinates))

        # coordinate axis transformation
        transl_frame, orient_frame, markers_frame = axis_transform(transl_frame, orient_frame, markers_frame)

        transl.append(transl_frame)
        orient.append(orient_frame)
        markers.append(markers_frame)

    return np.array(transl), np.array(orient), list_to_matrix(markers), FPS


def list_to_matrix(input_list):
    # Find the maximum length of the second dimension, the number of markers on a chair
    max_markers_num = max(len(chair) for frame in input_list for chair in frame)

    # Overwrite missing elements
    for frame in range(len(input_list)):
        for chair in range(len(input_list[frame])):
            if len(input_list[frame][chair]) < max_markers_num:
                input_list[frame][chair] = input_list[frame-1][chair]

    # Create a 3D array
    # [frames, chairs_num, markers_num, xyz]
    matrix = np.full((len(input_list), len(input_list[0]), max_markers_num, 3), np.nan)

    # Fill the matrix with data from the input list
    for i, frame_points in enumerate(input_list):
        for k, chair_points in enumerate(frame_points):
            for j, row in enumerate(chair_points):
                matrix[i, k, j, :len(row)] = row

    return matrix


def rolling_average(data, window_size):
    left_padding = data[:window_size-1]
    right_padding = data[-(window_size-1):]
    padded_data = np.vstack((left_padding, data, right_padding))

    window = np.ones(window_size) / window_size
    smoothed_data = convolve1d(padded_data, window, axis=0, mode='constant')[window_size-1:-window_size+1]
    return smoothed_data


def quaternion_average(data, window_size):
    def quaternion_mean(quaternions):
        sum_q = np.sum(quaternions, axis=0)
        # Normalize the sum to get the mean quaternion
        mean_q = sum_q / np.linalg.norm(sum_q)
        return mean_q

    # [frames, chairs, 4]
    num_frames, num_chairs, num_features = data.shape

    smoothed_data = data
    for frame in range(num_frames - window_size):
        for chair in range(num_chairs):
            chair_orient = smoothed_data[frame:frame+window_size, chair:chair+1]
            smoothed_data[frame][chair] = quaternion_mean(np.squeeze(chair_orient))

    return smoothed_data


# Overwrite incorrect frames
def detect_and_fill_outliers(transl, orient, threshold):
    filled_transl = transl
    filled_orient = orient

    for frame in range(1, filled_transl.shape[0]):
        for chair in range(1, filled_transl.shape[1]):
            if np.any(abs(filled_transl[frame][chair] - filled_transl[frame-1][chair]) > threshold):
                filled_transl[frame][chair] = filled_transl[frame-1][chair]
                filled_orient[frame][chair] = filled_orient[frame-1][chair]

    return filled_transl, filled_orient


def interpolate_data(original_data, original_fps, target_fps, frames):
    original_interval = 1.0 / original_fps
    target_interval = 1.0 / target_fps

    num_frames_original = original_data.shape[0]
    num_frames_target = int(target_fps / original_fps * num_frames_original)
    interpolated_data = np.zeros((num_frames_target,) + original_data.shape[1:], dtype=original_data.dtype)

    for i in range(num_frames_target):
        target_time = i * target_interval
        original_index_float = target_time / original_interval
        original_index_1 = int(original_index_float)
        original_index_2 = min(original_index_1 + 1, num_frames_original - 1)

        t = original_index_float - original_index_1
        interpolated_data[i] = (1 - t) * original_data[original_index_1] + t * original_data[original_index_2]

    return interpolated_data[0:frames]


def save_rigidbody_info(save_folder, file_path, start_time, frames):
    transl, orient, markers, FPS = get_rigidbody_info(file_path, start_time, frames)

    print("ok")
    # remove erroneous data
    threshold = 0.15
    transl, orient = detect_and_fill_outliers(transl, orient, threshold)
    threshold = 0.08
    orient, transl = detect_and_fill_outliers(orient, transl, threshold)

    # smooth
    window_size = 25
    transl = rolling_average(transl, window_size)
    # orient = quaternion_average(orient, window_size)
    orient = rolling_average(orient, window_size)

    # # Interpolate frames, if necessary
    # original_fps = FPS
    # target_fps = 60.0
    # transl = interpolate_data(transl, original_fps, target_fps, frames)
    # orient = interpolate_data(orient, original_fps, target_fps, frames)
    # markers = interpolate_data(markers, original_fps, target_fps, frames)

    combined_matrix = np.zeros((transl.shape[0], transl.shape[1], 4, 4))
    for frame in range(transl.shape[0]):
        for chair in range(transl.shape[1]):
            combined_matrix[frame][chair][:3, :3] = Rotation.from_quat(orient[frame][chair]).as_matrix()
    combined_matrix[:, :, :3, 3] = transl
    combined_matrix[:, :, 3, 3] = 1

    print(combined_matrix.shape)
    np.save(os.path.join(save_folder, 'transformation_matrix'), combined_matrix)
    print(markers.shape)
    np.save(os.path.join(save_folder, 'rigidbody_markers'), markers)


if __name__ == "__main__":

    # replace the dataset path
    work_path = 'D:\\MotionCapture'
    data_path = os.path.join(work_path, "8.20_zeqian_multiple")

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        amc_dir = os.path.join(folder_path, 'AMC')
        amc_file = os.path.join(amc_dir, "Hand.amc")

        if os.path.exists(amc_dir): # 如果是数据文件夹
            print(f'processing {amc_dir}')
            # retrieve the timestamp when the skeleton starts moving
            with open(amc_file, 'r') as f:
                frames = (len(f.readlines()) - 3) // 120
                print("frames: " + str(frames))

            start_time = datetime.strptime(folder, "Data_%Y-%m-%d %H_%M_%S_%f")

            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            if len(txt_files) == 1:
                txt_file_path = os.path.join(folder_path, txt_files[0])
                save_rigidbody_info(folder_path, txt_file_path, start_time, frames)
                print(f"Found TXT file: {txt_file_path}")
            elif len(txt_files) == 0:
                print(f"No TXT file found in {amc_dir}")
            else:
                print(f"Multiple TXT files found in {amc_dir}")

        else:
            print(f"{amc_dir} not found")




