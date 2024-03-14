import bpy
import os
import numpy as np


# 文件路径
work_path = 'D:\\MotionCapture'
data_path = os.path.join(work_path, "8.20_zeqian_single")
file_list = sorted([f for f in os.listdir(data_path) if f.startswith('Data_')])

######################################################
# 将变换矩阵应用到物体上
def apply_transformations_to_object(obj, transformations):
    for i in range (transformations.shape[0]):
        # 将变换矩阵应用到物体上
        obj.matrix_world = np.transpose(transformations[i])
        
        start_frame_idx = -20
        # 添加关键帧以使物体动起来
        obj.keyframe_insert(data_path="location", frame=i+start_frame_idx)
        obj.keyframe_insert(data_path="rotation_euler", frame=i+start_frame_idx)
        obj.keyframe_insert(data_path="scale", frame=i+start_frame_idx)


index = 7
transformations = np.load(os.path.join(data_path ,file_list[index], 'transformation_matrix.npy'))
markers = np.load(os.path.join(data_path, file_list[index], 'rigidbody_markers.npy'))

#############################################################

#bvh_path = os.path.join(data_path ,file_list[index], 'AMC', 'Hand.bvh')  # 替换为实际的BVH文件路径
#import_scale = 0.056444  # 缩放因子
#bpy.ops.import_anim.bvh(filepath=bvh_path, filter_glob="*.bvh", global_scale=import_scale)



###########################################################

# 获取物体对象
obj = [
    bpy.data.objects.get("chair_1"),
    bpy.data.objects.get("desk_0")
]

if obj is not None:
    # 将变换矩阵应用到物体上并添加关键帧
    for chair in range(transformations.shape[1]):
        chair_transformations = transformations[:, chair:chair+1]
        apply_transformations_to_object(obj[chair], np.squeeze(chair_transformations))
else:
    print(f"Object with name '{object_name}' not found in the scene.")
    


###########################################################
## 设置每一帧的位置并插入关键帧
#for frame_index in range(markers.shape[0]):
#    for chair in range(markers.shape[1]):
#        for marker_index, marker_data in enumerate(markers[frame_index][chair]):
#            if not np.isnan(marker_data).all():
#                x, y, z = marker_data
#                cube = bpy.context.scene.objects[f"Cube_{chair}_{marker_index}"]
#                cube.location = (x, y, z)
#                cube.keyframe_insert(data_path="location", frame=frame_index+1)  # 插入关键帧
                