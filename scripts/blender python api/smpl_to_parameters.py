import bpy
import numpy as np
import os

joint_names = SMPLX_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye',
    'right_eye',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
    ]

# 获得网格顶点实际世界坐标
def get_evalualted_vertices(obj) -> np.ndarray:
    # 获得应用各种modifier之后的网格，例如骨骼驱动、形变、细分
    object_eval = obj.evaluated_get(depsgraph=bpy.context.evaluated_depsgraph_get())
    # 再乘上 matrix_world 矩阵，得到世界坐标系下的顶点坐标
    vertices = np.array([obj.matrix_world @ vert.co for vert in object_eval.data.vertices], dtype=float)   
    return vertices


def get_animation_frame_range():
    armature_object = bpy.data.objects['Hand']  # 替换为实际的骨骼对象名称
    action = armature_object.animation_data.action
    
    if action is not None:
        start_frame = int(action.frame_range[0])
        end_frame = int(action.frame_range[1])
        return start_frame, end_frame
    else:
        return None


def save_vertices(file_path, index):
    part_vertices = []
        # frame 1 is initial pose
    for frame in range(2, end_frame+1):
        # 设置当前帧
        bpy.context.scene.frame_set(frame)
        # 获取顶点坐标
        vertices = get_evalualted_vertices(mesh_object)[::1000]
#        vertices = vertices[::1000]
        part_vertices.append(np.array(vertices))
        

    # 定义文件路径
    save_path = os.path.join(file_path, "part_vertices.npy")
    # 将顶点坐标保存到矩阵文件
    np.save(save_path, part_vertices)


def get_pelvis_and_pose():
    
    def rodrigues_from_pose(armature, bone_name):
        # Use quaternion mode for all bone rotations
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

        quat = armature.pose.bones[bone_name].rotation_quaternion
        (axis, angle) = quat.to_axis_angle()
        rodrigues = axis
        rodrigues.normalize()
        rodrigues = rodrigues * angle
        return rodrigues
    pel = armature_object.pose.bones['pelvis'].location
    pel = np.array(pel)
    pos = []
    for index in range(1, 22):
        joint_name = joint_names[index]
        joint_pose = rodrigues_from_pose(armature_object, joint_name)
        pos.append(np.array(joint_pose))

    pos = np.array(pos)
    return pel, pos
        


start_frame, end_frame = get_animation_frame_range()
print(start_frame, end_frame)

#################################################################
#################################################################
# 获取 SMPL 模型的网格对象
mesh_object = bpy.data.objects['SMPLX-mesh-male']
armature_object = bpy.data.objects['SMPLX-male']
# 文件路径
work_path = 'D:\\MotionCapture'
data_path = os.path.join(work_path, "8.20_zeqian_single")
file_list = sorted([f for f in os.listdir(data_path) if f.startswith('Data_')])

index = 0
obj_id = np.array([
    "chair_0"
    ])
##################################################################
##################################################################

# save part_vertices
save_vertices(os.path.join(data_path, file_list[index], "AMC"), index)

## save betas[10]
#betas = []
#for i in range(0, 10):
#    key_name = f"Shape{'%0.3d' % i}"
#    key_block = mesh_object.data.shape_keys.key_blocks.get(key_name)
#    betas.append(key_block.value)
#    
#betas = np.array(betas).reshape(1, 10)
#np.save(os.path.join(data_path, file_list[index], "AMC", "betas.npy"), betas)

# save body_pose[32] pelvis and rigid_body_transformation
obj_transl = []
obj_orient = []
pelvis = []
body_pose = []

                    
for frame in range(2, end_frame+1):
        # 设置当前帧
    bpy.context.scene.frame_set(frame)
    t, p = get_pelvis_and_pose()
    pelvis.append(t)
    body_pose.append(p)
    
    obj_transl_frame = []
    obj_orient_frame = []
    for i in range(0, obj_id.shape[0]):
        t = bpy.data.objects[obj_id[i]].location
        r = bpy.data.objects[obj_id[i]].rotation_euler
        
        obj_transl_frame.append(np.array(t))
        obj_orient_frame.append(np.array(r))
    
    obj_transl.append(np.array(obj_transl_frame))
    obj_orient.append(np.array(obj_orient_frame))


np.save(os.path.join(data_path, file_list[index], "AMC", "pelvis.npy"), pelvis)
np.save(os.path.join(data_path, file_list[index], "AMC", "body_pose.npy"), body_pose)
np.save(os.path.join(data_path, file_list[index], "obj_transl.npy"), obj_transl)
np.save(os.path.join(data_path, file_list[index], "obj_orient.npy"), obj_orient)
np.save(os.path.join(data_path, file_list[index], "obj_id.npy"), obj_id)




