#
# This script is used to get 'transl' and 'global_orient'.
# Before that, the betas[10], body_pose[63], and two sets of joint or mesh_vertices coordinates are already available
# The process from initial transl and global_orient to the true value approximated as a 3D rigidbody transformation
#

import math
import os
import pickle
import numpy as np
import torch
import trimesh
import smplx
from scipy.spatial.transform import Rotation
from scipy.linalg import svd


def create_color_array(vlen, color):
    color_arr = np.zeros((vlen, 3), dtype=np.uint8)
    color_arr[:, 0:3] = color
    return color_arr


def show_vertices(origin_vertices, fit_vertices):
    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis())  # add a axis
    S.add_geometry(trimesh.Trimesh(vertices=origin_vertices, faces=faces,
                                   vertex_colors=create_color_array(len(origin_vertices), [0, 0, 255])))
    S.add_geometry(trimesh.Trimesh(vertices=fit_vertices, faces=faces,
                                   vertex_colors=create_color_array(len(fit_vertices), [255, 0, 0])))
    S.show()


def show_single_vertices(origin_vertices):
    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis())  # add a axis
    S.add_geometry(trimesh.Trimesh(vertices=origin_vertices, faces=faces,
                                   vertex_colors=create_color_array(len(origin_vertices), [0, 0, 255])))
    S.show()


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA), BB)
    U, S, Vt = svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)

    t = -np.matmul(R, centroid_A) + centroid_B
    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
    return R, t


def f(T, trans, orient, pelvis):
    ## TODO: your code
    R_cw = T[:3, :3]
    t_cw = T[3:, :3]

    # get new global_orient
    R_c = Rotation.from_rotvec(orient).as_matrix()
    R_w = np.dot(R_cw, R_c)
    R_w = Rotation.from_matrix(R_w).as_rotvec()

    # pelvis_new = np.dot(R_cw, pelvis) + t_cw
    # t_w = np.dot(R_c, trans - pelvis) + pelvis_new

    # get pelvis not apply trans
    pelvis = pelvis - trans
    t_w = np.dot(R_cw, pelvis + trans) + t_cw - pelvis

    tt = np.array(t_w, dtype=np.float32)
    rr = np.array(R_w, dtype=np.float32)

    return tt, rr


# the dataset path
work_path = 'D:\\MotionCapture'
data_path = os.path.join(work_path, "8.26_peiqi")
# notice the gender
smplx_model = smplx.create(
    "C:/Users/lenovo/Downloads/models_smplx_v1_1_2/models", model_type='smplx',
    gender='male', ext='npz',
    num_pca_comps=10,
    create_global_orient=True,
    create_body_pose=True,
    create_betas=True,
    create_left_hand_pose=True,
    create_right_hand_pose=True,
    create_expression=True,
    create_jaw_pose=True,
    create_leye_pose=True,
    create_reye_pose=True,
    create_transl=True,
    batch_size=1,
).to(device='cpu')
faces = smplx_model.faces

for motion_idx in range(len(os.listdir(data_path))):
    # processing part motion files
    # for motion_idx in range(7, 8):

    folder_path = os.path.join(data_path, os.listdir(data_path)[motion_idx])
    part_vertices_path = os.path.join(folder_path, 'AMC', 'part_vertices.npy')

    if os.path.exists(part_vertices_path):
        part_vertices = np.load(part_vertices_path)
        betas = np.load(os.path.join(data_path, os.listdir(data_path)[0], "AMC", 'betas.npy'))
        body_pose = np.load(os.path.join(folder_path, "AMC", 'body_pose.npy'))

        # print(part_vertices.shape[0])
        # obj_transl = np.load(os.path.join(folder_path, 'obj_transl.npy'))
        # # obj_orient = np.load(os.path.join(folder_path, 'obj_orient.npy'))
        # # obj_id = np.load(os.path.join(folder_path, 'obj_id.npy'))
        # print(obj_transl.shape[0])

        # print(obj_transl)
        # print(obj_orient)
        # print(obj_id)
        # transl = np.load(os.path.join(folder_path, "AMC", 'transl.npy'))
        # orient = np.load(os.path.join(folder_path, "AMC", 'orient.npy'))
        # print(transl)
        # print(orient)

        # for i in range(pose.shape[0]):
        #     transl[i] = [-transl[i][2], transl[i][0], -transl[i][1]]
        #     # orient[i] = [-orient[i][2], orient[i][0], -orient[i][1]]

        # print(transl)
        # print(orient)
        transl = np.zeros((body_pose.shape[0], 3))
        global_orient = np.zeros((body_pose.shape[0], 3))
        frames_len = body_pose.shape[0]

        for i in range(frames_len):
            params = {
                'transl': torch.zeros(1, 3),
                'global_orient': torch.zeros(1, 3),
                # 'transl': torch.Tensor(transl[i]).reshape(1, -1),
                # 'global_orient': torch.Tensor(orient[i]).reshape(1, -1),
                'body_pose': torch.Tensor(body_pose[i]).reshape(1, -1),
                'betas': torch.Tensor(betas).reshape(1, -1)
            }

            output = smplx_model(return_verts=True, **params)
            pelvis = output.joints.detach().squeeze(0).numpy()[0]
            part_vertices_param = output.vertices.detach().squeeze(0).numpy()

            A = part_vertices_param[::1000]
            B = part_vertices[i]
            # B = part_vertices[i][::1000]
            # print(part_vertices.shape)
            # print(A.shape)
            # print(B.shape)

            R, t = rigid_transform_3D(A, B)

            T = np.zeros((4, 4))
            T[:3, :3] = R
            T[3:, :3] = t
            T[3:, 3:] = 1

            transl[i], global_orient[i] = f(T, np.array([0, 0, 0]), np.array([0, 0, 0]), pelvis)

            # params = {
            #     # 'transl': torch.zeros(1, 3),
            #     # 'global_orient': torch.zeros(1, 3),
            #     'transl': torch.Tensor(transl[i]).reshape(1, -1),
            #     'global_orient': torch.Tensor(global_orient[i]).reshape(1, -1),
            #     'body_pose': torch.Tensor(body_pose[i]).reshape(1, -1),
            #     'betas': torch.Tensor(betas).reshape(1, -1)
            # }
            # output = smplx_model(return_verts=True, **params)
            # show_vertices(part_vertices[i], output.vertices.detach().squeeze(0).numpy())
            print(f"motion:{motion_idx}/39, frames:{i}/{frames_len - 1}")

        np.save(os.path.join(folder_path, "AMC", "transl.npy"), transl)
        np.save(os.path.join(folder_path, "AMC", "global_orient.npy"), global_orient)
        np.save(os.path.join(folder_path, "AMC", "betas"), betas)



#
# examine parameters
#
# for i in range(len(os.listdir(data_path))):
#
#     folder_path = os.path.join(data_path, os.listdir(data_path)[i])
#     transl_path = os.path.join(folder_path, 'AMC', 'transl.npy')
#
#     if os.path.exists(transl_path):
#         transl = np.load(os.path.join(folder_path, "AMC", 'transl.npy'))
#         global_orient = np.load(os.path.join(folder_path, "AMC", 'global_orient.npy'))
#         betas = np.load(os.path.join(folder_path, "AMC", 'betas.npy'))
#         body_pose = np.load(os.path.join(folder_path, "AMC", 'body_pose.npy'))
#
#         print(transl.shape)
#         print(global_orient.shape)
#
#         for j in range(0, body_pose.shape[0], 600):
#             params = {
#                 'transl': torch.Tensor(transl[j]).reshape(1, -1),
#                 'global_orient': torch.Tensor(global_orient[j]).reshape(1, -1),
#                 'body_pose': torch.Tensor(body_pose[j]).reshape(1, -1),
#                 'betas': torch.Tensor(betas).reshape(1, -1)
#             }
#             output = smplx_model(return_verts=True, **params)
#             show_single_vertices(output.vertices.detach().squeeze(0).numpy())
