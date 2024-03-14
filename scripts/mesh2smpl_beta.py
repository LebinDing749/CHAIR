
import os
import trimesh
import smplx
import torch
import numpy as np


def create_color_array(vlen, color):
    color_arr = np.zeros((vlen, 3), dtype=np.uint8)
    color_arr[:, 0:3] = color
    return color_arr


def vertices_to_params(initial_vertices, learning_rate, iterations, initial_parameters, is_beta_already):

    print(initial_vertices.shape)
    mocap_vertices = torch.Tensor(initial_vertices).to(device)

    # define initial t, r, p and beta
    t = initial_parameters['transl'].clone().detach().to(device)
    r = initial_parameters['global_orient'].clone().detach().to(device)
    p = initial_parameters['body_pose'].clone().detach().to(device)
    beta = initial_parameters['betas'].clone().detach().to(device)  # 身体比例参数
    t.requires_grad_(True)
    r.requires_grad_(True)
    p.requires_grad_(True)
    beta.requires_grad_(True)

    params = {
        'transl': t.detach(),
        'global_orient': r.detach(),
        'body_pose': p.detach(),
        'betas': beta.detach(),
    }

    output = smplx_model(**params)
    vertices = output.vertices.squeeze()

    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis()) # add a axis
    S.add_geometry(trimesh.Trimesh(vertices=initial_vertices, faces=smplx_model.faces,
                    vertex_colors=create_color_array(len(initial_vertices), [255, 0, 0]))) # add the mesh M, the color is red
    S.add_geometry(trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=smplx_model.faces,
                    vertex_colors=create_color_array(len(mocap_vertices), [0, 0, 255])))
    S.show()

    if is_beta_already:
        optimizer = torch.optim.Adam([t, r, p], lr=learning_rate)
    else:
        # 第一次拟合, 需要确定betas
        optimizer = torch.optim.Adam([t, r, beta], lr=learning_rate)

    for i in range(iterations):
        # 生成当前的人体模型
        output = smplx_model(global_orient=r, body_pose=p, betas=beta, transl=t)
        current_vertices = output.vertices.squeeze()

        # 计算均方误差损失
        loss = torch.nn.functional.mse_loss(current_vertices, mocap_vertices, reduction='sum')

        # 清除梯度并进行反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印当前迭代的损失
        print(f"Iteration {i}/{iterations}, Loss: {loss.item()}")

    optimized_params = {
        'transl': t.detach(),
        'global_orient': r.detach(),
        'body_pose': p.detach(),
        'betas': beta.detach(),
    }

    output = smplx_model(return_verts=True, **optimized_params)
    optimized_vertices = output.vertices.detach().squeeze()

    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis()) # add a axis
    # S.add_geometry(trimesh.Trimesh(vertices=initial_vertices, faces=smplx_model.faces,
    #                 vertex_colors=create_color_array(len(initial_vertices), [255, 0, 0]))) # add the mesh M, the color is red
    S.add_geometry(trimesh.Trimesh(vertices=optimized_vertices.cpu().numpy(), faces=smplx_model.faces,
                    vertex_colors=create_color_array(len(optimized_vertices), [0, 0, 255])))
    S.show()

    return optimized_params


if __name__ == '__main__':
    device = torch.device('cuda')
    smplx_model = smplx.create(
        "C:/Users/lenovo/Downloads/models_smplx_v1_1_2/models", model_type='smplx',
        gender='female', ext='npz',
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
    ).to(device)

    work_path = 'D:\\MotionCapture'
    data_path = os.path.join(work_path, "8.26_leyao")
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        vertices_path = os.path.join(folder_path, 'AMC', 'vertices.npy')

        if os.path.exists(vertices_path):
            vertices = np.load(vertices_path)

            transl = []
            global_orient = []
            body_pose = []
            betas = []
            # initial parameters
            smplx_parameters = {
                'transl': torch.zeros(1, 3),
                'global_orient': torch.zeros(1, 3),
                'body_pose': torch.zeros(1, 63),
                'betas': torch.zeros(1, 10),
            }

            def add_parameters(smplx_parameters):
                transl.append(np.array(smplx_parameters['transl'].cpu().numpy()))
                global_orient.append(np.array(smplx_parameters['global_orient'].cpu().numpy()))
                body_pose.append(np.array(smplx_parameters['body_pose'].cpu().numpy()))
                betas.append(np.array(smplx_parameters['betas'].cpu().numpy()))

            # frame[0], 确定betas 参数
            smplx_parameters = vertices_to_params(vertices[0], 0.03, 500, smplx_parameters, False)
            add_parameters(smplx_parameters)

            # frame[1]
            smplx_parameters = vertices_to_params(vertices[1], 0.3, 1800, smplx_parameters, True)
            add_parameters(smplx_parameters)

            # frame[2:]
            for i in range(2, vertices.shape[0]-vertices.shape[0] + 3):
                learning_rate = 0.003
                iterations = 200
                smplx_parameters = vertices_to_params(vertices[i], learning_rate, iterations, smplx_parameters, True)
                add_parameters(smplx_parameters)

            np.save(os.path.join(folder_path, "AMC", "transl.npy"), transl)
            np.save(os.path.join(folder_path, "AMC", "global_orient.npy"), global_orient)
            np.save(os.path.join(folder_path, "AMC", "body_pose.npy"), body_pose)
            np.save(os.path.join(folder_path, "AMC", "betas.npy"), betas)
            print(f"transform vertices to smplx parameters in {vertices_path}")
        else:
            print(f"{vertices_path} not found")

    print("Done")



