#
# This script is used to get smpl-x parameters(including transl, global_orient, body_pose and betas) from mesh vertices
# Before that, the smpl-x mesh vertices per frame are already available
#
#

import os
import trimesh
import smplx
import torch
import numpy as np


def create_color_array(vlen, color):
    color_arr = np.zeros((vlen, 3), dtype=np.uint8)
    color_arr[:, 0:3] = color
    return color_arr


def vertices_to_params(initial_vertices):

    mocap_vertices = torch.Tensor(initial_vertices).to(device)
    # define initial t、r、p 和 beta
    t = torch.zeros(1, 3).to(device)
    r = torch.zeros(1, 3).to(device)
    # p = torch.zeros(1, 63).to(device)
    p = torch.tensor([-0.2884081304073334, 1.4910929203033447, 0.5045854449272156, -0.9157024621963501, 0.14176025986671448, -0.07176809757947922, -1.0340226888656616, 0.13636811077594757, -0.1251145601272583, 0.42132705450057983, 0.0679553896188736, -0.09287300705909729, 1.4544965028762817, 1.8766237985801126e-07, 8.332470713412476e-09, 1.5986722707748413, 9.697287417509415e-09, 5.857107243656401e-09, 0.0779607743024826, 0.06919576972723007, -0.0318368598818779, 0.09033574908971786, -1.1608290861886417e-07, -7.170670812683966e-08, 0.08882349729537964, 9.603913042610657e-08, 8.000757389936552e-08, 0.1400921493768692, 0.05188397690653801, -0.038521669805049896, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.31440016627311707, -0.30507171154022217, -0.3177894651889801, -0.003436826169490814, -0.08531583845615387, -0.08047245442867279, 0.027578840032219887, 0.2575622498989105, -0.21214227378368378, -0.2175833135843277, -0.22000376880168915, -0.08699605613946915, -0.45316076278686523, -1.5004446506500244, -0.3295031189918518, 0.12980438768863678, 0.58123779296875, 1.2242382764816284, -0.0429435595870018, -2.129739761352539, 1.2296142578125, 0.003339876653626561, -0.16867807507514954, 0.09738173335790634, 0.2077886462211609, -0.0031409452203661203, 0.0018135353457182646, -1.484020709991455, -0.02243371680378914, 0.012951340526342392, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.031853266060352325, -0.04384808614850044, 0.9536815285682678, 0.16388562321662903, 0.0680784285068512, -0.7273507714271545, 0.09870555996894836, -0.0014540235279127955, -0.5326283574104309, -0.03185327723622322, -0.04384798929095268, 0.9536823630332947, 0.0005217281868681312, 0.048937778919935226, -0.1295851618051529, -0.003082728013396263, 5.3804706112714484e-05, -0.11774257570505142, -0.031853433698415756, -0.04384783282876015, 0.9536820650100708, -0.05584333464503288, -5.1563802117016166e-05, -0.11692961305379868, -0.05069923400878906, 0.0007634701323695481, -0.1063055470585823, -0.0318535752594471, -0.04384821653366089, 0.9536823034286499, -0.028366271406412125, 0.024453749880194664, -0.12634441256523132, -0.027492031455039978, 0.00041743804467841983, -0.1145266592502594, 0.023588180541992188, 0.01145669724792242, 0.8257052302360535, -0.024011874571442604, 0.19544538855552673, 0.06674138456583023, 0.023495551198720932, 0.11724168062210083, -0.036866698414087296, 0.043793678283691406, 0.25931671261787415, 0.3401681184768677, 0.10353561490774155, -0.020362647250294685, 0.5258567333221436, 0.06662020832300186, 0.000798042572569102, 0.3596489429473877, 0.04379358887672424, 0.2593170702457428, 0.3401685953140259, -0.024378899484872818, 0.08494529128074646, 0.3357110619544983, -0.006702639162540436, 0.0005532476934604347, 0.26098164916038513, 0.04379377141594887, 0.25931623578071594, 0.340167760848999, -0.1927379071712494, 0.3003149628639221, 0.28632599115371704, -0.11111752688884735, 0.005895573645830154, 0.2350332885980606, 0.04379340633749962, 0.25931641459465027, 0.34016770124435425, -0.11206773668527603, 0.19281038641929626, 0.3212360739707947, -0.06009862199425697, 0.0035962173715233803, 0.2536648213863373, 0.15441842377185822, 0.2746404707431793, 0.5029982924461365, 0.01978885754942894, -0.015899134799838066, 0.03618425503373146, 0.0021922101732343435, -0.011245795525610447, 0.0035006708931177855]).to(device)
    print(p.shape)
    p = p[3:66]
    p = p.reshape(1, 63)

    beta = torch.zeros(1, 10).to(device)
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
    # S.add_geometry(trimesh.Trimesh(vertices=initial_vertices, faces=smplx_model.faces,
    #                 vertex_colors=create_color_array(len(initial_vertices), [255, 0, 0]))) # add the mesh M, the color is red
    S.add_geometry(trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=smplx_model.faces,
                    vertex_colors=create_color_array(len(mocap_vertices), [0, 0, 255])))
    S.show()

    optimizer = torch.optim.Adam([t, r, beta], lr=learning_rate)

    for i in range(iterations):
        # generater current smpl-x model
        output = smplx_model(global_orient=r, body_pose=p, betas=beta, transl=t)
        current_vertices = output.vertices.squeeze()

        # calculate mes_loss
        # choosing part vertices is feasible and can expedite the computation
        loss = torch.nn.functional.mse_loss(current_vertices[:, ::10], mocap_vertices[:, ::10], reduction='sum')

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

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
    S.add_geometry(trimesh.Trimesh(vertices=initial_vertices, faces=smplx_model.faces,
                    vertex_colors=create_color_array(len(initial_vertices), [255, 0, 0]))) # add the mesh M, the color is red
    S.add_geometry(trimesh.Trimesh(vertices=optimized_vertices.cpu().numpy(), faces=smplx_model.faces,
                    vertex_colors=create_color_array(len(optimized_vertices), [0, 0, 255])))
    S.show()

    return optimized_params


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # replace your smplx model path, it could be downloaded from https://smpl-x.is.tue.mpg.de/
    smplx_model_path = ''
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

    # please replace the path of dataset
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

            def add_parameters(smplx_parameters):
                transl.append(np.array(smplx_parameters['transl'].cpu().numpy()))
                global_orient.append(np.array(smplx_parameters['global_orient'].cpu().numpy()))
                body_pose.append(np.array(smplx_parameters['body_pose'].cpu().numpy()))
                betas.append(np.array(smplx_parameters['betas'].cpu().numpy()))

            learning_rate = 0.001
            iterations = 600
            for i in range(0, 2):
                smplx_parameters = vertices_to_params(vertices[i])
                add_parameters(smplx_parameters)

            np.save(os.path.join(folder_path, "AMC", "transl.npy"), transl)
            np.save(os.path.join(folder_path, "AMC", "global_orient.npy"), global_orient)
            np.save(os.path.join(folder_path, "AMC", "body_pose.npy"), body_pose)
            np.save(os.path.join(folder_path, "AMC", "betas.npy"), betas)
            print(f"transform vertices to smplx parameters in {vertices_path}")
        else:
            print(f"{vertices_path} not found")

    print("Done")



