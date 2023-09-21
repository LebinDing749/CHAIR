# Dataset Description

This dataset focuses on 3D human motion, specifically capturing actions where humans interact with chairs, encompassing both individual chair interactions and more complex scenes.

**Dataset Contents:** This dataset comprises human motion data files organized into separate folders for each model's motion. Each motion is placed within its own subfolder. The motion files include SMPL-X parameters, original ASF/AMC files, and information regarding object motion.

**Source Files:** The "source_files" directory contains files related to object models and models themselves. It includes .obj format object model files and .fbx format model files.

**Videos:** The "videos" directory contains demonstration videos of human motions, with subdirectories named in the same way as the dataset's subdirectories.

```python
/Chairs++
    ├── dataset
    |   ├── motion file of a model
    |	|	├──motion of several actions in succession
   	|	|	|	├──AMC
    |	|	|	|	transl.npy
    |	|	|	|	global_orient.npy
    |	|	|	|	body_pose.npy
    |	|	|	|	betas.npy
    |	|	|	|	...
    |	|	|	├──C3D
    |	|	|	obj_id.npy
    |	|	|	obj_transl.npy
   	|	|	|	obj_orient.npy
    |	|	|	...
    |	|	├──...
    |   ├── ...
    |	├── lable
    |
    ├── source_files
    |   ├── chairs_img
    |   ├── chairs_obj
    |   ├── model_fbx
    |   ├── ...
    |
    ├── videos
    |   ├── video of a model
    |   ├── ...
```

This dataset is intended for use in various fields, including computer vision, 3D human motion generation. 



# Dataset Details

1. **Output Formats:** The dataset provides motion data in various formats, including C3D, ASF/AMC, BVH, SMPL-X , information regarding object motion, and demonstration videos.
2. **Data Volume:** The dataset comprises data from six models, featuring approximately 280 motion sequences. There are interactions with 32 different objects, with around 20 objects exhibiting motion during the interactions. Each motion sequence consists of six consecutive actions. The duration of each motion sequence ranges from 30 to 45 seconds, with a frame rate of 60 frames per second (FPS). The total duration of the dataset is approximately 2.8 hours.





# Demonstration

![](images/single.gif)

![](images/scene.gif)



# Method of converting AMC to SMPL

#### 1. convert the .asf/.amc to .bvh

We recommend using [amc2bvh](https://github.com/sxaxmz/amc2bvh), which is open-source and free. Although amc2bvh does not consider scaling, we can set the global scale when importing .bvh into Blender.

In addition, we modified the first frame of the .amc file to match the T-pose (initial body_pose) of smpl-x.

#### 2.retarget the motion of .bvh to smpl-x model in blender

To create a SMPL-X model in blender, we recommend the [SMPL-Blender-Addon](https://github.com/Meshcapade/SMPL_blender_addon). If you understand the source code of the SMPL-Blender-Addon, you can manually set the model's ***betas*** parameters, not just height and weight.

The root bone of the model created with the SMPL-X Blender add-on named '***root*'**,  which can be used in Blender to represent global displacement. 

However, it is not necessary for describing the pose of the SMPL-X model. In fact, the bone hierarchy with ***'pelvis'*** as the root bone is more similar to the structure of ASF skeletons, which is crucial for the retargeting. So, we removed the ***'root'*** bone of the model, making 'pelvis' the root bone instead.

Then, we used the [Rokoko Studio live for Blender](https://support.rokoko.com/hc/en-us/articles/4410463492241-Install-the-Blender-plugin) for retargeting. The [bone_mapping](scirpt\bone_mapping.json) is provided.

#### 3.export  and calculate SMPL-X parameters

The ***betas*** and ***body_pose*** per frame can be directly exported, and parts of the SMPL Blender add-on code can be used directly in Blender Python scripts. Additionally, after obtaining ***betas***, ***body_pose***, ***joint_position*** and ***mesh_vertices***, you can calculate ***transl*** and ***global_orient***.

All the script of data processing are located in the './script'.



