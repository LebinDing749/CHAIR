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