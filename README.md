# UAV_REC_INTRO

By harry huang from ntu cae

## Introduction

This Repository is mainly for some introduction of current code. We will add feature to our current model and through the process you will be more fimiliar with our algorithm

I highly recommend you to test our code using UNIX like system, e.g. macOS, Ubuntu, Dabien etc.

## Download the data

To download the data, please run the

```
./download.sh
```

You might need to run the

```
chmod +x download.sh
```

before downloading the data

TODO:
draw the data directory in this README.md, but this is considered as low priority, do this in the future.

## Papers

To fully understanding this repository, there's three paper i highly recommend you to read.

```bibtex
Scannet Paper, this paper go through how those data is collected.
@inproceedings{dai2017scannet,
    title={ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes},
    author={Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
    booktitle = {Proc. Computer Vision and Pattern Recognition (CVPR), IEEE},
    year = {2017}
}
```

```bibtex
Bundle Fusion, this paper go through how to build a mesh map in a fast way,
which is the mesh model scannet is using, This is the high priority task you need to do.
@article{dai2017bundlefusion,
  title={BundleFusion: Real-time Globally Consistent 3D Reconstruction using On-the-fly Surface Re-integration},
  author={Dai, Angela and Nie{\ss}ner, Matthias and Zoll{\"o}fer, Michael and Izadi, Shahram and Theobalt, Christian},
  journal={ACM Transactions on Graphics 2017 (TOG)},
  year={2017}
}
```

```bibtex
This paper has mention how to back project the 3d mesh data back into the image plane
@article{dai2017bundlefusion,
  title={Virtual Multi-view Fusion for 3D Semantic Segmentation},
  author={Abhijit Kundu, Xiaoqi Yin, Alireza Fathi, David Ross, Brian Brewington, Thomas Funkhouser, Caroline Pantofaru},
  journal={arXiv},
  year={2022}
}
```

## Installation

### Requirements

To run the code, you need to make sure the following package is installed on you python. While usually the version will not have huge effect on the package we used currently, it's not the case on other packages we'll use in the future. So we'll specify the version in the future.

- numpy
- imagio
- opencv-python
- open3d

## Run the visualizer

To run the visualizer, please run the following code

```
python3 visualize_mesh.py --scene_name {scene name to visualize} --base_dir {the ground truth data base directory} --seg_out_dir {the segmentation label image directory} --visualize_type {visualize type} --skip_nums {skip number of visualize} --seg_type' {segmentation type}
```

there few arguments that you might not be familiar, let's talk more on these things.

--scene_name is the scene name you want to visualize, in the data i provided, there's two type of scenes you can try out. scene0538_00 and scene0559_00
--base_dir is the base directory that store all the ground truth data i provided to you, which is ./data/scannet/ in this repository.
--seg_out_dir is the output directory of segmentation label image. There's two type of choices ./mesh_out and ./data/scannet , this argument should be choose with seg_type, if the seg_out_dir is mesh_out, seg_type should be set as pred, otherwise should be set as gt
--seg_type as describe above
--visualize_type this is to specify what you want visualize, if you simply want to visualize the mesh please set as mesh. If you want to visualize the output of prediction you should set as seg_mesh
--skip_nums is to set how much frame we want to back-project to the map, it's kind of unrealistic to back project all the image into the map, but i set 1 in the default, i recommend to use 12 instead, this means the code will skip every 12 frames to visualize one frame.

There's one thing you need to be aware of, in current mode we only visualize the "segmentation class" in the mesh map, while it is simple to convert this behavior in the future.

## TODO

The following is all the things i want you to proceed, i'll list all by the order of priority.

1. Figure out how to use bundle-fusion code and build a mesh map that can provide the similar performance of provided scene mesh.

2. Find the TODO in the code and discuss with harry

3. After you are familiar with the visualization and the output of our SD-DETR model, think and try to implement the hungarian match jointly with 3d vertices.

4. There's still lots of thing todo, lets discuss in the future.
