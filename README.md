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
