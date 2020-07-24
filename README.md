# Det3D

A general 3D Object Detection codebase in PyTorch.

## 1. Introduction

Det3D is the first 3D Object Detection toolbox which provides off the box implementations of many 3D object detection algorithms such as PointPillars, SECOND, PIXOR, etc, as well as state-of-the-art methods on major benchmarks like KITTI(ViP) and nuScenes(CBGS). Key features of Det3D include the following aspects:

* Multi Datasets Support: KITTI, nuScenes, Lyft
* Point-based and Voxel-based model zoo
* State-of-the-art performance
* DDP & SyncBN


## 2. Installation

Please refer to [INSTALATION.md](INSTALLATION.md).

## 3. Quick Start

Please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## 4. Model Zoo

### 4.1 nuScenes

|             | mAP  | mATE | mASE | mAOE | mAVE | mAAE | NDS | ckpt |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| [CBGS](https://github.com/poodarchu/Det3D/blob/master/examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py) | 49.9 | 0.335 | 0.256 | 0.323 | 0.251 | 0.197 | 61.3 | [link](https://drive.google.com/drive/folders/1rhamAqegE9iOp18tzQVam4rOMhHjjnRM?usp=sharing) |
| [PointPillar](examples/point_pillars/configs/nusc_all_point_pillars_mghead_syncbn.py) | 41.8 | 0.363 | 0.264 | 0.377 | 0.288 | 0.198 | 56.0 | [link](https://drive.google.com/drive/folders/1U0bkEQAhcxhDUD42nTCGC0uU0qaTO_Uv?usp=sharing) |

The original model and prediction files are available in the [CBGS README](https://github.com/poodarchu/Det3D/tree/master/examples/cbgs).

### 4.2 KITTI

### [Second](examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py) on KITTI(val) Dataset

```
car  AP @0.70, 0.70,  0.70:
bbox AP:90.54, 89.35, 88.43
bev  AP:89.89, 87.75, 86.81
3d   AP:87.96, 78.28, 76.99
aos  AP:90.34, 88.81, 87.66
```

### [PointPillars](examples/point_pillars/configs/kitti_point_pillars_mghead_syncbn.py) on KITTI(val) Dataset

```	
car  AP@0.70,  0.70,  0.70:
bbox AP:90.63, 88.86, 87.35
bev  AP:89.75, 86.15, 83.00
3d   AP:85.75, 75.68, 68.93
aos  AP:90.48, 88.36, 86.58
```


### 4.3 Lyft

* [Lyft Config](https://github.com/poodarchu/Det3D/blob/master/examples/cbgs/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py)

### 4.4 Waymo



## 5. Functionality

* Models
  - [x] VoxelNet
  - [x] SECOND
  - [x] PointPillars
* Features
    - [x] Multi task learning & Multi-task Learning
    - [x] Distributed Training and Validation
    - [x] SyncBN
    - [x] Flexible anchor dimensions
    - [x] TensorboardX
    - [x] Checkpointer & Breakpoint continue
    - [x] Self-contained visualization
    - [x] Finetune
    - [x] Multiscale Training & Validation
    - [x] Rotated RoI Align


## 6. TODO List
* To Be Released

  * [ ] [CGBS](examples/cbgs/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py) on Lyft(val) Dataset

* Models
  
  - [ ] PointRCNN
  - [ ] PIXOR

## 7. Call for contribution.
* Support Waymo Dataset.
* Add other 3D detection / segmentation models, such as VoteNet, STD, etc.

## 8. Developers

[Benjin Zhu](https://github.com/poodarchu/) , [Bingqi Ma](https://github.com/a157801)

## 9. License

Det3D is released under the [Apache licenes](LICENES).

## 10. Citation
Det3D is a derivative codebase of [CBGS](https://arxiv.org/abs/1908.09492), if you find this work useful in your research, please consider cite:
```
@article{zhu2019class,
  title={Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection},
  author={Zhu, Benjin and Jiang, Zhengkai and Zhou, Xiangxin and Li, Zeming and Yu, Gang},
  journal={arXiv preprint arXiv:1908.09492},
  year={2019}
}
```

## 11. Acknowledgement

* [mmdetection](https://github.com/open-mmlab/mmdetection) 
* [mmcv](https://github.com/open-mmlab/mmcv)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)


## memo
Evaluation official: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:90.62, 88.66, 87.40
bev  AP:89.83, 87.28, 85.66
3d   AP:86.53, 76.85, 73.75
aos  AP:87.52, 83.54, 80.94
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:90.62, 88.66, 87.40
bev  AP:90.74, 89.73, 89.11
3d   AP:90.74, 89.65, 88.91
aos  AP:87.52, 83.54, 80.94

Evaluation coco: car coco AP@0.50:0.05:0.95:
bbox AP:71.58, 67.93, 66.34
bev  AP:69.67, 66.09, 64.87
3d   AP:59.65, 55.08, 53.56
aos  AP:68.43, 63.57, 61.11

### Create onnx file
```./tools/create_onnx.py```

### todo
export pfe.onnx and rpn.onnx like Autoware implementation