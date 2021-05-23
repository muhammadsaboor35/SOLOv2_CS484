<div align="center">
    <img src="docs/adel-logo.svg" width="160" >
</div>

#  AdelaiDet

AdelaiDet is an open source toolbox for multiple instance-level recognition tasks on top of [Detectron2](https://github.com/facebookresearch/detectron2).
All instance-level recognition works from our group are open-sourced here.

To date, AdelaiDet implements the following algorithms:

* [FCOS](configs/FCOS-Detection/README.md)
* [BlendMask](configs/BlendMask/README.md)
* [MEInst](configs/MEInst-InstanceSegmentation/README.md)
* [ABCNet](configs/BAText/README.md)
* [CondInst](configs/CondInst/README.md)
* [SOLO](https://arxiv.org/abs/1912.04488) ([mmdet version](https://github.com/WXinlong/SOLO))
* [SOLOv2](configs/SOLOv2/README.md)
* [BoxInst](configs/BoxInst/README.md) ([video demo](https://www.youtube.com/watch?v=NuF8NAYf5L8))
* [DirectPose](https://arxiv.org/abs/1911.07451) _to be released_


# SOLOv2: Dynamic and Fast Instance Segmentation


> [**SOLOv2: Dynamic and Fast Instance Segmentation**](https://arxiv.org/abs/2003.10152),            
> Xinlong Wang, Rufeng Zhang, Tao Kong, Lei Li, Chunhua Shen     
> In: Proc. Advances in Neural Information Processing Systems (NeurIPS), 2020  
> *arXiv preprint ([arXiv 2003.10152](https://arxiv.org/abs/2003.10152))*  

# Datasets

* TrashCan Dataset: https://drive.google.com/file/d/1xFnUxug8IVpfForgp3STCT3vFnQ9cHwo/view?usp=sharing
* Augmented TrashCan Dataset: https://drive.google.com/file/d/1SJ4sAwsH9Ki24rutb8fYNBkmOihQRfdI/view?usp=sharing

# Installation & Quick Start
First, follow the [default instruction](../../README.md#Installation) to install the project and [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) 
set up the datasets (e.g., TrashCan).

For demo, run the following command lines:
```
wget https://drive.google.com/file/d/10f8mj8OUEQ1_YASCktXY6v9rRyMDZMLB/view?usp=sharing -O model_final.pth
python demo/demo.py \
    --config-file configs/SOLOv2/R50_1x.yaml \
    --input input1.jpg input2.jpg \
    --opts MODEL.WEIGHTS SOLOv2_R50_1x.pth
```

For training on TrashCan, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/SOLOv2/R50_1x.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/SOLOv2_R50_1x
```

# Utility Scripts
* tools/trash_can_coco.py: Script to convert trashcan dataset in MS-COCO format annotation
* tools/augmentation.py: Script to augment the dataset in MS-COCO format annotations
* tools/fix_bb_polygon.py: Script to fix some invalid augmentations in MS-COCO format annotations
* tools/get_frequencies.py: Script to obtain class-wise instance frequencies in MS-COCO format annotations


## Acknowledgements

The authors are grateful to
Nvidia, Huawei Noah's Ark Lab, ByteDance, Adobe who generously donated GPU computing in the past a few years.

## Citing AdelaiDet

If you use this toolbox in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@misc{tian2019adelaidet,
  author =       {Tian, Zhi and Chen, Hao and Wang, Xinlong and Liu, Yuliang and Shen, Chunhua},
  title =        {{AdelaiDet}: A Toolbox for Instance-level Recognition Tasks},
  howpublished = {\url{https://git.io/adelaidet}},
  year =         {2019}
}
```
and relevant publications:
```BibTeX

@inproceedings{tian2019fcos,
  title     =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year      =  {2019}
}

@article{tian2021fcos,
  title   =  {{FCOS}: A Simple and Strong Anchor-free Object Detector},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal =  {IEEE T. Pattern Analysis and Machine Intelligence (TPAMI)},
  year    =  {2021}
}

@inproceedings{chen2020blendmask,
  title     =  {{BlendMask}: Top-Down Meets Bottom-Up for Instance Segmentation},
  author    =  {Chen, Hao and Sun, Kunyang and Tian, Zhi and Shen, Chunhua and Huang, Yongming and Yan, Youliang},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{zhang2020MEInst,
  title     =  {Mask Encoding for Single Shot Instance Segmentation},
  author    =  {Zhang, Rufeng and Tian, Zhi and Shen, Chunhua and You, Mingyu and Yan, Youliang},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{liu2020abcnet,
  title     =  {{ABCNet}: Real-time Scene Text Spotting with Adaptive {B}ezier-Curve Network},
  author    =  {Liu, Yuliang and Chen, Hao and Shen, Chunhua and He, Tong and Jin, Lianwen and Wang, Liangwei},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{wang2020solo,
  title     =  {{SOLO}: Segmenting Objects by Locations},
  author    =  {Wang, Xinlong and Kong, Tao and Shen, Chunhua and Jiang, Yuning and Li, Lei},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}

@inproceedings{wang2020solov2,
  title     =  {{SOLOv2}: Dynamic and Fast Instance Segmentation},
  author    =  {Wang, Xinlong and Zhang, Rufeng and Kong, Tao and Li, Lei and Shen, Chunhua},
  booktitle =  {Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  year      =  {2020}
}

@article{tian2019directpose,
  title   =  {{DirectPose}: Direct End-to-End Multi-Person Pose Estimation},
  author  =  {Tian, Zhi and Chen, Hao and Shen, Chunhua},
  journal =  {arXiv preprint arXiv:1911.07451},
  year    =  {2019}
}

@inproceedings{tian2020conditional,
  title     =  {Conditional Convolutions for Instance Segmentation},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}

@inproceedings{tian2020boxinst,
  title     =  {{BoxInst}: High-Performance Instance Segmentation with Box Annotations},
  author    =  {Tian, Zhi and Shen, Chunhua and Wang, Xinlong and Chen, Hao},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2021}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).
