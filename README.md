# Consistency driven Sequential Transformers Attention Model for Partially Observable Scenes
**Authors**: Samrudhdhi Rangrej, Chetan Srinidhi, James Clark
**Accepted to**: CVPR'22
[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Rangrej_Consistency_Driven_Sequential_Transformers_Attention_Model_for_Partially_Observable_Scenes_CVPR_2022_paper.pdf)
&nbsp;

&nbsp;

![architecture](./figures/architecture.png)
&nbsp;

An overview of our *Sequential Transformers Attention Model* **(STAM)**. The STAM consists of a core *T*, classifiers *G* and *D*, an actor *A*, and a critic *C* (only used during training). Each training iteration consists of three steps: **Step 1** (green path): Given a complete image *X*, the teacher model predicts a soft pseudo-label *q(y|X)*. **Step 2** (blue path): Given glimpses *g<sub>0:t</sub>*, STAM predicts class distributions *p<sub>g</sub>(y<sub>t</sub>|f<sup>g</sup><sub>t</sub>)* and *p<sub>d</sub>(y<sub>t</sub>|f<sup>d</sup><sub>t</sub>)*, value *V(s<sub>t</sub>)*, and attention policy *&pi;(l<sub>t+1</sub>|s<sub>t</sub>)*. **Step 3** (orange path): An additional glimpse is sensed. Step 2 is repeated using all glimpses (including the additional glimpse) and the losses are computed. The model parameters are updated using the gradients from Step 2.

## Requirements
* torch==1.8.1
* torchvision==0.9.1
* tensorboard==2.5.0
* timm==0.4.9
* fire==0.4.0

## Datasets
* [ImageNet](https://www.image-net.org/)
* [fMoW](https://github.com/fMoW/dataset) (Download and prepare the dataset following the instructions provided in [PatchDrop](https://github.com/ermongroup/PatchDrop) repository)

## Training
* Adapt [`paths.py`](./paths.py)
* Teacher models:
  - ImageNet: Download weights for *DeiT-small distilled* model from [deit](https://github.com/facebookresearch/deit).
  - fMoW: Finetune *DeiT-small* model from [deit](https://github.com/facebookresearch/deit) on the fMoW dataset for 100 epochs using the default hyperparameter setting and vertical flip augmentation.
* Student models (**STAM**):
  - ImageNet: run [`run_imagenet.sh`](./run_imagenet.sh)
  - fMoW: run [`run_fmow.sh`](./run_fmow.sh)

## Evaluation
* ImageNet: run [`eval_imagenet.sh`](./eval_imagenet.sh)
* fMoW: run [`eval_fmow.sh`](./eval_fmow.sh)

## Visualization
![imagenet_visualization](./figures/imagenet_visualization.png)
![fmow_visualization](./figures/fmow_visualization.png)
Visualization of glimpses selected by STAM on example images from *t=0* to *15*. (top) ImageNet; (bottom) fMoW. Complete images are shown for reference only. STAM does not observe a complete image.

## Acknowledgement
Our code is based on [deit](https://github.com/facebookresearch/deit).
