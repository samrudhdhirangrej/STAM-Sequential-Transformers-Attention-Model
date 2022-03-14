# Consistency driven Sequential Transformers Attention Model for Partially Observable Scenes
**Authors**: Samrudhdhi Rangrej, Chetan Srinidhi, James Clark
**Accepted to**: CVPR'22
&nbsp;

&nbsp;

![architecture](./figures/architecture.png)
&nbsp;

An overview of our *Sequential Transformers Attention Model* **(STAM)**. The STAM consists of a core *T*, classifiers *G* and *D*, an actor *A*, and a critic *C* (only used during training). Each training iteration consists of three steps: **Step 1** (green path): Given a complete image *X*, the teacher model predicts a soft pseudo-label *q(y|X)*. **Step 2** (blue path): Given glimpses *g<sub>0:t</sub>*, STAM predicts class distributions *p<sub>g</sub>(y<sub>t</sub>|f<sup>g</sup><sub>t</sub>)* and *p<sub>d</sub>(y<sub>t</sub>|f<sup>d</sup><sub>t</sub>)*, value *V(s<sub>t</sub>)*, and attention policy *&pi;(l<sub>t+1</sub>|s<sub>t</sub>)*. **Step 3** (orange path): An additional glimpse is sensed. Step 2 is repeated using all glimpses (including the additional glimpse) and the losses are computed. The model parameters are updated using the gradients from Step 2.
