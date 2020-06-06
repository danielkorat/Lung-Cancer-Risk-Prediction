# Lung Cancer Risk Prediction With 3D Neural Network on CT

## Overview

This repository contains an implementation of the "full-volume" model from the paper:  
[Ardila, D., Kiraly, A.P., Bharadwaj, S. et al. <br/>End-to-end lung cancer screening with 
three-dimensional deep learning on low-dose chest computed tomography. <br/> Nat Med 25, 954–961 (2019).](https://doi.org/10.1038/s41591-019-0447-x)  
The model uses a three-dimensional (3D) CNN to perform end-to-end analysis of whole-CT volumes, using LDCT
volumes with pathology-confirmed cancer as training data.
The CNN architecture is Inflated 3D ConvNet (I3D) from "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](http://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)" by Joao Carreira and Andrew
Zisserman. 
The [ImageNet pre-trained Inception V1 model](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz) is inflated to 3D and then fine-tuned on pathology-confirmed CTs from the NLST dataset.

The repository also includes a pre-trained checkpoint using rgb inputs and trained from scratch on Kinetics-600.

Disclaimer: This is not an official product.

## Running the code

### Setup

Then, clone this repository using

`$ git clone https://github.com/danielkorat/Lung-Cancer-Risk-Prediction`
`$ cd Lung-Cancer-Risk-Prediction`
`$ pip install -U pip`
`$ pip install -r requirements.txt`

### Sample code

### Running the test

The test file can be run using

`$ python i3d_test.py`

This checks that the model can be built correctly and produces correct shapes.

## Further details

### Provided checkpoints

The default model has been pre-trained on ImageNet and then Kinetics; other
flags allow for loading a model pre-trained only on Kinetics and for selecting
only the RGB or Flow stream. The script `multi_evaluate.sh` shows how to run all
these combinations, generating the sample output in the `out/` directory.

The directory `data/checkpoints` contains the four checkpoints that were
trained. The ones just trained on Kinetics are initialized using the default
Sonnet / TensorFlow initializers, while the ones pre-trained on ImageNet are
initialized by bootstrapping the filters from a 2D Inception-v1 model into 3D,
as described in the paper. Importantly, the RGB and Flow streams are trained
separately, each with a softmax classification loss. During test time, we
combine the two streams by adding the logits with equal weighting, as shown in
the `evalute_sample.py` code.

We train using synchronous SGD using `tf.train.SyncReplicasOptimizer`. For each
of the RGB and Flow streams, we aggregate across 64 replicas with 4 backup
replicas. During training, we use 0.5 dropout and apply BatchNorm, with a
minibatch size of 6. The optimizer used is SGD with a momentum value of 0.9, and
we use 1e-7 weight decay. The RGB and Flow models are trained for 115k and 155k
steps respectively, with the following learning rate schedules.

RGB:

*   0 - 97k: 1e-1
*   97k - 108k: 1e-2
*   108k - 115k: 1e-3

Flow:

*   0 - 97k: 1e-1
*   97k - 104.5k: 1e-2
*   104.5k - 115k: 1e-3
*   115k - 140k: 1e-1
*   140k - 150k: 1e-2
*   150k - 155k: 1e-3

This is because the Flow models were determined to require more training after
an initial run of 115k steps.

The models are trained using the training split of Kinetics. On the Kinetics
test set, we obtain the following top-1 / top-5 accuracy:

Model          | ImageNet + Kinetics | Kinetics
-------------- | :-----------------: | -----------
RGB-I3D        | 71.1 / 89.3         | 68.4 / 88.0
Flow-I3D       | 63.4 / 84.9         | 61.5 / 83.4
Two-Stream I3D | 74.2 / 91.3         | 71.6 / 90.0

### Sample data and preprocessing


### Acknowledgments

The author thanks the National Cancer Institute for access to NCI's data collected by the National Screening Trial (NLST).
The statements contained herein are solely those of the author and do not represent or imply concurrence or endorsement by NCI.
