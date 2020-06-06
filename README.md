# Lung Cancer Risk Prediction With 3D Neural Network on CT

## Overview

This repository contains an implementation of the "full-volume" model from the paper:  

[End-to-end lung cancer screening with three-dimensional deep learning on low-dose chest computed tomography.](https://doi.org/10.1038/s41591-019-0447-x)<br/> Ardila, D., Kiraly, A.P., Bharadwaj, S. et al. Nat Med 25, 954–961 (2019).

The model uses a three-dimensional (3D) CNN to perform end-to-end analysis of whole-CT volumes, using LDCT
volumes with pathology-confirmed cancer as training data. 
The CNN architecture is Inflated 3D ConvNet (I3D) ([Carreira and
Zisserman](http://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html))

The repository also includes a pre-trained checkpoint `data/checkpoints`, which achieves a score of AUC 90.0 on a subset of NLST with 2,000 CT images. Data can only be made available by NLST and requires approval.

Disclaimer: This is not an official product.

## Data
We use the NLST dataset which cintains chest LDCT volumes with pathology-confirmed cancer evaluations. For description and access to the dataset refer to [NCI website](https://biometry.nci.nih.gov/cdas/learn/nlst/images/).

### Setup

Then, clone this repository using

```
$ git clone https://github.com/danielkorat/Lung-Cancer-Risk-Prediction
$ cd Lung-Cancer-Risk-Prediction
$ pip install -U pip
$ pip install -r requirements.txt
```

## Running the code


### Sample code


### Provided checkpoint
The model is pre-trained on ImageNet and then NLST for binary classification.
The directory `data/checkpoints` contains the best checkpoint that was
trained. The [ImageNet pre-trained Inception V1 model](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz) is inflated to 3D and then fine-tuned on pathology-confirmed CTs from NLST. This checkpoint is initialized by bootstrapping the filters from a [2D Inception-v1 model]((http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)) into 3D,
as described in the paper.

The model is initialized by bootstrapping the filters from a [ImageNet pre-trained 2D Inception-v1 model]((http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)) into 3D,
as described in the paper.
It is then trained on the preprocessed CT volumes to predict cancer within 1 year, fine-tuning from the pretrained checkpoint mentioned above. Each of these volumes was a large region cropped around the center of the bounding box, as determined by lung segmentation in the preprocessing stage. We use focal loss to try to mitigate the sparsity of positive examples.

We train using `tf.train.AdamOptimizer`. During training, we use ?.? dropout, with a
minibatch size of 3. The optimizer uses learning rate of 1e-4, with and exponential decay rate of 0.1.
We train the model for ???k steps (?? epochs).


### Data Preprocessing
Each CT volume downloaded from NLST is a folder of DICOM files (one per slice).
The `preprocess.py` module accepts a directory `path/to/data` containing multiple CT volumes, performs several preprocessing steps on each volume, and saves each preprocessed volume as 3D `.npy` file in `path/to/data_preprocssed`.
The preprocessing steps include: Resampling to 1.0mm^3 voxels, windowing, lung segmentation and centering, RGB normalization.

### Acknowledgments

The author thanks the National Cancer Institute for access to NCI's data collected by the National Screening Trial (NLST).
The statements contained herein are solely those of the author and do not represent or imply concurrence or endorsement by NCI.
