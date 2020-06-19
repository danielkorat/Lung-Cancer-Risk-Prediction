# 3D Neural Network for Lung Cancer Risk Prediction on CT Volumes

## Overview

This repository contains an implementation of the "full-volume" model from the paper:  

[End-to-end lung cancer screening with three-dimensional deep learning on low-dose chest computed tomography.](https://doi.org/10.1038/s41591-019-0447-x)<br/> Ardila, D., Kiraly, A.P., Bharadwaj, S. et al. Nat Med 25, 954–961 (2019).

![Model Workflow](https://raw.githubusercontent.com/danielkorat/Lung-Cancer-Risk-Prediction/master/figures/model_workflow.png)

The model uses a three-dimensional (3D) CNN to perform end-to-end analysis of whole-CT volumes, using LDCT
volumes with pathology-confirmed cancer as training data.
The CNN architecture is an Inflated 3D ConvNet (I3D) ([Carreira and
Zisserman](http://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)).

### Data

We use the NLST dataset which contains chest LDCT volumes with pathology-confirmed cancer evaluations. For description and access to the dataset refer to the [NCI website](https://biometry.nci.nih.gov/cdas/learn/nlst/images/).

![Example cases](https://raw.githubusercontent.com/danielkorat/Lung-Cancer-Risk-Prediction/master/figures/example_cases.png)

## Running the code

### Setup

```bash
git clone https://github.com/danielkorat/Lung-Cancer-Risk-Prediction
cd Lung-Cancer-Risk-Prediction
pip install -U pip
pip install -r requirements.txt
```

The `main.py` module contains training (fine-tuning) and inference procedures.
The inputs are preprocessed CT volumes, as produced by `preprocess.py`.
For usage example, refer to the arguments' description and default values in the bottom of `main.py`.

### Data Preprocessing

The `main.py` module operates only on preprocessed volumes, produced by `preprocess.py`.
Each CT volume in NLST is a folder of DICOM files (one file per slice).
The `preprocess.py` module accepts a directory `path/to/data` containing multiple CT volumes, performs several preprocessing steps, and writes each volume as an `.npz` file in `path/to/data_preprocssed`.
The preprocessing steps include methods from [this](https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial/notebook) tutorial and include:

- Resampling to a 1.4mm voxel size (slow).
- Coarse lung segmentation – used to compute lung center for alignment and reduction of problem space.

To save storage space, following preprocessing steps are performed online (during training/inference):

- Windowing – clip pixel values to focus on lung volume.
- RGB normalization

### Provided checkpoint

We also provide our cancer fine-tuned model for out-of-the box usage.
By default, this model checkpoint is downloaded in `main.py` and the model is then initialized with its weights.

To train this model we first initialized by bootstrapping the filters from the [ImageNet pre-trained 2D Inception-v1 model]((http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)) into 3D, as described in the paper.
It was then fine-tuned on the preprocessed CT volumes to predict cancer within 1 year (binary classification), fine-tuning from the pretrained checkpoint mentioned above. Each of these volumes was a large region cropped around the center of the bounding box, as determined by lung segmentation in the preprocessing stage. We use focal loss to try to mitigate the sparsity of positive examples.

Due to limited storage and compute time, we trained our model on a small subset of NLST containing 1,045 volumes (34% positive).
Nevertheless, we still achieved a high AUC score of 0.87 on a validation set of 448 volumes.
This is rouhly comparable to the original paper's result of 0.89 AUC for the full-volume model (see the paper's supplemtary material),
trained on 47,974 volumes (1.34% positive).
We set the dropout keep_prob to 0.7, and trained in minibatches of size of 2 (due to limited GPU memory).
We used `tf.train.AdamOptimizer` with a small learning rate of 5e-5 (without decay), due to the small batch size.
We stopped the training before overfitting started around epoch 32.

### Acknowledgments

The author thanks the National Cancer Institute for access to NCI's data collected by the National Screening Trial (NLST).
The statements contained herein are solely those of the author and do not represent or imply concurrence or endorsement by NCI.
