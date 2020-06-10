# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import numpy as np

def apply_window(image):
    # Windowing
    # Our values currently range from -1024 to around 2000. 
    # Anything above 400 is not interesting to us, as these are simply bones with different radiodensity.  
    # A commonly used set of thresholds in Lungs LDCT to normalize between are -1000 and 400. 
    min_bound = -1000.0
    max_bound = 400.0
    image = (image - min_bound) / (max_bound - min_bound)
    image[image>1] = 1.
    image[image<0] = 0.

    # Normalize rgb values to [-1, 1]
    image = (image * 2) - 1

    return np.stack((image, image, image), axis=4)

def write_number_list(lst, f_name, verbose=False):
    if verbose:
        print('INFO: Saving ' + f_name + '.npz ...')
        print(lst)
    np.savez(f_name + '.npz', np.array(lst))       

def append_and_write(*args):
    for lst, new_item, f_name in args:
        # print(f_name + ' :', '\n', 'Items:', lst, '\n New item:', new_item)
        lst.append(new_item)
        write_number_list(lst, f_name)

def batcher(iterable, batch_size=1):
    iter_len = len(iterable)
    for i in range(0, iter_len, batch_size):
        yield iterable[i: min(i + batch_size, iter_len)]

def load_data_list(path):
    coupled_data = []
    with open(path) as file_list_fp:
        for line in file_list_fp:
            image_path, label = line.split()
            coupled_data.append((image_path, int(label)))
    return coupled_data

def placeholder_inputs(num_frames, crop_size, rgb_channels=3):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.
    num_frame_per_clip: The num of frame per clib.
    crop_size: The crop size of per clib.
    channels: The input channel of per clib.

    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           num_frames,
                                                           crop_size,
                                                           crop_size,
                                                           rgb_channels))
    labels_placeholder = tf.placeholder(tf.int64, shape=(None))
    is_training = tf.placeholder(tf.bool)
    return images_placeholder, labels_placeholder, is_training

def focal_loss(logits, labels, alpha=0.75, gamma=2):
    """Compute focal loss for binary classification.

    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.sigmoid(logits)
    labels = tf.to_float(labels)
    losses = -(labels * (1 - alpha) * ((1 - y_pred) * gamma) * tf.log(y_pred)) - \
        (1 - labels) * alpha * (y_pred ** gamma) * tf.log(1 - y_pred)
    return tf.reduce_mean(losses)

def cross_entropy_loss(logits, labels):
    print('CE Loss variables:')
    print('labels:', labels)
    print('logits:', logits)
    print('logits.shape:', logits.shape)
    cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                  )
    return cross_entropy_mean


def accuracy(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def get_preds(preds):
    return preds[:, 1]

def get_logits(logits):
    return logits
