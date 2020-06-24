import tensorflow as tf
import numpy as np
# pylint: disable=no-name-in-module
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from lungs import i3d

def assign(global_vars, model_path):
    reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
    for var_3d in global_vars:
        if 'Logits' not in var_3d.op.name:
            var_2d_name = var_3d.op.name.replace('RGB/inception_i3d', 'InceptionV1')
            var_2d_name = var_2d_name.replace('Conv3d', 'Conv2d')
            var_2d_name = var_2d_name.replace('conv_3d/w', 'weights')
            var_2d_name = var_2d_name.replace('conv_3d/b', 'biases')
            var_2d_name = var_2d_name.replace('batch_norm', 'BatchNorm')
            
            var_value = reader.get_tensor(var_2d_name)

            if len(var_3d.get_shape()) - 1 == len(var_value.shape):
                n = var_value.shape[0]
                inflated = np.tile(np.expand_dims(var_value / n, 0), [n,1,1,1,1])
                var_3d.assign(tf.convert_to_tensor(inflated))
            else:
                var_3d.assign(tf.convert_to_tensor(var_value.reshape(var_3d.get_shape())))

def inflate_inception_v1_checkpoint_to_i3d(ckpt_2d, ckpt_3d, num_slices=145, volume_size=224, num_classes=2):
    '''
    Bootstrap the filters from a pre-trained two-dimensional 
    Inception-v1 checkpoint into a three-dimensional I3D checkpoint.
    
    Usage example:
    inflate_inception_v1_checkpoint_to_i3d('path/to/inception_v1.ckpt', 'path/to/new/i3d_model.ckpt')
    '''

    rgb_input = tf.placeholder(tf.float32,
        shape=(1, num_slices, volume_size, volume_size, 3))
    with tf.variable_scope('RGB'):
        i3d.InceptionI3d(num_classes, spatial_squeeze=True, final_endpoint='Logits')\
                        (rgb_input, is_training=False)
    
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        assign(tf.global_variables(), ckpt_2d)
        tf.train.Saver().save(sess, ckpt_3d)

def inspect_checkpoint(ckpt):
    print_tensors_in_checkpoint_file(ckpt, all_tensors=False, tensor_name='')
