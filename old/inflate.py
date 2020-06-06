import tensorflow as tf
import numpy as np
import i3d


_SAMPLE_VIDEO_FRAMES = 79
_IMAGE_SIZE = 224
_NUM_CLASSES  = 2
ROOT = '/home/daniel_nlp/Lung-Cancer-Detection-and-Classification/kinetics-i3d/data/'

def assign(global_vars, model_path):
    reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
    var_map = {}
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

def inflate_inception_v1_checkpoint_to_i3d(ckpt_2d, ckpt_3d):
    rgb_input = tf.placeholder(tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
        _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
    
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        assign(tf.global_variables(), ckpt_2d)
        tf.train.Saver().save(sess, ckpt_3d)

# inflate_inception_v1_checkpoint_to_i3d(ROOT + 'checkpoints/2d/inception_v1.ckpt', ROOT + 'checkpoints/inflated/model.ckpt')

# inspect checkpoint
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
init_ckpt = ROOT + 'checkpoints/init/model.ckpt'
# inflated_ckpt = ROOT + 'checkpoints/inflated/model.ckpt'
print_tensors_in_checkpoint_file(init_ckpt, all_tensors=False, tensor_name='')