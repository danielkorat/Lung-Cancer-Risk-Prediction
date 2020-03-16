import tensorflow as tf
import numpy as np
import i3d

# def rebuild_ckpoint_imagenet(checkpoint_dir, save_path):
    # """rebuild the checkpoint from imagenet 2d model
    # Inception-v2 inflated 3d ConvNet
    # """
    # checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    # fg = True
    # with tf.Session() as sess:
    #     for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
    #         print(var_name)
    #         raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
    #         if var_name.startswith('InceptionV2/Conv2d_1a_7x7'):
    #             if fg:
    #                 # var_name = 'v/SenseTime_I3D_V2/Conv3d_1a_7x7x7/kernel'
    #                 var_name = 'RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/w'
    #                 raw_var = np.random.normal(0.0, 1.0, (7, 7, 7, 3, 64)) / 7.0
    #                 fg = False
    #             else:
    #                 # print(var_name)
    #                 continue
    #         elif var_name.find('weights') > -1:
    #             kernel = raw_var.shape[0]
    #             res = [raw_var for i in range(kernel)]
    #             raw_var = np.stack(res, axis=0)
    #             raw_var = raw_var / (kernel * 1.0)
    #         for k, v in IMAGENET_NAME_MAP.items():
    #             var_name = var_name.replace(k, v)
    #         # print(var_name)
    #         var = tf.Variable(raw_var, name=var_name)

    #     saver = tf.train.Saver()
    #     sess.run(tf.global_variables_initializer())
    #     saver.save(sess, save_path)

# rebuild_ckpoint_imagenet(
#     '/home/daniel_nlp/Lung-Cancer-Detection-and-Classification/kinetics-i3d/data/checkpoints/2d/inception_v2.ckpt', 
#     '/home/daniel_nlp/Lung-Cancer-Detection-and-Classification/kinetics-i3d/data/checkpoints/inflated/v2/model.ckpt')

# rebuild_ckpoint_imagenet(
#     '/home/daniel_nlp/Lung-Cancer-Detection-and-Classification/kinetics-i3d/data/checkpoints/2d/inception_v1.ckpt', 
#     '/home/daniel_nlp/Lung-Cancer-Detection-and-Classification/kinetics-i3d/data/checkpoints/inflated/v1/model.ckpt')


IMAGENET_NAME_MAP = {
    # 'InceptionV2/': 'v/SenseTime_I3D_V2/',
    # 'Conv2d': 'Conv3d',
    # 'weights': 'kernel',
    # '1x1': '1x1x1',
    # '3x3': '3x3x3',
    # '7x7': '7x7x7',
    'InceptionV2/': 'RGB/inception_i3d/',
    'Conv2d': 'Conv3d',
    'weights': 'w',
    '1x1': '1x1',
    '3x3': '3x3',
    '7x7': '7x7'
}

def assign_from_checkpoint_2d_to_3d_scale(model_path, var_list):
    """Creates an operation to assign specific variables from a checkpoint.

    Args:
        model_path: The full path to the model checkpoint. To get latest checkpoint
        use model_path = tf.train.latest_checkpoint(checkpoint_dir)
        var_list: A list of Variable objects or a dictionary mapping names in the
        checkpoint to the corresponding variables to initialize. If empty or
        None, it would return no_op(), None.

    Returns:
        the restore_op and the feed_dict that need to be run to restore var_list.

    Raises:
        ValueError: If the checkpoint specified at model_path is missing one of
        the variables in var_list.
    """

    reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
    if isinstance(var_list, (tuple, list)):
        var_list = {var.op.name: var for var in var_list}
    feed_dict = {}
    assign_ops = []

    for checkpoint_var_name in var_list:
        var = var_list[checkpoint_var_name]
        if not reader.has_tensor(checkpoint_var_name):
            raise ValueError('Checkpoint is missing variable [%s]' % checkpoint_var_name)

        var_value = reader.get_tensor(checkpoint_var_name)
        placeholder_name = var.op.name
        placeholder_value = tf.placeholder(
            dtype=var.dtype.base_dtype,
            shape=var.get_shape(),
            name=placeholder_name)
        assign_ops.append(var.assign(placeholder_value))

        if var.get_shape() != var_value.shape:
            n = var_value.shape[0]
            inflated = np.tile(np.expand_dims(var_value / n,0), [n,1,1,1,1])
            feed_dict[placeholder_value] = tf.convert_to_tensor(inflated, dtype=var.dtype.base_dtype)
        else:
            feed_dict[placeholder_value] = tf.convert_to_tensor(var_value.reshape(var.get_shape()), dtype=var.dtype.base_dtype)

    assign_op = tf.group(*assign_ops)
    return assign_op, feed_dict

# rgb_input = tf.placeholder(
#     tf.float32, shape=(1, 79, 224, 224, 3))
# with tf.variable_scope('RGB'):
#     rgb_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
#     rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

#     var_map = {}
#     for var_3d in tf.global_variables():
#         var_2d_name = var_3d.op.name.replace('RGB/inception_i3d', 'InceptionV1')
#         var_2d_name = var_2d_name.replace('Conv3d', 'Conv2d')
#         var_2d_name = var_2d_name.replace('conv_3d/w', 'weights')
#         var_2d_name = var_2d_name.replace('conv_3d/b', 'biases')
#         var_2d_name = var_2d_name.replace('batch_norm', 'BatchNorm')
#         var_map[var_2d_name] = var_3d

# assign_from_checkpoint_2d_to_3d_scale(
#     '/home/daniel_nlp/Lung-Cancer-Detection-and-Classification/kinetics-i3d/data/checkpoints/2d/inception_v1.ckpt', 
#     var_map)

def assign(global_vars):
    model_path = '/home/daniel_nlp/Lung-Cancer-Detection-and-Classification/kinetics-i3d/data/checkpoints/2d/inception_v1.ckpt'
    reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
    var_map = {}
    for var_3d in global_vars:
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

_SAMPLE_VIDEO_FRAMES = 79
_IMAGE_SIZE = 224
_NUM_CLASSES  = 1001
def inflate_inception_v1_checkpoint_to_i3d(save_path):
    rgb_input = tf.placeholder(tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
        _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
    
    with tf.Session() as sess:
        sess.run(init_op)
        assign(tf.global_variables())
        tf.train.Saver().save(sess, save_path)

inflate_inception_v1_checkpoint_to_i3d(ROOT + 'data/checkpoints/inflated')