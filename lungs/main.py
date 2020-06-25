VERBOSE_TF = False
import os
if not VERBOSE_TF:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    import tensorflow as tf
from random import shuffle
import numpy as np
import argparse
from time import time, strftime
from tqdm import tqdm
from os.path import join, dirname, realpath
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from datetime import date
from pathlib import Path

from lungs.preprocess import preprocess, walk_dicom_dirs, walk_np_files
from lungs import utils
from lungs.i3d import InceptionI3d

class I3dForCTVolumes:
    def __init__(self, args):
        self.args = args

        # This is the shape of both dimensions of each slice of the volume.
        # The final volume shape fed to the model is [self.args['num_slices, 224, 224]
        self.slice_size = 224

        # pylint: disable=not-context-manager
        with tf.Graph().as_default():
            global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )

            # Placeholders
            self.volumes_placeholder, self.labels_placeholder, self.is_training_placeholder = utils.placeholder_inputs(
                    num_slices=self.args['num_slices'],
                    crop_size=self.slice_size,
                    rgb_channels=3
                    )
            
            # Learning rate and optimizer
            lr = tf.train.exponential_decay(self.args['lr'], global_step, decay_steps=5000, decay_rate=0.1, staircase=True)
            optimizer = tf.train.AdamOptimizer(lr)

            # Init I3D model
            with tf.device('/device:' + self.args['device'] + ':0'):
                with tf.compat.v1.variable_scope('RGB'):
                    _, end_points = InceptionI3d(num_classes=2, final_endpoint='Predictions')\
                        (self.volumes_placeholder, self.is_training_placeholder, dropout_keep_prob=args['keep_prob'])
                self.logits = end_points['Logits']
                self.preds = end_points['Predictions']

                # Loss function
                # self.loss = utils.focal_loss(self.logits[:, 1], self.labels_placeholder)
                self.loss = utils.cross_entropy_loss(self.logits, self.labels_placeholder)

                # Evaluation metrics
                self.get_preds = utils.get_preds(self.preds)
                self.get_logits = utils.get_logits(self.logits)
                self.accuracy = utils.accuracy(self.logits, self.labels_placeholder)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grads = optimizer.compute_gradients(self.loss)
                    apply_gradient = optimizer.apply_gradients(grads, global_step=global_step)
                    self.train_op = tf.group(apply_gradient)

            # Create a saver for loading pretrained checkpoints.
            pretrained_variable_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'RGB' and 'Adam' not in variable.name.split('/')[-1] \
                    and variable.name.split('/')[2] != 'Logits':
                    pretrained_variable_map[variable.name.replace(':0', '')] = variable
            self.pretrained_saver = tf.train.Saver(var_list=pretrained_variable_map, reshape=True)

            # Create a saver for writing training checkpoints.
            self.saver = tf.train.Saver()

            # Init local and global vars
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            # Create a session for running Ops on the Graph.
            run_config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=run_config)
            self.sess.run(init)

    def train_loop(self, train_list, metrics_dir):
        train_batches = utils.batcher(train_list, self.args['batch_size'])
        for coupled_batch in tqdm(train_batches):
            feed_dict, _ = self.process_data_into_to_dict(coupled_batch, is_training=True)
            self.sess.run(self.train_op, feed_dict=feed_dict)

        metrics = self.evaluate(train_list, ds='Train')
        utils.write_number_list(metrics[-1], join(metrics_dir, 'tr_true'), verbose=self.args['verbose'])
        return metrics

    def evaluate(self, coupled_list, ds='Val.'):
        coupled_batches = utils.batcher(coupled_list, self.args['batch_size'])

        loss_list, acc_list, preds_list, labels_list = [], [], [], []
        
        print('\nINFO: ++++++++++++++++++++ {} Evaluation ++++++++++++++++++++'.format(ds))
        for coupled_batch in tqdm(coupled_batches):
            feed_dict, labels = self.process_data_into_to_dict(coupled_batch)
            acc, loss, preds = self.sess.run([self.accuracy, self.loss, self.get_preds], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)
            preds_list.extend(preds)
            labels_list.extend(labels)

        if self.args['verbose']:
            print('\nDEBUG: {}. Preds/Labels: {}'.format(ds, list(zip(preds_list, labels_list))))
            print('\nDEBUG: {} Batch accuracy/loss: {}'.format(ds, list(zip(acc_list, loss_list))))

        mean_acc = np.mean(acc_list)
        mean_loss = np.mean(loss_list)
        auc_score = roc_auc_score(labels_list, preds_list)
        print('\n' + '=' * 34)
        print("||  INFO: {} Accuracy: {:.4f} ||".format(ds, mean_acc))
        print("||  INFO: {} Loss:     {:.4f} ||".format(ds, mean_loss))
        print("||  INFO: {} AUC:      {:.4f} ||".format(ds, auc_score))
        print('=' * 34)
        return mean_loss, mean_acc, auc_score, preds_list, labels_list

    def predict(self, inference_data):
        if inference_data == 'sample_data':
            inference_data = join(dirname(realpath(__file__)), 'sample_data')

        errors_map = defaultdict(int)
        volume_iterator = walk_np_files(inference_data) if self.args['preprocessed'] else walk_dicom_dirs(inference_data)
        
        for i, volume_path in enumerate(volume_iterator):
            try:
                if not self.args['preprocessed']:
                    print('\nINFO: Preprocessing volume...')
                    preprocessed, _ = preprocess(volume_path, errors_map, self.args['num_slices'], self.slice_size, \
                        sample_volume=False, verbose=self.args['verbose'])
                else:
                    preprocessed = self.load_np_volume(volume_path)
                    preprocessed = np.expand_dims(preprocessed, axis=0)
            except ValueError as e:
                raise e

            print('\nINFO: Predicting cancer for volume no. {}...'.format(i + 1))
            singleton_batch = [[preprocessed, None]]
            feed_dict, _ = self.process_data_into_to_dict(singleton_batch, is_paths=False)
            preds = self.sess.run([self.get_preds], feed_dict=feed_dict)
            print('\nINFO: Probability of cancer within 1 year: {:.5f}\n\n'.format(preds[0][0]))

    def process_data_into_to_dict(self, coupled_batch, is_paths=True, is_training=False):
        volumes = []
        labels = []
        for volume, label in coupled_batch:
            try:
                if is_paths:
                    volume = self.load_np_volume(volume)

                # Crop volume to shape [self.args['num_slices, 224, 224]
                crop_start = volume.shape[0] // 2 - self.args['num_slices'] // 2
                volume = volume[crop_start: crop_start + self.args['num_slices']]
                volumes.append(volume)

                if label is not None:
                    labels.append(label)
            except:
                print('\nERROR! Could not load:', volume)

        # Perform windowing online volume, to save storage space of preprocessed volumes
        volume_batch = np.array(volumes)
        volume_batch = utils.apply_window(volume_batch)

        if labels:
            labels_np = np.array(labels).astype(np.int64)
        else:
            labels_np = np.zeros(volume_batch.shape[0], dtype=np.int64)

        feed_dict = {self.volumes_placeholder: volume_batch, self.labels_placeholder: labels_np, self.is_training_placeholder: is_training}
        return feed_dict, labels

    def load_np_volume(self, volume_file):
        if volume_file.endswith('.npz'):
            scan_arr = np.load(join(self.args['data_dir'], volume_file))['data']
        else:
            scan_arr = np.load(join(self.args['data_dir'], volume_file)).astype(np.float32)
        return scan_arr

def create_output_dirs(args):
    # Create model dir and log dir if they doesn't exist
    timestamp = date.today().strftime("%A_") + strftime("%H:%M:%S")
    out_dir_time = Path(str(args['out_dir']) + '_' + timestamp)
    save_dir = out_dir_time / 'models'
    metrics_dir = out_dir_time / 'metrics'
    val_preds_dir = metrics_dir / 'val_preds'
    tr_preds_dir = metrics_dir / 'tr_preds'
    plots_dir = out_dir_time / 'plots'

    for new_dir in out_dir_time, save_dir, val_preds_dir, tr_preds_dir, plots_dir:
        os.makedirs(new_dir, exist_ok=True)

    return save_dir, metrics_dir, plots_dir

def main(args):
    print('\nINFO: Initializing...')

    # Set GPU
    if args['device'] == 'GPU':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu_id'])

    # Init model wrapper
    model = I3dForCTVolumes(args)

    # Load pre-trained weights
    pre_trained_ckpt = utils.load_pretrained_ckpt(args['ckpt'], args['data_dir'])
    model.pretrained_saver.restore(model.sess, pre_trained_ckpt)

    if args['input']:
        print('\nINFO: Begin Inference \n')
        model.predict(args['input'])
    else:
        print('\nINFO: Begin Training')

        print('\nINFO: Hyperparams:')
        print('\n'.join([str(item) for item in args.items()]))

        save_dir, metrics_dir, plots_dir = create_output_dirs(args)

        train_list = utils.load_data_list(args['train'])
        val_list = utils.load_data_list(args['val'])
        val_labels = utils.get_list_labels(val_list)
        utils.write_number_list(val_labels, join(metrics_dir, 'val_true'), verbose=args['verbose'])

        metrics = defaultdict(list)
        for epoch in range(1, args['epochs'] + 1):
            print('\nINFO: +++++++++++++++++++++ EPOCH {} +++++++++++++++++++++'.format(epoch))
            start_time = time()
            shuffle(train_list)

            # Run training for 1 epoch and save weights to file
            tr_epoch_metrics = model.train_loop(train_list, metrics_dir)
            print("\nINFO: Saving Weights...")
            model.saver.save(model.sess, "{}/epoch_{}/model.ckpt".format(save_dir, epoch))
            
            train_end_time = time()
            print('\nINFO: Train epoch duration: {:.2f} secs'.format(train_end_time - start_time))

            # Run validation at end of each epoch
            print("\nINFO: Begin Validation")
            val_metrics = model.evaluate(val_list)

            print('\nINFO: Val duration: {:.2f} secs'.format(time() - train_end_time))

            print('\nINFO: Writing metrics plotting them...')
            utils.write_metrics(metrics, tr_epoch_metrics, val_metrics, metrics_dir, epoch, verbose=args['verbose'])
            utils.plot_metrics(epoch, metrics_dir, plots_dir)

def train(**kwargs):
    '''
    Run prediction. 
    For arguments description, see General and Training sections in params() function below.
    '''
    final_kwargs = params()
    # Override default parameters with given arguments
    for key, value in kwargs.items():
        final_kwargs[key] = value
    main(final_kwargs)

def predict(**kwargs):
    '''
    Run prediction. 
    For arguments description, see General and Inference sections in params() function below.
    '''
    final_kwargs = params()
    # Override default parameters with given arguments
    for key, value in kwargs.items():
        final_kwargs[key] = value
    main(final_kwargs)

def params():
    parser = argparse.ArgumentParser()

    default_out_dir = Path.home() / 'Lung-Cancer-Risk-Prediction' / 'out'
    default_data_dir = Path.home() / 'Lung-Cancer-Risk-Prediction' / 'data'
    lists_dir = default_data_dir / 'lists'

    ########################################   General parameters #########################################
    parser.add_argument('--ckpt', default='cancer_fine_tuned', type=str, help="pre-trained weights to load. \
        Either 'i3d_imagenet', 'cancer_fine_tuned' or a path to a directory containing model.ckpt file")

    parser.add_argument('--num_slices', default=220, type=int, \
        help='number of slices (z dimension) from the volume to be used by the model')

    parser.add_argument('--verbose', default=False, type=bool, help='whether to print detailed logs')

    ########################################   Training parameters ########################################
    parser.add_argument('--epochs', default=40, type=int,  help='the number of epochs')

    parser.add_argument('--lr', default=0.0001, type=int, help='initial learning rate')

    parser.add_argument('--keep_prob', default=0.8, type=int, help='dropout keep prob')

    parser.add_argument('--batch_size', default=2, type=int, help='the batch size for training/validation')

    parser.add_argument('--gpu_id', default=1, type=int, help='gpu id')

    parser.add_argument('--device', default='GPU', type=str, help='the device to execute on')

    parser.add_argument('--data_dir', default=default_data_dir, \
        help='path to data directory (for raw/processed volumes, train/val lists, checkpoints etc.)')

    parser.add_argument('--train', default=lists_dir / 'train.list', help='path to train data .list file')

    parser.add_argument('--val', default=lists_dir / 'val.list', help='path to validation data .list file')

    parser.add_argument('--out_dir', default=default_out_dir, help='path to output dir for models, metrics and plots')

    ########################################   Inference parameters ########################################
    parser.add_argument('--input', default=None, type=str, help="path to volumes for cancer prediction or 'sample_data' to use included CT samples.")

    parser.add_argument('--preprocessed', default=False, type=bool, help='whether data for inference is \
        preprocessed (.npz files) or raw volumes (dirs of .dcm files)')

    parser.set_defaults()
    args, _ = parser.parse_known_args()
    kwargs = vars(args)
    return kwargs

if __name__ == "__main__":
    kwargs = params()
    main(kwargs)