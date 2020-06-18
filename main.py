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
from i3d import InceptionI3d
from os.path import join, dirname, realpath
from time import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from preprocess import preprocess, walk_dicom_dirs, walk_np_files
import utils
from time import strftime
from datetime import date
from collections import defaultdict

class I3dForCTVolumes:
    def __init__(self, args):
        self.args = args

        # This is the shape of both dimensions of each slice of the volume.
        # The final volume shape fed to the model is [self.args.num_slices, 224, 224]
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
            self.images_placeholder, self.labels_placeholder, self.is_training_placeholder = utils.placeholder_inputs(
                    num_slices=self.args.num_slices,
                    crop_size=self.slice_size,
                    rgb_channels=3
                    )
            
            # Learning rate and optimizer
            # learning_rate = tf.train.exponential_decay(self.args.lr, global_step, decay_steps=3000, decay_rate=0.1, staircase=True)
            optimizer = tf.train.AdamOptimizer(self.args.lr)

            # Init I3D model
            with tf.device('/device:' + self.args.device + ':0'):
                with tf.compat.v1.variable_scope('RGB'):
                    _, end_points = InceptionI3d(num_classes=2, final_endpoint='Predictions')\
                        (self.images_placeholder, self.is_training_placeholder, dropout_keep_prob=args.keep_prob)
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

    def batches(self, images, labels):
        image_batches = utils.batcher(images, self.args.batch_size)
        label_batches = utils.batcher(labels, self.args.batch_size)
        return list(zip(image_batches, label_batches))

    def train_loop(self, data_list, metrics_dir):
        images, labels = self.process_coupled_data(data_list)
        utils.write_number_list(labels, join(metrics_dir, 'tr_true'), verbose=self.args.verbose)

        coupled_batches = self.batches(images, labels)
        for image_batch, label_batch in tqdm(coupled_batches):
            feed_dict = self.coupled_data_to_dict(image_batch, label_batch, is_training=True)
            self.sess.run(self.train_op, feed_dict=feed_dict)
            
        mean_loss, mean_acc, auc, preds_list = self.evaluate(images, labels, ds='Train')
        return mean_loss, mean_acc, auc, preds_list

    def evaluate(self, images, labels, ds='Val.'):
        coupled_batches = self.batches(images, labels)
        loss_list, acc_list, preds_list = [], [], []
        
        print('\nINFO: ++++++++++++++++++++ Validation ++++++++++++++++++++')
        for image_batch, label_batch in tqdm(coupled_batches):
            feed_dict = self.coupled_data_to_dict(image_batch, label_batch, is_training=False)
            acc, loss, preds = self.sess.run([self.accuracy, self.loss, self.get_preds], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)
            preds_list.extend(preds)

        if self.args.verbose:
            print('\nDEBUG: {}. Preds/Labels: {}'.format(ds, zip(preds_list, labels)))
            print('\nDEBUG: {} Batch accuracy/loss: {}'.format(ds, zip(acc_list, loss_list)))

        mean_acc = np.mean(acc_list)
        mean_loss = np.mean(loss_list)
        auc_score = roc_auc_score(labels, preds_list)
        print('\n' + '=' * 34)
        print("||  INFO: {} Accuracy: {:.4f} ||".format(ds, mean_acc))
        print("||  INFO: {} Loss:     {:.4f} ||".format(ds, mean_loss))
        print("||  INFO: {} AUC:      {:.4f} ||".format(ds, auc_score))
        print('=' * 34)
        return mean_loss, mean_acc, auc_score, preds_list

    def predict(self, inference_data):
        errors_map = defaultdict(int)
        img_iterator = walk_np_files(inference_data) if self.args.is_preprocessed else walk_dicom_dirs(inference_data)
        
        for image_path in tqdm(img_iterator, file=os.sys.stderr):
            try:
                if not self.args.is_preprocessed:
                    print('\nINFO: Preprocessing image...')
                    preprocessed, _ = preprocess(image_path, errors_map, self.args.num_slices, self.slice_size, \
                        sample_img=False, verbose=self.args.verbose)
                else:
                    preprocessed = self.load_np_image(image_path)
            except ValueError as e:
                raise e

            print('\nINFO: Predicting...')
            cropped = self.crop_image(preprocessed)
            singleton_batch = np.expand_dims(cropped, axis=0)
            feed_dict = self.coupled_data_to_dict(singleton_batch)
            preds = self.sess.run([self.get_preds], feed_dict=feed_dict)
            print('\nINFO: Probability of cancer within 1 year: {}\n\n'.format(preds[0][0]))

            preds = self.sess.run([self.get_preds], feed_dict=feed_dict)
            print('\nINFO: 2 Probability of cancer within 1 year: {}\n\n'.format(preds[0][0]))

    def coupled_data_to_dict(self, image_batch, labels=None, is_training=False):
        # Perform online windowing of image, to save storage space of preprocessed images
        image_batch = utils.apply_window(image_batch)
        if labels is None:
            labels = np.zeros_like(self.labels_placeholder)
        feed_dict = {self.images_placeholder: image_batch, self.labels_placeholder: labels, self.is_training_placeholder: is_training}

        return feed_dict

    def crop_image(self, scan_arr):
        crop_start = scan_arr.shape[0] // 2 - self.args.num_slices // 2
        image = scan_arr[crop_start: crop_start + self.args.num_slices]
        return image

    def load_np_image(self, img_file):
        if img_file.endswith('.npz'):
            scan_arr = np.load(join(self.args.data_dir, img_file))['data']
        else:
            scan_arr = np.load(join(self.args.data_dir, img_file)).astype(np.float32)
        return scan_arr

    def process_coupled_data(self, coupled_data, progress=False):
        images = []
        labels = []
        if progress:
            coupled_data = tqdm(coupled_data)
        for img_file, label in coupled_data:
            try:
                scan_arr = self.load_np_image(img_file)
                image = self.crop_image(scan_arr)
                images.append(image)
                labels.append(label)
            except:
                print('ERROR !!! Could not load:', img_file)
                
        np_arr_images = np.array(images)
        np_arr_labels = np.array(labels).astype(np.int64)
        return np_arr_images, np_arr_labels

def create_output_dirs(args):
    # Create model dir and log dir if they doesn't exist
    timestamp = date.today().strftime("%A_") + strftime("%H:%M:%S")
    out_dir_time = args.out_dir + '_' + timestamp

    os.makedirs(out_dir_time, exist_ok=True)
    save_dir = join(out_dir_time, 'models')
    metrics_dir = join(out_dir_time, 'metrics')
    plots_dir = join(out_dir_time, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(join(metrics_dir, 'val_preds'), exist_ok=True)
    os.makedirs(join(metrics_dir, 'tr_preds'), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return save_dir, metrics_dir, plots_dir

def main(args):
    print('\nINFO: Initializing...')

    # Set GPU
    if args.device == 'GPU':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Init model wrapper
    model = I3dForCTVolumes(args)

    print('\nINFO: Hyperparams:')
    print('\n'.join([str(item) for item in vars(args).items()]))

    # Load pretrained weights
    ckpt = join(args.data_dir, args.ckpt if args.inference else args.i3d_ckpt, 'model.ckpt')
    print('\nINFO: Loading pre-trained model:', ckpt)
    model.pretrained_saver.restore(model.sess, ckpt)

    if args.inference:
        print('\nINFO: Begin Inference \n')
        model.predict(args.inference)
    else:
        print('\nINFO: Begin Training')
        
        save_dir, metrics_dir, plots_dir = create_output_dirs(args)

        prefix = join(args.data_dir, 'lists', args.debug)
        train_list = utils.load_data_list(prefix + args.train)
        val_list = utils.load_data_list(prefix + args.test)
    
        print('\nINFO: Loading validation set...')
        val_images, val_labels = model.process_coupled_data(val_list, progress=True)
        utils.write_number_list(val_labels, join(metrics_dir, 'val_true'), verbose=args.verbose)
        metrics = defaultdict(list)
        
        for epoch in range(1, args.epochs + 1):
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
            val_metrics = model.evaluate(val_images, val_labels)

            print('\nINFO: Val duration: {:.2f} secs'.format(time() - train_end_time))

            print('\nINFO: Writing metrics and their plots...')
            utils.write_metrics(metrics, tr_epoch_metrics, val_metrics, metrics_dir, epoch, verbose=args.verbose)
            utils.plot_metrics(epoch, metrics_dir, plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################################
    EPOCHS = 80
    BATCH = 2
    DEBUG = ''
    # DEBUG = '1_'
    GPU = 1
    ##################################################

    parser.add_argument('--epochs', default=EPOCHS, type=int,  help='the number of epochs')

    parser.add_argument('--batch_size', default=BATCH, type=int, help='the training batch size')

    parser.add_argument('--debug', default=DEBUG, type=str, help='prefix of debug train/val sets to run')

    parser.add_argument('--gpu_id', default=GPU, type=int, help='gpu id')

    parser.add_argument('--train', default='train.list', help='path to training data')

    parser.add_argument('--test', default='test.list', help='path to training data')

    parser.add_argument('--data_dir', default=str(dirname(realpath(__file__))) + '/data', help='path to training data')

    parser.add_argument('--out_dir', default=str(dirname(realpath(__file__))) + '/out', help='path to output dir for models, metrics and plots')

    parser.add_argument('--device', default='GPU', type=str, help='the device to execute on')

    parser.add_argument('--i3d_ckpt', default='checkpoints/inflated', type=str, help='path to previously saved model to load')

    parser.add_argument('--ckpt', default='best_model', type=str, help='path to previously saved model to load')

    parser.add_argument('--inference', default=None, type=str, help='path to scan for cancer prediction')

    parser.add_argument('--verbose', default=True, type=bool, help='whether to print detailed logs')

    parser.add_argument('--is_preprocessed', default=True, type=bool, help='whether data for inference is preprocessed np files or raw DICOM dirs')

    parser.add_argument('--num_slices', default=220, type=int, help='number of slices (z dimension) used by the model')

    parser.add_argument('--lr', default=0.00005, type=int, help='initial learning rate')

    parser.add_argument('--keep_prob', default=1.0, type=int, help='dropout keep prob')

    parser.set_defaults()
    main(parser.parse_args())