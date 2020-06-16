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

class I3dForCTVolumes:
    def __init__(self, data_dir, batch_size, is_compressed, is_preprocessed, learning_rate=0.0001, device='GPU', 
                num_slices=220, crop_size=224, verbose=False):
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.num_slices = num_slices
        self.batch_size = batch_size
        self.verbose = verbose
        self.is_compressed = is_compressed
        self.is_preprocessed = is_preprocessed

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
                    num_slices=self.num_slices,
                    crop_size=self.crop_size,
                    rgb_channels=3
                    )
            
            # Learning rate
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=3000, decay_rate=0.1, staircase=True)
            
            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Init I3D model
            with tf.device('/device:' + device + ':0'):
                with tf.compat.v1.variable_scope('RGB'):
                    _, end_points = InceptionI3d(num_classes=2, final_endpoint='Predictions')\
                        (self.images_placeholder, self.is_training_placeholder, dropout_keep_prob=0.8)
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

    def train_loop(self, data_list):
        loss_list, acc_list = [], []
        batches = list(utils.batcher(data_list, self.batch_size))
        for i, list_batch in tqdm(enumerate(batches), file=os.sys.stderr):
            images_batch, labels_batch = self.process_coupled_data(list_batch)
            feed_dict = self.coupled_data_to_dict(images_batch, labels=labels_batch, is_training=True)
            self.sess.run(self.train_op, feed_dict=feed_dict)

            if i % 1 == 0:
                feed_dict = self.coupled_data_to_dict(images_batch, labels=labels_batch, is_training=False)
                acc, loss = self.sess.run([self.accuracy, self.loss], feed_dict=feed_dict)
                loss_list.append(loss)
                acc_list.append(acc)
        
        tr_loss = np.mean(loss_list)
        tr_acc = np.mean(acc_list)
        print("\nINFO: Train accuracy: {:.4f}".format(tr_acc))
        print("\nINFO: Train loss: {:.4f}".format(tr_loss))
        return tr_loss, tr_acc

    def val_loop(self, images, labels):
        acc_list = []
        loss_list = []
        preds_list = []

        image_iterator = utils.batcher(images, self.batch_size)
        label_iterator = utils.batcher(labels, self.batch_size)
        
        print('\nINFO: ++++++++++++++++++++ Validation ++++++++++++++++++++')
        for image_batch, label_batch in tqdm(list(zip(image_iterator, label_iterator)), file=os.sys.stderr):
            feed_dict = self.coupled_data_to_dict(image_batch, labels=label_batch, is_training=False)
            batch_acc, batch_loss, preds = self.sess.run([self.accuracy, self.loss, self.get_preds], feed_dict=feed_dict)
            acc_list.append(batch_acc)
            loss_list.append(batch_loss)
            preds_list.extend(preds)

        if self.verbose:
            print('\nDEBUG: Val. preds: ', preds_list)
            print('\nDEBUG: Val. labels: ', labels)
            print('\nDEBUG: Val Batch accuracy: ', acc_list)
            print('\nDEBUG: Val Batch Loss: ', loss_list)

        mean_acc = np.mean(acc_list)
        mean_loss = np.mean(loss_list)
        auc_score = roc_auc_score(labels, preds_list)

        print('\n' + '=' * 34)
        print("||  INFO: Val Accuracy: {:.4f} ||".format(mean_acc))
        print("||  INFO: Val Loss:     {:.4f} ||".format(mean_loss))
        print("||  INFO: Val AUC:      {:.4f} ||".format(auc_score))
        print('=' * 34)
        return mean_loss, mean_acc, auc_score, preds_list

    def predict(self, inference_data):
        errors_map = {}
        img_iterator = walk_np_files(inference_data) if self.is_preprocessed else walk_dicom_dirs(inference_data)

        for image_path in tqdm(img_iterator, file=os.sys.stderr):
            try:
                if not self.is_preprocessed:
                    print('\nINFO: Preprocessing image...')
                    preprocessed, _  = preprocess_old(image_path, errors_map)
                    # preprocessed, _ = preprocess(image_path, errors_map, self.num_slices, self.crop_size, \
                        # sample_img=False, verbose=self.verbose)
                else:
                    preprocessed = self.load_np_image(image_path)
            except ValueError as e:
                print(e)
            print('\nINFO: Predicting...')

            preprocessed = self.crop_image(preprocessed)
            preprocessed = np.expand_dims(preprocessed, axis=0)
            feed_dict = self.coupled_data_to_dict(preprocessed)
            preds = self.sess.run([self.get_preds], feed_dict=feed_dict)
            print('\nINFO: Probability of cancer within 1 year: {}\n\n'.format(preds[0][0]))

    def coupled_data_to_dict(self, images, labels=None, is_training=False):
        # Perform online windowing of image, to save storage space of preprocessed images
        if self.is_compressed:
            images = utils.apply_window(images)

        # DEBUG - TODO: Remove this
        # print('image max val, min val: ', np.max(images), np.min(images))
        # img = images[0]
        # plt.imshow(img[img.shape[0] // 2])
        # plt.savefig('debug/' + str(time()) + '.png', bbox_inches='tight')

        feed_dict = {self.images_placeholder: images, self.is_training_placeholder: is_training}
        if labels is not None:
                feed_dict[self.labels_placeholder] = labels
        return feed_dict

    def crop_image(self, scan_arr):
        if self.is_compressed:
            crop_start = scan_arr.shape[0] // 2 - self.num_slices // 2
            image = scan_arr[crop_start: crop_start + self.num_slices]
        else:
            try:
                image = np.zeros((self.num_slices, self.crop_size, self.crop_size, 3)).astype(np.float32)
                image[:scan_arr.shape[0], :scan_arr.shape[1], :scan_arr.shape[2], :3] = \
                    scan_arr[:self.num_slices, :self.crop_size, :self.crop_size, :3]

            except Exception as e:
                # TODO: filter images which are too small
                print("\nERROR Loading image with shape {}".format(scan_arr.shape))
                print(e)
        return image

    def load_np_image(self, img_file):
        if self.is_compressed:
            scan_arr = np.load(join(self.data_dir, img_file))['data']
        else:
            scan_arr = np.load(join(self.data_dir, img_file)).astype(np.float32)
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
    os.makedirs(plots_dir, exist_ok=True)
    return save_dir, metrics_dir, plots_dir

def main(args):
    print('\nINFO: Initializing...')

    # Set GPU
    if args.device == 'GPU':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Init model wrapper
    model = I3dForCTVolumes(data_dir=args.data_dir, batch_size=args.batch_size, is_compressed=args.is_compressed, \
        is_preprocessed=args.is_preprocessed, device=args.device, num_slices=args.num_slices, verbose=args.verbose)

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
        utils.write_number_list(val_labels, join(metrics_dir, 'val_true'), verbose=model.verbose)
        metrics = {'tr_loss': [], 'tr_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
        
        for epoch in range(1, args.epochs + 1):
            print('\nINFO: +++++++++++++++++++++ EPOCH {} +++++++++++++++++++++'.format(epoch))
            start_time = time()
            shuffle(train_list)

            # Run training for 1 epoch
            tr_epoch_metrics = model.train_loop(train_list)

            # Save Weights after each epoch
            print("\nINFO: Saving Weights...")
            model.saver.save(model.sess, "{}/epoch_{}/model.ckpt".format(save_dir, epoch))
            
            train_end_time = time()
            print('\nINFO: Train epoch duration: {:.2f} secs'.format(train_end_time - start_time))

            # Run validation at end of each epoch
            print("\nINFO: Begin Validation")
            val_metrics = model.val_loop(val_images, val_labels)

            print('\nINFO: Val duration: {:.2f} secs'.format(time() - train_end_time))

            print('\nINFO: Writing metrics and their plots...')
            utils.write_metrics(metrics, tr_epoch_metrics, val_metrics, metrics_dir, epoch, verbose=model.verbose)
            utils.plot_metrics(epoch, metrics_dir, plots_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################################
    EPOCHS = 60
    BATCH = 3
    # DEBUG = 'ra_'
    DEBUG = ''
    GPU = 0
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

    #####################################################################################################################################
    parser.add_argument('--ckpt', default='epoch_70', type=str, help='path to previously saved model to load')

    # parser.add_argument('--inference', default='/workdisk/Lung-Cancer-Risk-Prediction/data/datasets/NLST_preprocessed/conflc=confirmed', \
    #     type=str, help='path to scan for cancer prediction')
    parser.add_argument('--inference', default=None, \
        type=str, help='path to scan for cancer prediction')

    parser.add_argument('--verbose', default=True, type=bool, help='whether to print detailed logs')

    parser.add_argument('--is_compressed', default=False, type=bool, \
        help='whether preprocessed data is compressed (unwindowed, npz), or uncompressed (windowed, npy)')

    parser.add_argument('--is_preprocessed', default=False, type=bool, \
        help='whether data for inference is preprocessed np files or raw DICOM dirs')

    parser.add_argument('--num_slices', default=140, type=int, help='number of slices (z dimension) used by the model')

    parser.set_defaults()
    main(parser.parse_args())