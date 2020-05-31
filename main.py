# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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
# ******************************************************************************
from __future__ import division
from __future__ import print_function
from random import shuffle
import os
import numpy as np
import utils
from utils import batcher 
import argparse
import tensorflow as tf
from i3d import InceptionI3d
from os.path import join, dirname, realpath
from time import time

def main(args):
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Create model dir if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = args.model_dir

    # Create train and dev sets
    print("Creating training and validation sets")

    if not args.debug:
        train_list, test_list = args.train, args.test
    else:
        train_list, test_list = 'debug_' + args.train, 'debug_' + args.test

    train_list_path = join(args.data_dir, 'lists', train_list)
    test_list_path = join(args.data_dir, 'lists', test_list)
    
    train_list = utils.load_data_list(train_list_path)
    val_list = utils.load_data_list(test_list_path)

    model = I3dForCTVolumes(data_dir=args.data_dir, batch_size=args.batch_size, device=args.select_device)

    # Intitialze with pretrained weights
    print('\nINFO: Loading from previously stored session \n')
    # pretrained_saver.restore(sess, model_ckpt.model_checkpoint_path)
    model.pretrained_saver.restore(model.sess, join(args.data_dir, args.ckpt))
    print('\nINFO: Loaded pretrained model')

    val_images, val_labels = model.process_coupled_data(val_list)

    if args.inference_mode:
        print('\nINFO: Begin Inference Mode \n')
        # Shuffle Validation Set
        shuffle(dev)
        # Run Inference Mode
        model.inference_mode(sess, dev, [vocab_dict, vocab_rev],
                            num_examples=args.num_examples, dropout=1.0)
    else:
        print('\nINFO: Begin Training \n')

        for epoch in range(args.epochs):
            print("\nINFO: +++++++++++++++++++++ EPOCH ", epoch + 1)
            start_time = time()

            # Shuffle Dataset
            shuffle(train_list)

            # Run training for 1 epoch
            model.train_loop(train_list)

            # Save Weights after each epoch
            # print("\nINFO: Saving Weights")
            # model_saver.save(sess, "{}/trained_model_{}.ckpt".format(model_path, epoch))
            
            train_end_time = time()
            print('\nINFO: Train Epoch duration: {:.2f} secs'.format(train_end_time - start_time))

            # Start validation phase at end of each epoch
            print("\nINFO: Begin Validation")
            model.val_loop(val_images, val_labels)
            print('\nINFO: Val duration: {:.2f} secs'.format(time() - train_end_time))



class I3dForCTVolumes:
    def __init__(self, data_dir, batch_size, learning_rate=0.0001, device='GPU', num_frames=140, crop_size=224):
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.batch_size = batch_size

        with tf.Graph().as_default():
            global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )

            # Placeholders
            self.images_placeholder, self.labels_placeholder, self.is_training_placeholder = utils.placeholder_inputs(
                    batch_size=self.batch_size,
                    num_frame_per_clip=self.num_frames,
                    crop_size=self.crop_size,
                    rgb_channels=3
                    )

            # self.lr = tf.train.exponential_decay(learning_rate, self.global_step, decay_steps=3000, decay_rate=0.1, staircase=True)
            
            # Learning rate
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=3000, decay_rate=0.1, staircase=True)
            
            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate) #ORIGINAL OPTIMIZER
            # opt_rgb = tf.train.AdamOptimizer()

            # Init I3D model
            with tf.device('/device:' + device + ':0'):
                with tf.variable_scope('RGB'):
                    self.logits, _ = InceptionI3d(num_classes=2, final_endpoint='Predictions')(self.images_placeholder, self.is_training_placeholder)

                # Loss function
                self.loss = utils.cross_entropy_loss(self.logits, self.labels_placeholder)

                # Evaluation metrics
                self.accuracy = utils.accuracy(self.logits, self.labels_placeholder)
                self.auc = tf.metrics.auc(self.labels_placeholder, self.logits[:, 1])

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grads = optimizer.compute_gradients(self.loss)
                    apply_gradient = optimizer.apply_gradients(grads, global_step=global_step)
                    self.train_op = tf.group(apply_gradient)
                    null_op = tf.no_op()

            # Create a saver for loading pretrained checkpoints.
            pretrained_variable_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'RGB' and 'Adam' not in variable.name.split('/')[-1] and variable.name.split('/')[2] != 'Logits':
                    pretrained_variable_map[variable.name.replace(':0', '')] = variable
            self.pretrained_saver = tf.train.Saver(var_list=pretrained_variable_map, reshape=True)

            # init = tf.global_variables_initializer()
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            # Create a session for running Ops on the Graph.
            run_config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=run_config)
            self.sess.run(init)

    def train_loop(self, data_list):
        for i, list_batch in enumerate(batcher(data_list, self.batch_size)):
            print('\nINFO: ========== STEP', i + 1)
            images_batch, labels_batch = self.process_coupled_data(list_batch)

            feed_dict = self.coupled_data_to_dict(images_batch, labels_batch, is_training=True)

            self.sess.run(self.train_op, feed_dict=feed_dict)
            # res = self.sess.run([self.train_op, self.accuracy, self.loss], feed_dict=feed_dict)

            if i % 10 == 0:
                feed_dict = self.coupled_data_to_dict(images_batch, labels_batch, is_training=False)
                acc, loss = self.sess.run([self.accuracy, self.loss], feed_dict=feed_dict)
                print("\nINFO Train accuracy: {:.5f}".format(acc))
                print("\nINFO Train loss: {:.5f}".format(loss)) 

    def val_loop(self, images, labels):
        acc_list = []
        auc_list = []
        for image_batch, label_batch in zip(batcher(images, self.batch_size), batcher(labels, self.batch_size)):
            feed_dict = self.coupled_data_to_dict(image_batch, label_batch, is_training=False)
            acc, auc = self.sess.run([self.accuracy, self.auc], feed_dict=feed_dict)
            acc_list.append(acc)
            auc_list.append(auc)
        
        print('\nDEBUG: Batch accuracy: ', acc_list)
        print('\nDEBUG: Batch AUC: ', auc_list)
        mean_acc = np.mean(acc_list)
        # TODO: Is it ok to average AUC over batches?
        mean_auc = np.mean(auc_list)
        print('\n' + '=' * 34)
        print("||  INFO: Val accuracy: {:.5f} ||".format(mean_acc))
        print("||  INFO: Val AUC:      {:.5f} ||".format(mean_auc))
        print('=' * 34)

    def coupled_data_to_dict(self, train_images, train_labels, is_training):
        return {
                self.images_placeholder: train_images,
                self.labels_placeholder: train_labels,
                self.is_training_placeholder: is_training
                }

    def process_coupled_data(self, coupled_data):
        data = []
        labels = []
        for cur_file, label in coupled_data:
            try:
                result = np.zeros((self.num_frames, self.crop_size, self.crop_size, 3)).astype(np.float32)
                print("\INFO: Loading image from {}".format(cur_file))
                scan_arr = np.load(join(self.data_dir, cur_file)).astype(np.float32)
                print('orig shape:', scan_arr.shape)
                result[:self.num_frames, :scan_arr.shape[1], :scan_arr.shape[2], :3] = \
                    scan_arr[:self.num_frames, :self.crop_size, :self.crop_size, :3]
            except Exception as e:
                print("\ERROR Loading image from {} with shape {}".format(cur_file, scan_arr.shape))
                raise e

            data.append(result)
            labels.append(label)
        np_arr_data = np.array(data)
        np_arr_labels = np.array(labels).astype(np.int64).reshape(len(coupled_data))
        return np_arr_data, np_arr_labels


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=str(dirname(realpath(__file__))) + '/data', help='path to training data')

    parser.add_argument('--debug', default=True, type=bool, help='whether to run debug dataset')

    parser.add_argument('--train', default='train.list', help='path to training data')

    parser.add_argument('--test', default='test.list', help='path to training data')

    parser.add_argument('--gpu_id', default="0", type=str, help='gpu id')

    parser.add_argument('--epochs', default=6, type=int,  help='the number of epochs')

    parser.add_argument('--select_device', default='GPU', type=str, help='the device to execute on')

    parser.add_argument('--model_dir', default='trained_model', help='path to save model')

    parser.add_argument('--ckpt', default='checkpoints/inflated/model.ckpt', type=str, help='path to previously saved model to load')

    parser.add_argument('--inference_mode', default=False, type=bool, help='whether to run inference only')

    parser.add_argument('--batch_size', default=3, type=int, help='the training batch size')

    parser.add_argument('--num_examples', default=50, type=int, help='the number of examples to run inference')

    parser.set_defaults()
    main(parser.parse_args())