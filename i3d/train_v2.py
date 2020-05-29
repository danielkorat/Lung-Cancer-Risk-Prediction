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

# from nlp_architect.models.matchlstm_ansptr import MatchLSTMAnswerPointer
# from nlp_architect.utils.mrc_utils import (
#     create_squad_training, max_values_squad, get_data_array_squad, create_data_dict)
import argparse
import tensorflow as tf
from i3d import InceptionI3d
# from nlp_architect.utils.io import validate_existing_directory, check_size, validate_parent_exists


def main(args):
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Create a dictionary of all parameters
    params_dict = {}
    params_dict['batch_size'] = args.batch_size
    params_dict['epoch_no'] = args.epochs
    params_dict['inference_only'] = args.inference_mode

    # Create dictionary of filenames
    # file_name_dict = {}
    # file_name_dict['train_para_ids'] = 'train.ids.context'
    # file_name_dict['train_ques_ids'] = 'train.ids.question'
    # file_name_dict['train_answer'] = 'train.span'
    # file_name_dict['val_para_ids'] = 'dev.ids.context'
    # file_name_dict['val_ques_ids'] = 'dev.ids.question'
    # file_name_dict['val_ans'] = 'dev.span'
    # file_name_dict['vocab_file'] = 'vocab.dat'
    # file_name_dict['embedding'] = 'glove.trimmed.300.npz'

    # # Paths to preprcessed files
    # path_gen = args.data_path
    # train_para_ids = os.path.join(path_gen, file_name_dict['train_para_ids'])
    # train_ques_ids = os.path.join(path_gen, file_name_dict['train_ques_ids'])
    # answer_file = os.path.join(path_gen, file_name_dict['train_answer'])
    # val_paras_ids = os.path.join(path_gen, file_name_dict['val_para_ids'])
    # val_ques_ids = os.path.join(path_gen, file_name_dict['val_ques_ids'])
    # val_ans_file = os.path.join(path_gen, file_name_dict['val_ans'])
    # vocab_file = os.path.join(path_gen, file_name_dict['vocab_file'])

    # Create model dir if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = args.model_dir

    # # Create lists for train and validation sets
    # data_train = create_squad_training(train_para_ids, train_ques_ids, answer_file)
    # data_dev = create_squad_training(val_paras_ids, val_ques_ids, val_ans_file)
    # vocab_list = []
    # with open(vocab_file) as f:
    #     for ele in f:
    #         vocab_list.append(ele)
    # vocab_dict = {}
    # vocab_rev = {}

    # for i in range(len(vocab_list)):
    #     vocab_dict[i] = vocab_list[i].strip()
    #     vocab_rev[vocab_list[i].strip()] = i

    # if args.train_set_size is None:
    #     params_dict['train_set_size'] = len(data_train)
    # else:
    #     params_dict['train_set_size'] = args.train_set_size

    # # Combine train and dev data
    # data_total = data_train + data_dev

    # # obtain maximum length of question
    # _, max_question = max_values_squad(data_total)
    # params_dict['max_question'] = max_question

    # # Load embeddings for vocab
    # print('Loading Embeddings')
    # embeddingz = np.load(os.path.join(path_gen, file_name_dict['embedding']))
    # embeddings = embeddingz['glove']

    # Create train and dev sets
    print("Creating training and validation sets")
    train_list = utils.load_data_list(os.path.join(args.data_dir, args.train))
    val_list = utils.load_data_list(os.path.join(args.data_dir, args.test))

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder, is_training = utils.placeholder_inputs(
                batch_size=1,
                num_frame_per_clip=200,
                crop_size=224,
                rgb_channels=3
                )

        # Init I3D model
        with tf.device('/device:' + args.select_device + ':0'):
            with tf.variable_scope('RGB'):
                model = InceptionI3d(num_classes=2)(images_placeholder, is_training)

        # Create a saver for loading pretrained checkpoints.
        pretrained_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB' and 'Adam' not in variable.name.split('/')[-1] and variable.name.split('/')[2] != 'Logits':
                pretrained_variable_map[variable.name.replace(':0', '')] = variable
        pretrained_saver = tf.train.Saver(var_list=pretrained_variable_map, reshape=True)

        placeholders = images_placeholder, labels_placeholder
        
        # Define Configs for training
        run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        # Create session run training
        with tf.Session(config=run_config) as sess:
            # init = tf.global_variables_initializer()

            # Model Saver
            model_saver = tf.train.Saver()
            # model_ckpt = tf.train.get_checkpoint_state(model_path)
            # idx_path = model_ckpt.model_checkpoint_path + ".index" if model_ckpt else ""

            # Intitialze with pretrained weights
            print('\nINFO: Loading from previously stored session \n')
            # pretrained_saver.restore(sess, model_ckpt.model_checkpoint_path)
            pretrained_saver.restore(sess, os.path.join(args.data_dir, args.ckpt))
            print('\nINFO: Loaded pretrained model \n')

            # processed_val = utils.process_coupled_data(val_list, args.data_dir)

            if args.inference_mode:
                print('\nINFO: Begin Inference Mode \n')
                # Shuffle Validation Set
                shuffle(dev)
                # Run Inference Mode
                model.inference_mode(sess, dev, [vocab_dict, vocab_rev],
                                    num_examples=args.num_examples, dropout=1.0)
            else:
                print('\nINFO: Begin Training \n')

                for epoch in range(params_dict['epoch_no']):
                    print("Epoch Number: ", epoch)

                    # Shuffle Dataset
                    shuffle(train_list)

                    # Run training for 1 epoch
                    run_loop(sess, train_list, mode='train', placeholders, dropout=1)

                    # Save Weights after each epoch
                    print("Saving Weights")
                    model_saver.save(sess, "{}/trained_model_{}.ckpt".format(model_path, epoch))

                    # Start validation phase at end of each epoch
                    print("Begin Validation")
                    run_loop(sess, processed_val, placeholders, mode='val')

def run_loop(sess, data_list, mode, placeholders, batch_size=None, dropout=1):
    """
    Function to run training/validation loop and display training loss, F1 & EM scores

    Args:
        session: tensorflow session
        train:   data dictionary for training/validation
        dropout: float value
        mode: 'train'/'val'
    """
    nbatches = int((len(train['para']) / self.batch_size))
    f1_score = 0
    em_score = 0
    # for idx in range(nbatches):
        # Train for all batches
        # start_batch = self.batch_size * idx
        # end_batch = self.batch_size * (idx + 1)
        # if end_batch > len(train['para']):
        #     break

        # # Create feed dictionary
        # feed_dict_qa = {
        #     self.para_ids: np.asarray(train['para'][start_batch:end_batch]),
        #     self.question_ids: np.asarray(train['question'][start_batch:end_batch]),
        #     self.para_length: np.asarray(train['para_len'][start_batch:end_batch]),
        #     self.question_length: np.asarray(train['question_len'][start_batch:end_batch]),
        #     self.labels: np.asarray(train['answer'][start_batch:end_batch]),
        #     self.para_mask: np.asarray(train['para_mask'][start_batch:end_batch]),
        #     self.ques_mask: np.asarray(train['question_mask'][start_batch:end_batch]),
        #     self.dropout: dropout
        # }

    if batch_size is None:
        data = utils.process_coupled_data(data, args.data_dir)

    for image, label in batch(data_list, batch_size)

    # Training Phase
    if mode == 'train':
        _, train_loss, _, logits, labels = session.run(
            [self.optimizer, self.loss, self.learning_rate, self.logits_withsf,
                self.labels], feed_dict=feed_dict_qa)

        if (idx % 20 == 0):
            print('iteration = {}, train loss = {}'.format(idx, train_loss))
            f1_score, em_score = self.cal_f1_score(labels, logits)
            print("F-1 and EM Scores are", f1_score, em_score)

        self.global_step.assign(self.global_step + 1)

    else:
        logits, labels = session.run([self.logits_withsf, self.labels],
                                        feed_dict=feed_dict_qa)

        f1_score_int, em_score_int = self.cal_f1_score(labels, logits)
        f1_score += f1_score_int
        em_score += em_score_int

    # Validation Phase
    if mode == 'val':
        print(
            "Validation F1 and EM scores are",
            f1_score / nbatches,
            em_score / nbatches)


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/daniel_nlp/Lung-Cancer-Risk-Prediction/i3d/data', help='path to training data')

    parser.add_argument('--train', default='debug_train.list', help='path to training data')

    parser.add_argument('--test', default='debug_test.list', help='path to training data')

    parser.add_argument('--gpu_id', default="0", type=str, help='gpu id')

    parser.add_argument('--epochs', default=15, type=int,  help='the number of epochs')

    parser.add_argument('--select_device', default='GPU', type=str, help='the device to execute on')

    parser.add_argument('--model_dir', default='trained_model', help='path to save model')

    parser.add_argument('--ckpt', default='checkpoints/inflated/model.ckpt', type=str, help='path to previously saved model to load')

    parser.add_argument('--inference_mode', default=False, type=bool, help='whether to run inference only')

    parser.add_argument('--batch_size', default=8, type=int, help='the training batch size')

    parser.add_argument('--num_examples', default=50, type=int, help='the number of examples to run inference')

    parser.set_defaults()
    main(parser.parse_args())