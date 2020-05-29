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

"""Functions for load train data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import time
from PIL import Image

def sample_data(ori_arr, num_frames_per_clip, sample_rate):
    ret_arr = []
    for i in range(int(num_frames_per_clip/sample_rate)):
        ret_arr.append(ori_arr[int(i*sample_rate)])
    return ret_arr

def get_frames(filename, s_index, num_frames_per_clip, crop_size, sample_rate, add_flow, position=-1):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    # rgb_ret_arr, s_index = get_data(filename, num_frames_per_clip, sample_rate, int(s_index))
    
    # rgb_ret_arr = data_process_pos(rgb_ret_arr, crop_size, int(position))
    return rgb_ret_arr, [], s_index
        
def get_data(filename, num_frames_per_clip, sample_rate, s_index=-1):
    ret_arr = []
    filenames = ''
    for parent, dirnames, filenames in os.walk(filename):
        if len(filenames)==0:
            print('DATA_ERRO: %s'%filename)
            return [], s_index
        if (len(filenames)-s_index) <= num_frames_per_clip:
            filenames = sorted(filenames)
            if len(filenames) < num_frames_per_clip:
                for i in range(num_frames_per_clip):
                    if i >= len(filenames):
                        i = len(filenames)-1
                    image_name = str(filename) + '/' + str(filenames[i])
                    img = Image.open(image_name)
                    img_data = np.array(img)
                    ret_arr.append(img_data)
            else:
                for i in range(num_frames_per_clip):
                    image_name = str(filename) + '/' + str(filenames[len(filenames)-num_frames_per_clip+i])
                    img = Image.open(image_name)
                    img_data = np.array(img)
                    ret_arr.append(img_data)
            return sample_data(ret_arr, num_frames_per_clip, sample_rate), s_index
    filenames = sorted(filenames)
    if s_index < 0:
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    for i in range(int(num_frames_per_clip/sample_rate)):
        image_name = str(filename) + '/' + str(filenames[int(i*sample_rate)+s_index])
        img = Image.open(image_name)
        img_data = np.array(img)
        ret_arr.append(img_data)
    return ret_arr, s_index

def read_scan_and_label(base_dir, file_list, batch_size, start_pos, crop_size=198, num_frames=197):
    with open(file_list) as file_list_fp:
        lines = [base_dir + '/' + path.strip() for path in file_list_fp]
    read_files = []
    rgb_data = []
    label = []
    batch_index = 0
    next_batch_start = -1

    for index in range(start_pos, min(len(lines), start_pos + batch_size)):
        cur_file, tmp_label = lines[index].split()
        print("Loading a video clip from {}...".format(cur_file))

        result = np.zeros((num_frames, crop_size, crop_size, 3)).astype(np.float32)
        scan_arr = np.load(cur_file).astype(np.float32)
        result[:num_frames, :scan_arr.shape[1], :scan_arr.shape[2], :3] = scan_arr[:num_frames, :crop_size, :crop_size, :3]

        if len(result) != 0:
            rgb_data.append(result)
            label.append(int(tmp_label))
            batch_index += 1
            read_files.append(cur_file)

    # pad (duplicate) data/label if less than batch_size
    valid_len = len(rgb_data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            rgb_data.append(rgb_data[-1])
            label.append(int(label[-1]))

    np_arr_rgb_data = np.array(rgb_data)
    np_arr_label = np.array(label).astype(np.int64).reshape(batch_size)

    return np_arr_rgb_data, np_arr_label #, next_batch_start, read_files, valid_len

# def read_clip_and_label_old(base_dir, file_list, batch_size, start_pos=-1, crop_size=198, num_frames=197, shuffle=True):
#     lines = [base_dir + '/' + path for path in open(file_list, 'r').readlines()]
#     read_files = []
#     rgb_data = []
#     label = []
#     batch_index = 0
#     next_batch_start = -1
#     lines = list(lines)
#     # lines = filename
#     # Forcing shuffle, if start_pos is not specified
#     if start_pos < 0:
#         shuffle = True
#     if shuffle:
#         video_indices = list(range(len(lines)))
#         random.seed(time.time())
#         random.shuffle(video_indices)
#     else:
#         # Process videos sequentially
#         video_indices = range(start_pos, len(lines))
#     for index in video_indices:
#         if batch_index >= batch_size:
#             next_batch_start = index
#             break
#         line = lines[index].strip('\n').split()
#         cur_file = line[0]
#         tmp_label = line[1]
#         if not shuffle:
#             print("Loading a video clip from {}...".format(cur_file))
#         # tmp_rgb_data = [Image.fromarray(slc.astype(np.uint8)) for slc in np.load(cur_file)]
#         # tmp_rgb_data = [slc for slc in np.load(cur_file).astype(np.float32)]

#         result = np.zeros((num_frames, crop_size, crop_size, 3)).astype(np.float32)
#         scan_arr = np.load(cur_file).astype(np.float32)
#         result[:num_frames, :scan_arr.shape[1], :scan_arr.shape[2], :3] = scan_arr[:num_frames, :crop_size, :crop_size, :3]
#         # video_list = np.stack(padded_scans).astype(np.float32)

#         if len(result) != 0:
#             rgb_data.append(result)
#             label.append(int(tmp_label))
#             batch_index += 1
#             read_files.append(cur_file)

#     # pad (duplicate) data/label if less than batch_size
#     valid_len = len(rgb_data)
#     pad_len = batch_size - valid_len
#     if pad_len:
#         for i in range(pad_len):
#             rgb_data.append(rgb_data[-1])
#             label.append(int(label[-1]))

#     np_arr_rgb_data = np.array(rgb_data)#.astype(np.float32)
#     np_arr_label = np.array(label).astype(np.int64).reshape(batch_size)

#     return np_arr_rgb_data, np_arr_label #, next_batch_start, read_files, valid_len


# def data_process_pos(tmp_data, crop_size, position):
#     img_datas = []
#     crop_x = 0
#     crop_y = 0
#     for j in xrange(len(tmp_data)):
#         img = Image.fromarray(tmp_data[j].astype(np.uint8))
#         if img.width > img.height:
#             scale = float(256) / float(img.height)
#             img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), 256))).astype(np.float32)
#             if j == 0:
#                 if position==-1:
#                     crop_x = random.randint(0, int(img.shape[0] - crop_size))
#                     crop_y = random.randint(0, int(img.shape[1] - crop_size))
#                 elif position==0:
#                     crop_x = int((img.shape[0] - crop_size) / 2)
#                     crop_y = 0
#                 elif position==1:
    #                 crop_x = int((img.shape[0] - crop_size) / 2)
    #                 crop_y = int((img.shape[1] - crop_size) / 2)
    #             else:
    #                 crop_x = int((img.shape[0] - crop_size) / 2)
    #                 crop_y = int(img.shape[1] - crop_size)
    #     else:
    #         scale = float(256) / float(img.width)
    #         img = np.array(cv2.resize(np.array(img), (256, int(img.height * scale + 1)))).astype(np.float32)
    #         if j == 0:
    #             if position==-1:
    #                 crop_x = random.randint(0, int(img.shape[0] - crop_size))
    #                 crop_y = random.randint(0, int(img.shape[1] - crop_size))
    #             elif position==0:
    #                 crop_x = 0
    #                 crop_y = int((img.shape[1] - crop_size) / 2)
    #             elif position==1:
    #                 crop_x = int((img.shape[0] - crop_size) / 2)
    #                 crop_y = int((img.shape[1] - crop_size) / 2)
    #             else:
    #                 crop_x = int(img.shape[0] - crop_size)
    #                 crop_y = int((img.shape[1] - crop_size) / 2)

    #     img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
    #     img_datas.append(img)
    # return img_datas

