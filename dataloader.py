"""AVE dataset"""
import numpy as np
import torch
import h5py
import pickle
import random
from itertools import product
import os
import pdb

ave_dataset = ['bell', 'Male', 'Bark', 'aircraft', 'car', 'Female', 'Helicopter',
    'Violin', 'Flute', 'Ukulele', 'Fry food', 'Truck', 'Shofar', 'Motorcycle',
    'guitar', 'Train', 'Clock', 'Banjo', 'Goat', 'Baby', 'Bus',
    'Chainsaw', 'Cat', 'Horse', 'Toilet', 'Rodents', 'Accordion', 'Mandolin', 'background']
STANDARD_AVE_DATASET = ['Church bell', 'Male speech, man speaking', 'Bark', 'Fixed-wing aircraft, airplane', 'Race car, auto racing', \
                    'Female speech, woman speaking', 'Helicopter', 'Violin, fiddle', 'Flute', 'Ukulele', 'Frying (food)', 'Truck', 'Shofar', \
                    'Motorcycle', 'Acoustic guitar', 'Train horn', 'Clock', 'Banjo', 'Goat', 'Baby cry, infant cry', 'Bus', 'Chainsaw',\
                    'Cat', 'Horse', 'Toilet flush', 'Rodents, rats, mice', 'Accordion', 'Mandolin']

class AVE_Fully_Dataset(object):
    """Data preparation for fully supervised setting.
    """
    def __init__(self, video_dir, audio_dir, label_dir, order_dir, batch_size, status):

        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.status = status

        with h5py.File(audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:] # shape: (4143, 10, 128)
        with h5py.File(label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:] # shape: (4143, 10, 29)
        with h5py.File(video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:] # shape: (4143, 10, 7, 7, 512)
        print('>> visual feature: ', self.video_features.shape)
        print('>> audio feature: ', self.audio_features.shape)

        with h5py.File(order_dir, 'r') as hf:
            order = hf['order'][:] # list, lenth=3339

        self.lis = order.tolist() # the index of training samples.
        self.list_copy = self.lis.copy().copy()

        self.video_batch = np.float32(np.zeros([self.batch_size, 10, 7, 7, 512]))
        self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        self.pos_audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        self.label_batch = np.float32(np.zeros([self.batch_size, 10, 29]))
        self.segment_label_batch = np.float32(np.zeros([self.batch_size, 10]))
        self.segment_avps_gt_batch = np.float32(np.zeros([self.batch_size, 10]))

    def get_segment_wise_relation(self, batch_labels):
        # batch_labels: [bs, 10, 29]
        bs, seg_num, category_num = batch_labels.shape
        all_seg_idx = list(range(seg_num))
        for i in range(bs):
            col_sum = np.sum(batch_labels[i].T, axis=1)
            category_bg_cols = col_sum.nonzero()[0].tolist()
            category_bg_cols.sort() # [category_label_idx, 28(background_idx, optional)]

            category_col_idx = category_bg_cols[0]
            category_col = batch_labels[i, :, category_col_idx]
            same_category_row_idx = category_col.nonzero()[0].tolist()
            if len(same_category_row_idx) != 0:
                self.segment_avps_gt_batch[i, same_category_row_idx] = 1 / (len(same_category_row_idx))

        for i in range(bs):
            row_idx, col_idx = np.where(batch_labels[i] == 1)
            self.segment_label_batch[i, row_idx] = col_idx


    def __len__(self):
        return len(self.lis)


    def get_batch(self, idx, shuffle_samples=False):
        if shuffle_samples:
            random.shuffle(self.list_copy)
        select_ids = self.list_copy[idx * self.batch_size : (idx + 1) * self.batch_size]

        for i in range(self.batch_size):
            id = select_ids[i]
            v_id = id
            self.video_batch[i, :, :, :, :] = self.video_features[v_id, :, :, :, :]
            self.audio_batch[i, :, :] = self.audio_features[id, :, :]
            self.label_batch[i, :, :] = self.labels[id, :, :]

        self.get_segment_wise_relation(self.label_batch)


        return torch.from_numpy(self.audio_batch).float(), \
                torch.from_numpy(self.video_batch).float(), \
                torch.from_numpy(self.label_batch).float(), \
                torch.from_numpy(self.segment_label_batch).long(), \
                torch.from_numpy(self.segment_avps_gt_batch).float(), \




class AVE_Weakly_Dataset(object):
    """Data preparation for weakly supervised setting.
    """
    def __init__(self, video_dir, video_dir_bg, audio_dir, audio_dir_bg, label_dir, prob_label_dir, label_dir_bg, label_dir_gt, order_dir, batch_size, status='train'):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.video_dir_bg = video_dir_bg
        self.audio_dir_bg = audio_dir_bg
        self.status = status
        self.batch_size = batch_size
        with h5py.File(order_dir, 'r') as hf:
            train_l = hf['order'][:] # lenth: 3339, array
        self.lis = train_l
        self.list_copy = self.lis.copy().copy().tolist()

        with h5py.File(audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:] # (4143, 10, 128)
        with h5py.File(label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:] # (4143, 29)
        with h5py.File(prob_label_dir, 'r') as hf:
            self.prob_labels = hf['avadataset'][:] # (4143, 29)
        with h5py.File(video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:] # (4143, 10, 7, 7, 512)
            self.video_features = self.video_features[train_l, :, :]
        print('video_features.shape', self.video_features.shape)

        self.audio_features = self.audio_features[train_l, :, :] # 3339
        self.labels = self.labels[train_l, :]
        self.prob_labels = self.prob_labels[train_l, :]

        if status == "train":
            with h5py.File(label_dir_bg, 'r') as hf:
                self.negative_labels = hf['avadataset'][:] # negative, shape (178, 29)
            with h5py.File(audio_dir_bg, 'r') as hf:
                self.negative_audio_features = hf['avadataset'][:] # shape:[178, 10, 128]
            with h5py.File(video_dir_bg, 'r') as hf:
                self.negative_video_features = hf['avadataset'][:] # shape: (178, 10, 7, 7, 512)
            ng_num = self.negative_audio_features.shape[0]

            size = self.audio_features.shape[0] + self.negative_audio_features.shape[0]
            audio_train_new = np.zeros((size, self.audio_features.shape[1], self.audio_features.shape[2]))
            audio_train_new[0:self.audio_features.shape[0], :, :] = self.audio_features
            audio_train_new[self.audio_features.shape[0]:size, :, :] = self.negative_audio_features
            self.audio_features = audio_train_new

            video_train_new = np.zeros((size, 10, 7, 7, 512))
            video_train_new[0:self.video_features.shape[0], :, :] = self.video_features
            video_train_new[self.video_features.shape[0]:size, :, :] = self.negative_video_features
            self.video_features = video_train_new

            y_train_new = np.zeros((size, 29))
            y_train_new[0:self.labels.shape[0], :] = self.labels
            y_train_new[self.labels.shape[0]:size, :] = self.negative_labels
            self.labels = y_train_new

            prob_y_train_new = np.zeros((size, 29))
            prob_y_train_new[0:self.prob_labels.shape[0], :] = self.prob_labels
            prob_y_train_new[self.prob_labels.shape[0]:size, :] = self.negative_labels
            self.prob_labels = prob_y_train_new
            self.list_copy.extend(list(range(8000, 8000+ng_num, 1)))
        else: # testing, label for each video segment is known
            with h5py.File(label_dir_gt, 'r') as hf:
                self.labels = hf['avadataset'][:]
                self.labels = self.labels[train_l, :, :]

        self.video_batch = np.float32(np.zeros([self.batch_size, 10, 7, 7, 512]))
        self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        if status == "train":
            self.label_batch = np.float32(np.zeros([self.batch_size, 29])) # weak supervised, only have access to the event level tag.
            self.prob_label_batch = np.float32(np.zeros([self.batch_size, 29])) # weak supervised, only have access to the event level tag.
        else:
            self.label_batch = np.float32(np.zeros([self.batch_size,10, 29])) # during testing, segment label should be predicted.

    def __len__(self):
        return len(self.labels)

    def get_batch(self, idx, shuffle_samples=False):
        self.list_copy_copy = self.list_copy.copy().copy()
        if shuffle_samples:
            random.shuffle(self.list_copy)
        select_ids = self.list_copy[idx * self.batch_size : (idx + 1) * self.batch_size]

        for i in range(self.batch_size):
            id = select_ids[i]
            real_id = self.list_copy_copy.index(id)
            self.video_batch[i, :, :, :, :] = self.video_features[real_id, :, :, :, :] # [10, 7, 7, 512]
            self.audio_batch[i, :, :] = self.audio_features[real_id, :, :] #[10, 128]
            if self.status == "train":
                self.label_batch[i, :] = self.labels[real_id, :] # [1, 29] one-hot
                self.prob_label_batch[i, :] = self.prob_labels[real_id, :] # [1, 29] normalized label
            else:
                self.label_batch[i, :, :] = self.labels[real_id, :, :]


        if self.status == 'train':
            return torch.from_numpy(self.audio_batch).float(), \
                    torch.from_numpy(self.video_batch).float(), \
                    torch.from_numpy(self.label_batch).float(), \
                    torch.from_numpy(self.prob_label_batch).float()
        else:
            return torch.from_numpy(self.audio_batch).float(), \
                torch.from_numpy(self.video_batch).float(), \
                torch.from_numpy(self.label_batch).float()


