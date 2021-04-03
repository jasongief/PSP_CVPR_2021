from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import numpy as np
import random

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataloader import AVE_Fully_Dataset
from fully_model import psp_net
from measure import compute_acc, AVPSLoss
from Optim import ScheduledOptim

import warnings
warnings.filterwarnings("ignore")
import argparse
import pdb


parser = argparse.ArgumentParser(description='Fully supervised AVE localization')

# data
parser.add_argument('--model_name', type=str, default='PSP', help='model name')
parser.add_argument('--dir_video', type=str, default="./data/visual_feature.h5", help='visual features')
parser.add_argument('--dir_audio', type=str, default='./data/audio_feature.h5', help='audio features')
parser.add_argument('--dir_labels', type=str, default='./data/right_labels.h5', help='labels of AVE dataset')

parser.add_argument('--dir_order_train', type=str, default='./data/train_order.h5', help='indices of training samples')
parser.add_argument('--dir_order_val', type=str, default='./data/val_order.h5', help='indices of validation samples')
parser.add_argument('--dir_order_test', type=str, default='./data/test_order.h5', help='indices of testing samples')

parser.add_argument('--nb_epoch', type=int, default=300,  help='number of epoch')
parser.add_argument('--batch_size', type=int, default=128, help='number of batch size')
parser.add_argument('--save_epoch', type=int, default=5, help='number of epoch for saving models')
parser.add_argument('--check_epoch', type=int, default=5, help='number of epoch for checking accuracy of current models during training')
parser.add_argument('--LAMBDA', type=float, default=100, help='weight for balancing losses')
parser.add_argument('--threshold', type=float, default=0.099, help='key-parameter for pruning process')

parser.add_argument('--trained_model_path', type=str, default=None, help='pretrained model')
parser.add_argument('--train', action='store_true', default=False, help='train a new model')


FixSeed = 123
random.seed(FixSeed)
np.random.seed(FixSeed)
torch.manual_seed(FixSeed)
torch.cuda.manual_seed(FixSeed)


def train(args, net_model, optimizer):
    AVEData = AVE_Fully_Dataset(video_dir=args.dir_video, audio_dir=args.dir_audio, label_dir=args.dir_labels,
                        order_dir=args.dir_order_train, batch_size=args.batch_size, status='train')
    nb_batch = AVEData.__len__() // args.batch_size
    print('nb_batch:', nb_batch)
    epoch_l = []
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    for epoch in range(args.nb_epoch):
        net_model.train()

        epoch_loss = 0
        epoch_loss_cls = 0
        epoch_loss_avps = 0
        n = 0
        start = time.time()
        SHUFFLE_SAMPLES = True
        for i in range(nb_batch):
            audio_inputs, video_inputs, labels, segment_label_batch, segment_avps_gt_batch = AVEData.get_batch(i, SHUFFLE_SAMPLES)
            SHUFFLE_SAMPLES = False

            audio_inputs = audio_inputs.cuda()
            video_inputs = video_inputs.cuda()
            labels = labels.cuda()
            segment_label_batch = segment_label_batch.cuda()
            segment_avps_gt_batch = segment_avps_gt_batch.cuda()

            net_model.zero_grad()
            fusion, out_prob, cross_att = net_model(audio_inputs, video_inputs, args.threshold) # shape:

            # out_prob: [bs, 10, 29], score_max: [bs, 29]
            loss_cls = nn.CrossEntropyLoss()(out_prob.permute(0, 2, 1), segment_label_batch) # segment_label_batch: [bs, 10]
            loss_avps = AVPSLoss(cross_att, segment_avps_gt_batch)
            loss = loss_cls + args.LAMBDA * loss_avps
            epoch_loss += loss.cpu().data.numpy()
            epoch_loss_cls += loss_cls.cpu().data.numpy()
            epoch_loss_avps += loss_avps.cpu().data.numpy()

            loss.backward()
            optimizer.step_lr()
            n = n + 1

        SHUFFLE_SAMPLES = True

        if (epoch+1) % 60 == 0 and epoch < 170:
            optimizer.update_lr()

        end = time.time()
        epoch_l.append(epoch_loss)
        labels = labels.cpu().data.numpy()
        x_labels = out_prob.cpu().data.numpy()
        acc = compute_acc(labels, x_labels, nb_batch)

        print("=== Epoch {%s}   lr: {%.6f} | Loss: [{%.4f}] loss_cls: [{%.4f}] | loss_frame: [{%.4f}] | training_acc {%.4f}" \
            % (str(epoch), optimizer._optimizer.param_groups[0]['lr'], (epoch_loss) / n, epoch_loss_cls/n, epoch_loss_avps/n, acc))

        if epoch % args.save_epoch == 0 and epoch != 0:
            val_acc = val(args, net_model)
            print('val accuracy:', val_acc, 'epoch=', epoch)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                print('best val accuracy:', best_val_acc)
                print('best val accuracy: {} ***************************************'.format(best_val_acc))
                # torch.save(net_model, model_name + "_" + str(epoch) + "_fully.pt")
        if epoch % args.check_epoch == 0 and epoch != 0:
            test_acc = test(args, net_model)
            print('test accuracy:', test_acc, 'epoch=', epoch)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                print('best test accuracy: {} ======================================='.format(best_test_acc))
                torch.save(net_model, model_name + "_" + str(epoch) + "_fully.pt")
    print('[best val accuracy]: ', best_val_acc)
    print('[best test accuracy]: ', best_test_acc)


def val(args, net_model):
    net_model.eval()
    AVEData = AVE_Fully_Dataset(video_dir=args.dir_video, audio_dir=args.dir_audio, label_dir=args.dir_labels,
                        order_dir=args.dir_order_val, batch_size=402, status='val')
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels, _, _ = AVEData.get_batch(0)

    audio_inputs = audio_inputs.cuda()
    video_inputs = video_inputs.cuda()
    labels = labels.cuda()

    fusion, out_prob, cross_att = net_model(audio_inputs, video_inputs, args.threshold)

    labels = labels.cpu().data.numpy()
    x_labels = out_prob.cpu().data.numpy()
    acc = compute_acc(labels, x_labels, nb_batch)

    print('[val]acc: ', acc)
    return acc



def test(args, net_model, model_path=None):
    if model_path is not None:
        net_model = torch.load(model_path)
        print(">>> [Testing] Load pretrained model from " + model_path)


    net_model.eval()
    AVEData = AVE_Fully_Dataset(video_dir=args.dir_video, audio_dir=args.dir_audio, label_dir=args.dir_labels,
                         order_dir=args.dir_order_test, batch_size=402, status='test')
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels, _, _, = AVEData.get_batch(0)

    audio_inputs = audio_inputs.cuda()
    video_inputs = video_inputs.cuda()
    labels = labels.cuda()

    fusion, out_prob, cross_att = net_model(audio_inputs, video_inputs, args.threshold)

    labels = labels.cpu().data.numpy()
    x_labels = out_prob.cpu().data.numpy()
    acc = compute_acc(labels, x_labels, nb_batch)
    print('[test]acc: ', acc)

    return acc



if __name__ == "__main__":
    args = parser.parse_args()
    print("args: ", args)

    # model and optimizer
    model_name = args.model_name
    if model_name == "PSP":
        net_model = psp_net(128, 512, 128, 29)
    else:
        raise NotImplementedError
    net_model.cuda()
    optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
    optimizer = ScheduledOptim(optimizer)

    if args.train:
        train(args, net_model, optimizer)
    else:
        test_acc = test(args, net_model, model_path=args.trained_model_path)
        print("[test] accuracy: ", test_acc)

