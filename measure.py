
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_acc(labels, x_labels, nb_batch):
    """ compute the classification accuracy
    Args:
        labels: ground truth label
        x_labels: predicted label
        nb_batch: batch size
    """
    N = int(nb_batch * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(nb_batch):
        for j in range(x_labels.shape[1]): # x_labels.shape: [bs, 10, 29]
            pre_labels[c] = np.argmax(x_labels[i, j, :]) #
            real_labels[c] = np.argmax(labels[i, j, :])
            c += 1
    target_names = []
    for i in range(29):
        target_names.append("class" + str(i))

    return accuracy_score(real_labels, pre_labels)



def AVPSLoss(av_simm, soft_label):
    """audio-visual pair similarity loss for fully supervised setting,
    please refer to Eq.(8, 9) in our paper.
    """
    # av_simm: [bs, 10]
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss
