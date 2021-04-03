'''A wrapper class for scheduled optimizer '''
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer):
        self._optimizer = optimizer


    def step_lr(self):
        "Step with the inner optimizer"
        self._optimizer.step()

    def update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()

    def _update_learning_rate(self):
        ''' Learning rate scheduling '''
        for param_group in self._optimizer.param_groups:
            # print('before', param_group['lr'])
            lr = param_group['lr'] * 0.8
            # print('after', param_group['lr'])
            param_group['lr'] = lr