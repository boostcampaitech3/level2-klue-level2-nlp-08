import numpy as np
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AdamW, Trainer, get_cosine_with_hard_restarts_schedule_with_warmup
import torch

import math
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, TrainingArguments
from sklearn.metrics import confusion_matrix

from metric import *
from model import *
from load_data import *
from tokenizing import *
from equip import *


def get_optimizer(model, config, freezing=False):
    if config['optimizer']=='AdamW':
        optimizer = AdamW(model.parameters(), lr = config['lr'])
    elif freezing:
        optimizer = AdamW(model.parameters(), lr = config['freezing_lr'])

    return optimizer

def get_scheduler(optimizer, config, num_train_steps:int = 0):
    if config['scheduler']=="warmup_cosine":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps=config['num_warmup_steps'],
            num_training_steps=num_train_steps,
            num_cycles=config['num_cycles'])
    return scheduler

def get_loss(name:str=None):
    if name=='focal':
        criterion = FocalLoss()


    return criterion

class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., classes=30, reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.classes = classes

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
