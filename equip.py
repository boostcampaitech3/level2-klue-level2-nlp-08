import numpy as np
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AdamW, Trainer, get_cosine_with_hard_restarts_schedule_with_warmup
import torch

import math

from sklearn.model_selection import train_test_split
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, TrainingArguments
from sklearn.metrics import confusion_matrix

from metric import *
from model import *
from load_data import *
from tokenizing import *
from equip import *

import matplotlib.pyplot as plt
import seaborn as sns
import wandb, os, random


def get_optimizer(model):
    optimizer = AdamW(model.parameters(), lr = 3e-3)

    return optimizer

def get_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=0)
    return scheduler

def get_loss():
    criterion = FocalLoss()
    return criterion

class MyTrainer(Trainer):
    def __init__(self, loss, optimizers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # config에 저장된 loss_name에 따라 다른 loss 계산

        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = self.criterion(outputs[0], labels)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        no_decay = ["bias", "LayerNorm.weight"]
        # Add any new parameters to optimize for here as a new dict in the list of dicts
        optimizer_grouped_parameters = self.model.parameters()

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=self.args.learning_rate,
                               eps=self.args.adam_epsilon)
        self.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                                self.optimizer, num_warmup_steps=500,
                                num_training_steps= num_training_steps,
                                num_cycles = 2)

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)

        preds = output.predictions
        labels = output.label_ids
        self.draw_confusion_matrix(preds, labels)

        return output

    def draw_confusion_matrix(self, pred, label_ids):
        cm = confusion_matrix(label_ids, np.argmax(pred, axis=-1))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cmn = cmn.astype('int')
        fig = plt.figure(figsize=(22, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        cm_plot = sns.heatmap(cm, cmap='Blues', fmt='d', annot=True, ax=ax1)
        cm_plot.set_xlabel('pred')
        cm_plot.set_ylabel('true')
        cm_plot.set_title('confusion matrix')
        cmn_plot = sns.heatmap(
            cmn, cmap='Blues', fmt='d', annot=True, ax=ax2)
        cmn_plot.set_xlabel('pred')
        cmn_plot.set_ylabel('true')
        cmn_plot.set_title('confusion matrix normalize')
        wandb.log({'confusion_matrix': wandb.Image(fig)})

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }