from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable

# https://github.com/clcarwin/focal_loss_pytorch
class other_FocalLoss(nn.Module):
    def __init__(self, gamma=5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim = -1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

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

# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=30, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


"""https://kyunghyunlim.github.io/nlp/ml_ai/2021/10/01/hf_culoss.html"""
class CustomTrainer(Trainer):
    def __init__(self, loss_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name

    def compute_loss(self, model, inputs, return_outputs=False):
        """ Default Loss : CrossEntropyLoss (defined at RobertaForMaskedLM) """
        labels = inputs.pop('labels')

        outputs = model(**inputs)
        custom_loss = torch.nn.CrossEntropyLoss()
        loss = custom_loss(outputs, labels)
        """
        if self.loss_name == 'focal':
            custom_loss = FocalLoss()
        elif self.loss_name == 'f1':
            custom_loss = F1Loss()
        elif self.loss_name == 'other_focal':
            custom_loss = other_FocalLoss()
        elif self.loss_name == 'LabelSmoothing':
            loss = self.label_smoother(outputs, labels)
        elif self.loss_name == 'CrossEntropy':
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        if custom_loss is not None:
            loss = custom_loss(outputs[0], labels)
        """
        
        return (loss, outputs) if return_outputs else loss

    """https://github.com/l-yohai/korean-entity-relation-extraction/blob/73688dee55ae93094c21142299586f04efd9a8ee/trainer/trainer.py#L41"""

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