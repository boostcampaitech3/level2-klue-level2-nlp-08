import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, \
    AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, \
    RobertaForSequenceClassification, BertTokenizer, BertModel
from load_data import *
import wandb
import random
from transformers import XLNetTokenizer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size = 768, num_classes = 30, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.model = bert_model
        self.dr_rate = dr_rate
        print(self.model)

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids = input_ids, attention_mask = attention_mask)
        if self.dr_rate:
            out = self.dropout(out.pooler_output)
        else:
            out = out.pooler_output
        real_out = self.classifier(out)
        return real_out