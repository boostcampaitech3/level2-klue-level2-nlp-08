import pickle as pickle
import os
import pandas as pd
import torch, wandb, random
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
)
from load_data import *
from torch.utils.data import Subset
import gc
import argparse
from utils import *

path="./results/"
checkpoints = search_checkpoint(path)
checkpoints = checkpoints[0] + checkpoints

model = []
for checkpoint in checkpoints:
    model.append(AutoModelForSequenceClassification.from_pretrained(path + '/' + checkpoint))

# 모델의 state_dict 가중평균 구하기

# 1. 0번째 모델의 state_dict를 모두 0으로 만들기

for param_tensor in list(model[0].state_dict())[1:]:
    model[0].state_dict()[param_tensor] -= model[0].state_dict()[param_tensor]

# 2. 1번째~3번째 모델의 state_dict/3 더하기

for i in range(1,len(model)):
    for param_tensor in list(model[i].state_dict())[1:]:
        model[0].state_dict()[param_tensor] += (model[i].state_dict()[param_tensor]/3).float()

# # 모델의 state_dict 출력
# print("Model's state_dict:")
# for param_tensor in model[0].state_dict():
#     print(param_tensor, "\t", model[0].state_dict()[param_tensor])

weighted_model = model[0]
weighted_model.save_pretrained(f"./best_model/")
