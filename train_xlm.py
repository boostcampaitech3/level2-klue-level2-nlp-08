import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from load_data import *
from metric import *
from model import *

import wandb
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch.utils.data import Subset
from custom_trainer import CustomTrainer
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM, XLMRobertaForSequenceClassification

def train(MODE="default", run_name="Not_Setting"):
  torch.cuda.empty_cache()
  seed_everything(1004)
  # load model and tokenizer
  MODEL_NAME = "xlm-roberta-large"

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  DATA_PATH = '../../dataset/train/final_train.csv'

  # load dataset
  train_dataset = load_data(DATA_PATH, entity_tk_type='add_entity_type_punct')
  train_label = label_to_num(train_dataset['label'].values)
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)

  valid = False
  valid_size = 0.1
  if valid:
    print("MODE : VALID\n")
    RE_train, RE_valid = train_test_split(RE_train_dataset, test_size=valid_size,
                                                     shuffle=True, stratify=train_dataset['label'])
  else:
    print("MODE : NO VALID\n")
    RE_train = RE_train_dataset
    RE_valid = RE_train

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

  print(model.config)
  model.parameters
  model.to(device)

  torch.cuda.empty_cache()

  output_dir = './results/xlm_large_final' # TODO : output_dir 설정
  label_smoothing_factor = 0.0 # TODO : label_smoothing factor

  wandb.init(
    project='KLUE',
    entity='miml',
    name=run_name
  )

  training_args = TrainingArguments(
    output_dir=output_dir,  # output directory
    save_total_limit=2,  # number of total save model.
    save_steps=500,  # model saving step.
    num_train_epochs=3,  # total number of training epochs
    learning_rate=2e-5,  # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,  # batch size for evaluation
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=100,  # log saving step.
    evaluation_strategy='steps',  # evaluation strategy to adopt during training
    # `no`: No evaluation during training.
    # `steps`: Evaluate every `eval_steps`.
    # `epoch`: Evaluate every end of epoch.
    eval_steps=500,  # evaluation step.
    load_best_model_at_end=True,
    report_to="wandb",
    fp16=True,
    fp16_opt_level="O1",
    label_smoothing_factor=label_smoothing_factor
  )

  custom = False
  if custom:
    trainer = CustomTrainer(
      loss_name='CrossEntropy',#'LabelSmoothing',
      model=model,  # the instantiated 🤗 Transformers model to be trained
      args=training_args,  # training arguments, defined above
      train_dataset=RE_train,  # training dataset
      eval_dataset=RE_valid,  # evaluation dataset
      compute_metrics=compute_metrics  # define metrics function
    )
  else:
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train,  # training dataset
        eval_dataset=RE_valid,  # evaluation dataset
        compute_metrics=compute_metrics  # define metrics function
    )

  trainer.train()
  model.save_pretrained('./best_model/' + run_name)

def main():
  MODE = "default"
  run_name = "xlm_large_final"

  train(MODE=MODE, run_name=run_name)

if __name__ == '__main__':
    main()