import os
import pandas as pd
import torch
import sklearn
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from load_data import *
from metric import *

import wandb
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch.utils.data import Subset
from custom_trainer import CustomTrainer

def train(MODE="default", run_name="Not_Setting"):
  seed_everything(1004)
  # load model and tokenizer
  MODEL_NAME = "klue/roberta-large"

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  if MODE=="cv":
      num_added_sptoks = tokenizer.add_special_tokens({"additional_special_tokens": ['[TP]', '[/TP]']})
  # TODO : [TP], [/TP] special token 추가할 경우

  DATA_PATH = '../dataset/train/train.csv'
  # TODO : train.csv 파일 경로

  # load dataset
  train_dataset = load_data(DATA_PATH)

  train_label = label_to_num(train_dataset['label'].values)
  # dev_label = label_to_num(dev_dataset['label'].values)

  add_special_tokenizer = False
  # tokenizing dataset
  if add_special_tokenizer:
    tokenized_train, added_special = tokenized_dataset(train_dataset, tokenizer, add_special=True)
  else:
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)

  valid = False
  if valid:
      RE_train_dataset, RE_dev_dataset = train_test_split(RE_train_dataset, test_size=0.1,
                                                   shuffle=True, stratify=train_dataset['label'])
  else:
      RE_dev_dataset = RE_train_dataset

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)

  initial = True
  if initial:
      for param in model.classifier.modules():
          if isinstance(param, nn.Linear):
              nn.init.kaiming_uniform_(param.weight.data)
              param.bias.data.fill_(0)


  torch.cuda.empty_cache()

  output_dir = './results' # TODO : output_dir 설정
  label_smoothing_factor = 0.0 # TODO : label_smoothing factor

  wandb.init(
      project='KLUE',
      entity='miml',
      name=run_name
  )

  training_args = TrainingArguments(
      output_dir=output_dir,  # output directory
      save_total_limit=5,  # number of total save model.
      save_steps=500,  # model saving step.
      num_train_epochs=5,  # total number of training epochs
      learning_rate=5e-5,  # learning_rate
      per_device_train_batch_size=32,  # batch size per device during training
      per_device_eval_batch_size=32,  # batch size for evaluation
      warmup_steps=500,  # number of warmup steps for learning rate scheduler
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
          loss_name='LabelSmoothing',
          model=model,  # the instantiated 🤗 Transformers model to be trained
          args=training_args,  # training arguments, defined above
          train_dataset=RE_train_dataset,  # training dataset
          eval_dataset=RE_train_dataset,  # evaluation dataset
          compute_metrics=compute_metrics  # define metrics function
      )
  else:
      trainer = Trainer(
          model=model,  # the instantiated 🤗 Transformers model to be trained
          args=training_args,  # training arguments, defined above
          train_dataset=RE_train_dataset,  # training dataset
          eval_dataset=RE_dev_dataset,  # evaluation dataset
          compute_metrics=compute_metrics  # define metrics function
      )

  ensemble = False
  valid_add = False
  if ensemble:
      train_val_split = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=1004)
      idx = 0
      for train_idx, valid_idx in train_val_split.split(RE_train_dataset, RE_train_dataset.labels):
          idx += 1
          model_config = AutoConfig.from_pretrained(MODEL_NAME)
          model_config.num_labels = 30

          model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
          model.resize_token_embeddings(tokenizer.vocab_size + num_added_sptoks)
          # TODO : MODE가 "cv"여야지만 num_added_sptoks가 설정됨
          print(model.config)
          model.parameters
          model.to(device)

          train_data = Subset(RE_train_dataset, train_idx)
          valid_data = Subset(RE_train_dataset, valid_idx)

          trainer = Trainer(
              model=model,  # the instantiated 🤗 Transformers model to be trained
              args=training_args,  # training arguments, defined above
              train_dataset=train_data,
              eval_dataset=valid_data,
              compute_metrics=compute_metrics  # define metrics function
          )
          # train model
          trainer.train()
          model.save_pretrained('./best_model/' + run_name + '_' + str(idx))

  elif valid_add:
      train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1004)
      idx = 0
      for train_idx, valid_idx in train_val_split.split(RE_train_dataset, RE_train_dataset.labels):
          idx += 1
          train_data = Subset(RE_train_dataset, train_idx)
          valid_data = Subset(RE_train_dataset, valid_idx)

          trainer = Trainer(
              model=model,  # the instantiated 🤗 Transformers model to be trained
              args=training_args,  # training arguments, defined above
              train_dataset=train_data,
              eval_dataset=valid_data,
              compute_metrics=compute_metrics  # define metrics function
          )
          trainer.train()

          # val set train
          trainer = Trainer(
              model=model,  # the instantiated 🤗 Transformers model to be trained
              args=training_args,  # training arguments, defined above
              train_dataset=valid_data,
              compute_metrics=compute_metrics  # define metrics function
          )
          # train model
          trainer.train()
      model.save_pretrained('./best_model/' + run_name)

  else:
      trainer.train()
      model.save_pretrained('./best_model/' + run_name)

def main():
  MODE = "default"
  run_name = "dongjin_kaiming_initalize_classifier"

  train(MODE=MODE, run_name=run_name)

if __name__ == '__main__':
    main()