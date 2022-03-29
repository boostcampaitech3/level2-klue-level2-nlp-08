import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *

import wandb
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Subset
from custom_trainer import CustomTrainer
from transformers import EarlyStoppingCallback

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

run_name = 'bolim_permuTok_robLag_20ep_5e5_2'
def train(MODE="default"):
  seed_everything(1004)
  # load model and tokenizer
  MODEL_NAME = "klue/roberta-large"

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  if MODE=="cv":
      num_added_sptoks = tokenizer.add_special_tokens({"additional_special_tokens": ['[TP]', '[/TP]']})

  DATA_PATH = '../dataset/train/train.csv'
  # TODO : 누가 ../../dataset/train/train_origin.csv로 설정해놨는데 수정 요구

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
  # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)

  if MODE=="bolim":
      run_name = 'bolim_permuTok_robLag_20ep_5e5_2'

      training_args = TrainingArguments(
          output_dir='./results',  # output directory
          save_total_limit=5,  # number of total save model.
          save_steps=500,  # model saving step.
          num_train_epochs=20,  # total number of training epochs
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
          run_name=run_name,
          fp16=True,
          fp16_opt_level="O1"
      )

      # trainer = Trainer(
      #     model=model,  # the instantiated 🤗 Transformers model to be trained
      #     args=training_args,  # training arguments, defined above
      #     train_dataset=RE_train_dataset,         # training dataset
      #     eval_dataset=RE_train_dataset,             # evaluation dataset
      #     compute_metrics=compute_metrics  # define metrics function
      # )
      # trainer.train()
      # model.save_pretrained('./best_model/' + run_name)
      train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1004)
      idx = 0
      for train_idx, valid_idx in train_val_split.split(RE_train_dataset, RE_train_dataset.labels):
          # for train_idx, valid_idx in kfold.split(RE_train_dataset, RE_train_dataset.labels):
          idx += 1
          train_data = Subset(RE_train_dataset, train_idx)
          valid_data = Subset(RE_train_dataset, valid_idx)

          trainer = Trainer(
              model=model,  # the instantiated 🤗 Transformers model to be trained
              args=training_args,  # training arguments, defined above
              # train_dataset=RE_train_dataset,         # training dataset
              train_dataset=train_data,
              # eval_dataset=RE_train_dataset,             # evaluation dataset
              eval_dataset=valid_data,
              compute_metrics=compute_metrics  # define metrics function
          )
          # train model
          trainer.train()

          # val set train
          trainer = Trainer(
              model=model,  # the instantiated 🤗 Transformers model to be trained
              args=training_args,  # training arguments, defined above
              # train_dataset=RE_train_dataset,         # training dataset
              train_dataset=valid_data,
              # eval_dataset=RE_train_dataset,             # evaluation dataset
              compute_metrics=compute_metrics  # define metrics function
          )
          # train model
          trainer.train()

      model.save_pretrained('./best_model/' + run_name)

  elif MODE=="cv":
      torch.cuda.empty_cache()
      train_val_split = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=1004)
      idx = 0
      for train_idx, valid_idx in train_val_split.split(RE_train_dataset, RE_train_dataset.labels):
          idx += 1
          # setting model hyperparameter
          model_config = AutoConfig.from_pretrained(MODEL_NAME)
          model_config.num_labels = 30

          model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
          model.resize_token_embeddings(tokenizer.vocab_size + num_added_sptoks)
          print(model.config)
          model.parameters
          model.to(device)

          # 사용한 option 외에도 다양한 option들이 있습니다.
          # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
          training_args = TrainingArguments(
              output_dir='./results',  # output directory
              save_total_limit=3,  # number of total save model.
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
              # load_best_model_at_end = True,
              report_to="wandb",
              run_name=run_name,
              fp16=True,
              fp16_opt_level="O1"
          )

          train_data = Subset(RE_train_dataset, train_idx)
          valid_data = Subset(RE_train_dataset, valid_idx)

          trainer = Trainer(
              model=model,  # the instantiated 🤗 Transformers model to be trained
              args=training_args,  # training arguments, defined above
              # train_dataset=RE_train_dataset,         # training dataset
              train_dataset=train_data,
              # eval_dataset=RE_train_dataset,             # evaluation dataset
              eval_dataset=valid_data,
              compute_metrics=compute_metrics  # define metrics function
          )
          # train model
          trainer.train()
          run_name = 'bolim_permuTok_robLag_20ep_5e5_2'
          model.save_pretrained('./best_model/' + run_name + '_' + str(idx))

          # # val set train
          # trainer = Trainer(
          #     model=model,  # the instantiated 🤗 Transformers model to be trained
          #     args=training_args,  # training arguments, defined above
          #     # train_dataset=RE_train_dataset,         # training dataset
          #     train_dataset=valid_data,
          #     # eval_dataset=RE_train_dataset,             # evaluation dataset
          #     compute_metrics=compute_metrics  # define metrics function
          # )
          # # train model
          # trainer.train()

  elif MODE=="chi0":
      # 사용한 option 외에도 다양한 option들이 있습니다.
      # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
      training_args = TrainingArguments(
          output_dir=f'./results/RL_typed_punct_LS',  # {n_fold}',          # output directory
          save_total_limit=2,  # number of total save model.
          save_steps=500,  # model saving step.
          num_train_epochs=5,  # total number of training epochs
          learning_rate=5e-5,  # learning_rate
          per_device_train_batch_size=32,  # batch size per device during training
          per_device_eval_batch_size=32,  # batch size for evaluation
          warmup_steps=500,  # number of warmup steps for learning rate scheduler[]
          weight_decay=0.01,  # strength of weight decay
          logging_dir='./logs',  # directory for storing logs
          logging_steps=100,  # log saving step.
          evaluation_strategy='steps',  # evaluation strategy to adopt during training
          # `no`: No evaluation during training.[]
          # `steps`: Evaluate every `eval_steps`.
          # `epoch`: Evaluate every end of epoch.
          # save_strategy="steps",
          eval_steps=500,  # evaluation step.
          load_best_model_at_end=True,  # Wandb에 best model checkpoint 저장
          report_to="wandb",  # Wandb에 log
          run_name=f"RL_typed_punct_LS",
          # _{n_fold}",               # Wandb run name   {번호}_{Model}_{이전 Model 번호}_{변경점}
          fp16=True,
          fp16_opt_level="O1",
          label_smoothing_factor=0.1
      )

      added_special = 0
      print(model.get_input_embeddings())
      model.resize_token_embeddings(tokenizer.vocab_size + added_special)
      print(model.get_input_embeddings())

      # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1004)
      # train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1004)
      run = wandb.init(project="KLUE", entity="miml", name=f"chi0 RL_typed_punct_LS")
      # for train_idx, valid_idx in train_val_split.split(RE_train_dataset, RE_train_dataset.labels):
      # for n_fold, (train_idx, valid_idx) in enumerate(kfold.split(RE_train_dataset, RE_train_dataset.labels)):

      # train_data = Subset(RE_train_dataset, train_idx)
      # valid_data = Subset(RE_train_dataset, valid_idx)
      # #print(f"\n{'='*50} {n_fold} Fold Start {'='*50}\n")

      # training_args = TrainingArguments(
      #   output_dir = f'./results/RL_typed_punct',#{n_fold}',          # output directory
      #   save_total_limit=2,              # number of total save model.
      #   save_steps=500,                 # model saving step.
      #   num_train_epochs=5,              # total number of training epochs
      #   learning_rate=5e-5,               # learning_rate
      #   per_device_train_batch_size=32,  # batch size per device during training
      #   per_device_eval_batch_size=32,   # batch size for evaluation
      #   warmup_steps=500,                # number of warmup steps for learning rate scheduler[]
      #   weight_decay=0.01,               # strength of weight decay
      #   logging_dir='./logs',            # directory for storing logs
      #   logging_steps=100,              # log saving step.
      #   evaluation_strategy='steps', # evaluation strategy to adopt during training
      #                               # `no`: No evaluation during training.[]
      #                                 # `steps`: Evaluate every `eval_steps`.
      #                               # `epoch`: Evaluate every end of epoch.
      #   #save_strategy="steps",
      #   eval_steps = 500,            # evaluation step.
      #   load_best_model_at_end = True,  # Wandb에 best model checkpoint 저장
      #   report_to = "wandb",         # Wandb에 log
      #   run_name = f"RL_typed_punct",#_{n_fold}",               # Wandb run name   {번호}_{Model}_{이전 Model 번호}_{변경점}
      #   fp16=True,
      #   fp16_opt_level="O1"
      # )

      # trainer = CustomTrainer(
      #   loss_name='focal',
      #   model=model,                         # the instantiated 🤗 Transformers model to be trained
      #   args=training_args,                  # training arguments, defined above
      #   train_dataset=train_data,            # training dataset
      #   eval_dataset=valid_data,             # evaluation dataset
      #   compute_metrics=compute_metrics      # define metrics function
      #  # callbacks=[EarlyStoppingCallback(early_stopping_patience = 3)]
      # )
      # trainer.train()

      trainer = CustomTrainer(
          loss_name='LabelSmoothing',
          model=model,  # the instantiated 🤗 Transformers model to be trained
          args=training_args,  # training arguments, defined above
          train_dataset=RE_train_dataset,  # training dataset
          eval_dataset=RE_train_dataset,  # evaluation dataset
          compute_metrics=compute_metrics  # define metrics function
      )
      trainer.train()

      # train model
      model.save_pretrained(f'./best_model/RL_typed_punct_LS')  # _{n_fold}')

      # run.finish()
      # print(f"\n{'='*50}Fold {n_fold} Finish{'='*50}\n")

  else:
      # 사용한 option 외에도 다양한 option들이 있습니다.
      # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
      training_args = TrainingArguments(
          output_dir=f'./results/RL_typed_punct',  # {n_fold}',          # output directory
          save_total_limit=2,  # number of total save model.
          save_steps=500,  # model saving step.
          num_train_epochs=5,  # total number of training epochs
          learning_rate=5e-5,  # learning_rate
          per_device_train_batch_size=32,  # batch size per device during training
          per_device_eval_batch_size=32,  # batch size for evaluation
          warmup_steps=500,  # number of warmup steps for learning rate scheduler[]
          weight_decay=0.01,  # strength of weight decay
          logging_dir='./logs',  # directory for storing logs
          logging_steps=100,  # log saving step.
          evaluation_strategy='steps',  # evaluation strategy to adopt during training
          # `no`: No evaluation during training.[]
          # `steps`: Evaluate every `eval_steps`.
          # `epoch`: Evaluate every end of epoch.
          # save_strategy="steps",
          eval_steps=500,  # evaluation step.
          load_best_model_at_end=True,  # Wandb에 best model checkpoint 저장
          report_to="wandb",  # Wandb에 log
          run_name=f"RL_typed_punct",  # _{n_fold}",               # Wandb run name   {번호}_{Model}_{이전 Model 번호}_{변경점}
          fp16=True,
          fp16_opt_level="O1"
      )

      added_special = 0
      print(model.get_input_embeddings())
      model.resize_token_embeddings(tokenizer.vocab_size + added_special)
      print(model.get_input_embeddings())

      # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1004)
      # train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1004)
      run = wandb.init(project="KLUE", entity="miml", name=f"chi0 RL_typed_punct")
      # for train_idx, valid_idx in train_val_split.split(RE_train_dataset, RE_train_dataset.labels):
      # for n_fold, (train_idx, valid_idx) in enumerate(kfold.split(RE_train_dataset, RE_train_dataset.labels)):

      # train_data = Subset(RE_train_dataset, train_idx)
      # valid_data = Subset(RE_train_dataset, valid_idx)
      # #print(f"\n{'='*50} {n_fold} Fold Start {'='*50}\n")

      # training_args = TrainingArguments(
      #   output_dir = f'./results/RL_typed_punct',#{n_fold}',          # output directory
      #   save_total_limit=2,              # number of total save model.
      #   save_steps=500,                 # model saving step.
      #   num_train_epochs=5,              # total number of training epochs
      #   learning_rate=5e-5,               # learning_rate
      #   per_device_train_batch_size=32,  # batch size per device during training
      #   per_device_eval_batch_size=32,   # batch size for evaluation
      #   warmup_steps=500,                # number of warmup steps for learning rate scheduler[]
      #   weight_decay=0.01,               # strength of weight decay
      #   logging_dir='./logs',            # directory for storing logs
      #   logging_steps=100,              # log saving step.
      #   evaluation_strategy='steps', # evaluation strategy to adopt during training
      #                               # `no`: No evaluation during training.[]
      #                                 # `steps`: Evaluate every `eval_steps`.
      #                               # `epoch`: Evaluate every end of epoch.
      #   #save_strategy="steps",
      #   eval_steps = 500,            # evaluation step.
      #   load_best_model_at_end = True,  # Wandb에 best model checkpoint 저장
      #   report_to = "wandb",         # Wandb에 log
      #   run_name = f"RL_typed_punct",#_{n_fold}",               # Wandb run name   {번호}_{Model}_{이전 Model 번호}_{변경점}
      #   fp16=True,
      #   fp16_opt_level="O1"
      # )

      # trainer = CustomTrainer(
      #   loss_name='focal',
      #   model=model,                         # the instantiated 🤗 Transformers model to be trained
      #   args=training_args,                  # training arguments, defined above
      #   train_dataset=train_data,            # training dataset
      #   eval_dataset=valid_data,             # evaluation dataset
      #   compute_metrics=compute_metrics      # define metrics function
      #  # callbacks=[EarlyStoppingCallback(early_stopping_patience = 3)]
      # )
      # trainer.train()

      trainer = CustomTrainer(
          # loss_name='focal',
          model=model,  # the instantiated 🤗 Transformers model to be trained
          args=training_args,  # training arguments, defined above
          train_dataset=RE_train_dataset,  # training dataset
          eval_dataset=RE_train_dataset,  # evaluation dataset
          compute_metrics=compute_metrics  # define metrics function
      )
      trainer.train()

      # train model
      model.save_pretrained(f'./best_model/RL_typed_punct')  # _{n_fold}')

      # run.finish()
      # print(f"\n{'='*50}Fold {n_fold} Finish{'='*50}\n")

def main():
  MODE = "default"
  wandb_setting = False
  if wandb_setting:
    run_name = 'bolim_pucTok_robLag_5ep_5e5'
    wandb.init(project="KLUE", entity="miml", name=run_name)

  train(MODE)

if __name__ == '__main__':
  main()