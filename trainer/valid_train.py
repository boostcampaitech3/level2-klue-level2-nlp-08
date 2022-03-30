import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments
from load_data import *
from metric import *

import wandb
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from custom_trainer import CustomTrainer

def valid_train(config):
  seed_everything(config['random_seed'])
  # load model and tokenizer
  MODEL_NAME = config['MODEL_NAME']

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # train_cv.py
  # num_added_sptoks = tokenizer.add_special_tokens({"additional_special_tokens": ['[TP]', '[/TP]']})
  # TODO : [TP], [/TP] special token 추가할 경우

  DATA_PATH = config['dataset_path']
  # TODO : train.csv 파일 경로

  # load dataset
  train_dataset = load_data(DATA_PATH)
  train_label = label_to_num(train_dataset['label'].values)
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)

  train_valid_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1004)
  for train_idx, valid_idx in train_valid_split(RE_train_dataset, RE_train_dataset.labels):
      RE_train_dataset = Subset(RE_train_dataset, train_idx)
      RE_dev_dataset = Subset(RE_train_dataset, valid_idx)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)

  torch.cuda.empty_cache()

  wandb_configs = config['wandb']
  wandb.init(
      project=wandb_configs['project'],
      entity=wandb_configs['entity'],
      name=wandb_configs['run_name']
  )

  training_configs = config['TrainingArguments']
  training_args = TrainingArguments(
      output_dir=training_configs['output_dir'],  # output directory
      save_total_limit=training_configs['save_total_limit'],  # number of total save model.
      save_steps=training_configs['save_steps'],  # model saving step.
      num_train_epochs=training_configs['num_train_epochs'],  # total number of training epochs
      learning_rate=training_configs['learning_rate'],  # learning_rate
      per_device_train_batch_size=training_configs['per_device_train_batch_size'],  # batch size per device during training
      per_device_eval_batch_size=training_configs['per_device_eval_batch_size'],  # batch size for evaluation
      warmup_steps=training_configs['warmup_steps'],  # number of warmup steps for learning rate scheduler
      weight_decay=training_configs['weight_decay'],  # strength of weight decay
      logging_dir=training_configs['logging_dir'],  # directory for storing logs
      logging_steps=training_configs['logging_steps'],  # log saving step.
      evaluation_strategy=training_configs['evaluation_strategy'],  # evaluation strategy to adopt during training
                                                                    # `no`: No evaluation during training.
                                                                    # `steps`: Evaluate every `eval_steps`.
                                                                    # `epoch`: Evaluate every end of epoch.
      eval_steps=training_configs['eval_steps'],  # evaluation step.
      load_best_model_at_end=training_configs['load_best_model_at_end'],
      report_to=training_configs['report_to'],
      fp16=training_configs['fp16'],
      fp16_opt_level=training_configs['fp16_opt_level'],
      label_smoothing_factor=training_configs['label_smoothing_factor']
  )

  trainer_configs = config['Trainer']
  if trainer_configs['custom']:
      trainer = CustomTrainer(
          loss_name=trainer_configs['loss_name'],
          model=model,  # the instantiated 🤗 Transformers model to be trained
          args=training_args,  # training arguments, defined above
          train_dataset=RE_train_dataset,  # training dataset
          eval_dataset=RE_dev_dataset,  # evaluation dataset
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

  # Hard Voting Ensemble
  trainer.train()
  model.save_pretrained('./best_model/' + wandb_configs['run_name'])