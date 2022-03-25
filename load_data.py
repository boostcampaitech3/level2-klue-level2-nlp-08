import pickle as pickle
import os
import pandas as pd
import torch


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  # for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
  #   i = i[1:-1].split(',')[0].split(':')[1]
  #   j = j[1:-1].split(',')[0].split(':')[1]

  #   subject_entity.append(i)
  #   object_entity.append(j)
  # out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})

  print("NEW PREPROCESSING")
  sentence = []
  for i,j, sent, ids in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence'], dataset['id']):
    i_start = int(i.split('\':')[2].split(',')[0])
    i_end = int(i.split('\':')[3].split(',')[0])
    j_start = int(j.split('\':')[2].split(',')[0])
    j_end = int(j.split('\':')[3].split(',')[0])
    
    if i_start < j_start:
      result = sent[:i_start] + '[SUBJ]' + sent[i_start:i_end+1] + '[/SUBJ]' + sent[i_end+1:j_start] + '[OBJ]' + sent[j_start:j_end+1] + '[/OBJ]' + sent[j_end+1:]
    else:
      result = sent[:j_start] + '[OBJ]' + sent[j_start:j_end+1] + '[/OBJ]' + sent[j_end+1:i_start] + '[SUBJ]' + sent[i_start:i_end+1] + '[/SUBJ]' + sent[i_end+1:]
    
    subject_entity.append(sent[i_start:i_end+1])
    object_entity.append(sent[j_start:j_end+1])
    sentence.append(result)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  # concat_entity = []
  # for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
  #   temp = ''
  #   temp = e01 + '[SEP]' + e02
  #   concat_entity.append(temp)

  add_special = tokenizer.add_special_tokens({'additional_special_tokens':['[SUBJ]', '[/SUBJ]', '[OBJ]', '[/OBJ]']})
  # tokenizer.__call__
  tokenized_sentences = tokenizer(
      #concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences
