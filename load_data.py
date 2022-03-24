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
  sentence = []
  for i,j,k in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    tokens = [eval(i)['start_idx'], eval(i)['end_idx'], eval(i)['type'], eval(j)['start_idx'], eval(j)['end_idx'], eval(j)['type']]
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]
    k = k[:tokens[0]]\
    + f'[{tokens[2]}]' + k[tokens[0]:tokens[1]+1] + f'[/{tokens[2]}]'\
    + k[tokens[1]+1:tokens[3]]\
    + f'[{tokens[5]}]' + k[tokens[3]:tokens[4]+1] + f'[/{tokens[5]}]'\
    + k[tokens[4]+1:]
    subject_entity.append(i)
    object_entity.append(j)
    sentence.append(k)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
    # tokenizer에 special token 추가
    user_defined_symbols = ['[ORG]', '[/ORG]', '[DAT]', '[/DAT]', '[LOC]', '[/LOC]', '[PER]', '[/PER]', '[POH]', '[/POH]', '[NOH]', '[/NOH]']
    special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
    tokenizer.add_special_tokens(special_tokens_dict)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences
