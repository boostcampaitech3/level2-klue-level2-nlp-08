import pickle as pickle
import os
import pandas as pd
import torch
import re
import urllib3
import json
from konlpy.tag import Mecab
import ast
# with open('key.json', 'r') as f:
#     key = json.load(f)['key']
# def get_Etri_API(text):
#     openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
#     accessKey = key
#     analysisCode = "ner"  # morp
#
#     requestJson = {
#         "access_key": accessKey,
#         "argument": {
#             "text": text,
#             "analysis_code": analysisCode
#         }
#     }
#
#     http = urllib3.PoolManager()
#     response = http.request(
#         "POST",
#         openApiURL,
#         headers={"Content-Type": "application/json; charset=UTF-8"},
#         body=json.dumps(requestJson)
#     )
#
#     if response.status == 200:
#         data = ast.literal_eval(response.data.decode('utf-8'))
#         NES = data['return_object']['sentence'][0]['NE']
#         for n in NES:
#             if n['text'] in text:
#                 text = text.replace(n['text'],'[NER]'+n['text']+'[/NER]')
#     else:
#         print(response.status)
mecab = Mecab()
def add_spTok(text):
    for noun in mecab.nouns(text):
        text = text.replace(noun,'[NER]'+noun+'[/NER]')
    return text
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
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
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
  tokenized_sentences = tokenizer(
      concat_entity,
      list(text for text in dataset['sentence']),#add_spTok(text)
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences
