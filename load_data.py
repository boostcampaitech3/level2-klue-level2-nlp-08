import pandas as pd
import torch

import utils

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

def preprocessing_dataset(dataset, entity_tk_type):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  print(f"Preprocessing type : {entity_tk_type}\n")
  subject_entity = []
  object_entity = []
  sentence = []
  for subj,obj,sent in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    subj_word = subj[1:-1].split('\', ')[0].split(':')[1].replace("'", '').strip()
    obj_word = obj[1:-1].split('\', ')[0].split(':')[1].replace("'", '').strip()

    subj_start = int(subj.split('\':')[2].split(',')[0])
    subj_end = int(subj.split('\':')[3].split(',')[0])
    obj_start = int(obj.split('\':')[2].split(',')[0])
    obj_end = int(obj.split('\':')[3].split(',')[0])
    subj_type = subj[1:-1].split('\':')[4].replace("'", '').strip()
    obj_type = obj[1:-1].split('\':')[4].replace("'", '').strip()

    preprocessed_sent = getattr(utils, entity_tk_type)(sent, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type)
    """
    entity_tk_type
    
    add_entity_type_suffix_kr(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type)
    add_entity_type_punct_kr(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type)
    def add_entity_type_token(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type)
    def add_entity_token(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type)
    def add_entity_token_with_type(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type)
    def swap_entity_token_with_type(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type)
    def default_sent(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
    def add_entity_type_punct_kr_subj_obj(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type)
    """


    subject_entity.append(subj_word)
    object_entity.append(obj_word)
    sentence.append(preprocessed_sent)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir, entity_tk_type='add_entity_type_punct_kr'):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset, entity_tk_type)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  # tokenizer.__call__

  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    # temp = e01 + '와 ' + e02 +'의 관계를 구하시오.'
    temp = ''
    temp = f'이 문장에서 *{e01}*과 ^{e02}^은 어떤 관계일까?'  # multi 방식 사용
    concat_entity.append(temp)
    
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
