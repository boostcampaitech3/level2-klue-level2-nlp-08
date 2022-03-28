from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import torch

def get_tokenizer(MODEL_NAME:str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[SUB]', '[/SUB]', '[OBJ]', '[/OBJ]']})
    return tokenizer

def special_token_sentence(dataset):
    concat_entity = []
    sentence1: str
    sentence2: str
    for idx, (subject_pos, sentences, object_pos) in enumerate(zip(dataset['subject_entity_pos'], dataset['sentence'], dataset['object_entity_pos'])):
        if subject_pos[0] < object_pos[0]:
            switch = True
        else:
            switch = False

        if switch:
            sentence1 = sentences[:object_pos[0]] + '[OBJ]' + sentences[object_pos[0]:object_pos[1]+1] + '[/OBJ]' + sentences[object_pos[1]+1:]
            sentence2 = sentence1[:subject_pos[0]] + '[SUB]' + sentence1[subject_pos[0]:subject_pos[1]+1] + '[/SUB]' + sentence1[subject_pos[1]+1:]
        else:
            sentence1 = sentences[:subject_pos[0]] + '[SUB]' + sentences[subject_pos[0]:subject_pos[1]+1] + '[/SUB]' + sentences[subject_pos[1]+1:]
            sentence2 = sentence1[:object_pos[0]] + '[OBJ]' + sentence1[object_pos[0]:object_pos[1]+1]+ '[/OBJ]' + sentence1[object_pos[1]+1:]
        concat_entity.append(sentence2)
    return concat_entity

def tokenized_dataset(dataset, tokenizer):
    concat_entity = special_token_sentence(dataset)
    tokenized_sentences = tokenizer(
        list(concat_entity),
        return_tensors = 'pt',
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True
      )
    return tokenized_sentences

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = self.labels[idx]
    return item

  def __len__(self):
    return len(self.labels)

def tokenizing_data(train_dataset, config):
    tokenizer = get_tokenizer(config['tokenizer_name'])

    train_tokenized = tokenized_dataset(train_dataset, tokenizer)

    RE_train_dataset = RE_Dataset(train_tokenized, train_dataset['label'])

    RE_train_data, RE_test_data = train_test_split(RE_train_dataset, test_size=config['valid_size'],
                                                   shuffle=True, stratify=train_dataset['label'])

    return tokenizer, RE_train_data, RE_test_data