import pandas as pd
from ast import literal_eval
import pickle

def label_to_num(label):
    num_label = []
    with open('./dict_num/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label

def preprocessing_dataset(data):
    subject_entity_type = []
    subject_entity_pos = []
    subject_entity_word = []

    object_entity_type = []
    object_entity_pos = []
    object_entity_word = []

    for subject_entity, object_entity in zip(data['subject_entity'], data['object_entity']):
        subject_entity = literal_eval(subject_entity)
        subject_entity_word.append(subject_entity['word'])
        subject_entity_pos.append([subject_entity['start_idx'], subject_entity['end_idx']])
        subject_entity_type.append(subject_entity['type'])

        object_entity = literal_eval(object_entity)
        object_entity_word.append(object_entity['word'])
        object_entity_pos.append([object_entity['start_idx'], object_entity['end_idx']])
        object_entity_type.append(object_entity['type'])

    dataset  = pd.DataFrame({'id' : data['id'], 'sentence' : data['sentence'], 'subject_entity_word' : subject_entity_word,
                             'subject_entity_pos' : subject_entity_pos,'subject_entity_type' : subject_entity_type
                            , 'object_entity_word' : object_entity_word, 'object_entity_pos' : object_entity_pos,
                             'object_entity_type' : object_entity_type, 'label' : data['label']
                            })

    return dataset

def load_data(config):
  pd_dataset = pd.read_csv(config['data_path'])
  dataset = preprocessing_dataset(pd_dataset)
  dataset['label'] = label_to_num(dataset['label'])

  return dataset