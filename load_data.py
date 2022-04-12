import pandas as pd

import utils

def preprocessing_dataset(dataset, entity_tk_type):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  sentence = []
  subject_entity_type = []
  object_entity_type = []
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

    subject_entity.append(subj_word)
    object_entity.append(obj_word)
    sentence.append(preprocessed_sent)
    subject_entity_type.append(subj_type)
    object_entity_type.append(obj_type)
  out_dataset = pd.DataFrame(
    {'id': dataset['id'], 'sentence': sentence, 'subject_entity': subject_entity, 'object_entity': object_entity,
     'subject_type': subject_entity_type, 'object_type': object_entity_type, 'label': dataset['label'], })
  return out_dataset

def load_data(dataset_dir, entity_tk_type='add_entity_type_punct_kr'):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset, entity_tk_type)

  return dataset