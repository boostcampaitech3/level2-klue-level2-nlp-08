import pandas as pd

def default_entity(dataset):
  subject_entity = []
  object_entity = []

  for subj, obj in zip(dataset['subject_entity'], dataset['object_entity']):
    subj = subj[1:-1].split(',')[0].split(':')[1]
    obj = obj[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(subj)
    object_entity.append(obj)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})

  return out_dataset


def add_entity_token(dataset):
  subject_entity = []
  object_entity = []
  sentence = []

  for subj, obj, sent in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    subj_start = int(subj.split('\':')[2].split(',')[0])
    subj_end = int(subj.split('\':')[3].split(',')[0])
    obj_start = int(obj.split('\':')[2].split(',')[0])
    obj_end = int(obj.split('\':')[3].split(',')[0])

    if subj_start < obj_start:
      result = sent[:subj_start] + '[SUBJ]' + sent[subj_start:subj_end+1] + '[/SUBJ]' + sent[subj_end+1:obj_start] + '[OBJ]' + sent[obj_start:obj_end+1] + '[/OBJ]' + sent[obj_end+1:]
    else:
      result = sent[:obj_start] + '[OBJ]' + sent[obj_start:obj_end+1] + '[/OBJ]' + sent[obj_end+1:subj_start] + '[SUBJ]' + sent[subj_start:subj_end+1] + '[/SUBJ]' + sent[subj_end+1:]
        
    subject_entity.append(sent[subj_start:subj_end+1])
    object_entity.append(sent[obj_start:obj_end+1])
    sentence.append(result)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})

  return out_dataset


def add_entity_typed_token(dataset):
  subject_entity = []
  object_entity = []
  sentence = []

  for subj, obj, sent in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    subj_start = int(subj.split('\':')[2].split(',')[0])
    subj_end = int(subj.split('\':')[3].split(',')[0])
    obj_start = int(subj.split('\':')[2].split(',')[0])
    obj_end = int(subj.split('\':')[3].split(',')[0])
    subj_type = subj.split('\':')[-1][2:-2]
    obj_type = obj.split('\':')[-1][2:-2]

    if subj_start < obj_start:
      result = sent[:subj_start] + f'[SUBJ:{subj_type}]' + sent[subj_start:subj_end+1] + '[/SUBJ]' + sent[subj_end+1:obj_start] + f'[OBJ:{obj_type}]' + sent[obj_start:obj_end+1] + '[/OBJ]' + sent[obj_end+1:]
    else:
      result = sent[:obj_start] + f'[OBJ:{obj_type}]' + sent[obj_start:obj_end+1] + '[/OBJ]' + sent[obj_end+1:subj_start] + f'[SUBJ:{subj_type}]' + sent[subj_start:subj_end+1] + '[/SUBJ]' + sent[subj_end+1:]
        
    subject_entity.append(sent[subj_start:subj_end+1])
    object_entity.append(sent[obj_start:obj_end+1])
    sentence.append(result)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})

  return out_dataset


def swap_entity_typed_token(dataset):
  sentence = []
  subject_entity = []
  object_entity = []

  for subj, obj, sent, ids in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    subj_start = int(subj.split('\':')[2].split(',')[0])
    subj_end = int(subj.split('\':')[3].split(',')[0])
    obj_start = int(obj.split('\':')[2].split(',')[0])
    obj_end = int(obj.split('\':')[3].split(',')[0])
    subj_type = subj.split('\':')[-1][2:-2]
    obj_type = obj.split('\':')[-1][2:-2]

    if subj_start < obj_start:
      result = sent[:subj_start] + f'[SUBJ:{subj_type}]' + sent[subj_end+1:obj_start] + f'[OBJ:{obj_type}]' + sent[obj_end+1:]
    else:
      result = sent[:obj_start] + f'[OBJ:{obj_type}]' + sent[obj_end+1:subj_start] + f'[SUBJ:{subj_type}]' + sent[subj_end+1:]
    
    subject_entity.append(sent[subj_start:subj_end+1])
    object_entity.append(sent[obj_start:obj_end+1])
    sentence.append(result)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  
  return out_dataset