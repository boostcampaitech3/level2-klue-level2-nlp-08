from konlpy.tag import Mecab
import torch
import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def add_spTok(text):
    mecab = Mecab()
    for noun in mecab.nouns(text):
        text = text.replace(noun,'[NER]'+noun+'[/NER]')
    return text

# *entity[TP]TYPE[/TP]*  * entity[TP]TYPE[/TP]*
def add_entity_type_suffix_kr(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
    TYPE = {'ORG':'단체정보','PER':'사람정보','DAT':'날짜정보','LOC':'위치정보','POH':'기타정보','NOH':'기타 수량표현'}
    subj_word = text[subj_start:subj_end + 1]
    obj_word = text[obj_start:obj_end + 1]
    
    if subj_start < obj_start:
        # new_text = text[:subj_start] + '*' + text[subj_start:subj_end + 1] + '[' + TYPE[subj_type] + ']*' + text[subj_end + 1:j_start] + '*' + \
        #        text[j_start:j_end + 1] + '[' + TYPE[j_type] + ']*'  + text[j_end + 1:]
        new_text = text[:subj_start] + f'*{subj_word}[TP]{TYPE[subj_type]}[/TP]*' + text[subj_end + 1:obj_start] + \
                   f'*{obj_word}[TP]{TYPE[obj_type]}[/TP]*' + text[obj_end + 1:]
    else:
        # new_text = text[:j_start] + '*' + text[j_start:j_end + 1] + '[' + TYPE[j_type] + ']*' + text[j_end + 1:subj_start] + '*' + \
        #        text[subj_start:subj_end + 1] + '[' + TYPE[subj_type] + ']*' + text[subj_end + 1:]
        new_text = text[:obj_start] + f'*{obj_word}[TP]{TYPE[obj_type]}[/TP]*' + text[obj_end + 1:subj_start] + \
                   f'*{subj_word}[TP]{TYPE[subj_type]}[/TP]*' + text[subj_end + 1:]
    return new_text

# @*TYPE*entity@  #^TYPE^entity#
def add_entity_type_punct_kr(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
    TYPE = {'ORG':'단체', 'PER':'사람', 'DAT':'날짜', 'LOC':'위치', 'POH':'기타', 'NOH':'수량'}
    subj_word = text[subj_start:subj_end + 1]
    obj_word = text[obj_start:obj_end + 1]
    
    if subj_start < obj_start:
      new_text = text[:subj_start] + f'@*{TYPE[subj_type]}*{subj_word}@' + text[subj_end + 1:obj_start] + \
                 f'#^{TYPE[obj_type]}^{obj_word}#' + text[obj_end + 1:]
    else:
      new_text = text[:obj_start] + f'@*{TYPE[obj_type]}*{obj_word}@' + text[obj_end + 1:subj_start] + \
                 f'#^{TYPE[subj_type]}^{subj_word}#' + text[subj_end + 1:]
    
    return new_text

# [subj_type]entity[/subj_type]  [obj_type]entity[/obj_type]
def add_entity_type_token(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
    subj_word = text[subj_start:subj_end + 1]
    obj_word = text[obj_start:obj_end + 1]
    
    if subj_start < obj_start:
        new_text = text[:subj_start] + f'[{subj_type}]{subj_word}[/{subj_type}]' + text[subj_end + 1:obj_start] + \
                   f'[{obj_type}]{obj_word}[/{obj_type}]' + text[obj_end + 1:]
    else:
        new_text = text[:obj_start] + f'[{obj_type}]{obj_word}[/{obj_type}]' + text[obj_end + 1:subj_start] + \
                   f'[{subj_type}]{subj_word}[/{subj_type}]' + text[subj_end + 1:]
    
    return new_text

# [SUBJ]entity[/SUBJ]
def add_entity_token(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
  subj_word = text[subj_start:subj_end + 1]
  obj_word = text[obj_start:obj_end + 1]
  
  if subj_start < obj_start:
    new_text = text[:subj_start] + f'[SUBJ]{subj_word}[/SUBJ]' + text[subj_end+1:obj_start] + f'[OBJ]{obj_word}[/OBJ]' + text[obj_end+1:]
  else:
    new_text = text[:obj_start] + f'[OBJ]{obj_word}[/OBJ]' + text[obj_end+1:subj_start] + f'[SUBJ]{subj_word}[/SUBJ]' + text[subj_end+1:]
  return new_text


# [SUBJ:type]entity[/SUBJ]
def add_entity_token_with_type(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
  subj_word = text[subj_start:subj_end + 1]
  obj_word = text[obj_start:obj_end + 1]

  if subj_start < obj_start:
    new_text = text[:subj_start] + f'[SUBJ:{subj_type}]{subj_word}[/SUBJ]' + text[subj_end+1:obj_start] + f'[OBJ:{obj_type}]{obj_word}[/OBJ]' + text[obj_end+1:]
  else:
    new_text = text[:obj_start] + f'[OBJ:{obj_type}]{obj_word}[/OBJ]' + text[obj_end+1:subj_start] + f'[SUBJ:{subj_type}]{subj_word}[/SUBJ]' + text[subj_end+1:]
  return new_text

# [OBJ] entity [/OBJ]
def special_token_sentence(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
    concat_entity = []
    sentence1: str
    sentence2: str

    if subj_start < obj_start:
        sentence1 = text[:obj_start] + '[OBJ]' + text[obj_start:obj_end + 1] + '[/OBJ]' + text[obj_end + 1:]
        new_text = sentence1[:subj_start] + '[SUB]' + sentence1[subj_start:subj_end + 1] + '[/SUB]' + sentence1[subj_end + 1:]

    else:
        sentence1 = text[:subj_start] + '[SUB]' + text[subj_start:subj_end + 1] + '[/SUB]' + text[subj_end + 1:]
        new_text = sentence1[:obj_start] + '[OBJ]' + sentence1[obj_start:obj_end + 1] + '[/OBJ]' + sentence1[obj_end + 1:]
    return new_text

def special_token_sentence_with_type(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
    concat_entity = []
    sentence1: str
    sentence2: str

    if subj_start < obj_start:
        sentence1 = text[:obj_start] + f'[OBJ;{obj_type}]' + text[obj_start:obj_end + 1] + f'[/OBJ;{obj_type}]' + text[obj_end + 1:]
        new_text = sentence1[:subj_start] + f'[SUB;{subj_type}]' + sentence1[subj_start:subj_end + 1] + f'[/SUB;{subj_type}]' + sentence1[
                                                                                                      subj_end + 1:]

    else:
        sentence1 = text[:subj_start] + f'[SUB;{subj_type}]' + text[subj_start:subj_end + 1] + f'[/SUB;{subj_type}]' + text[subj_end + 1:]
        new_text = sentence1[:obj_start] + f'[OBJ;{obj_type}]' + sentence1[obj_start:obj_end + 1] + f'[/OBJ;{obj_type}]' + sentence1[
                                                                                                   obj_end + 1:]
    return new_text

# entity --> [SUBJ:type]
def swap_entity_token_with_type(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):    
  if subj_start < obj_start:
    new_text = text[:subj_start] + f'[SUBJ:{subj_type}]' + text[subj_end+1:obj_start] + f'[OBJ:{obj_type}]' + text[obj_end+1:]
  else:
    new_text = text[:obj_start] + f'[OBJ:{obj_type}]' + text[obj_end+1:subj_start] + f'[SUBJ:{subj_type}]' + text[subj_end+1:]
  return new_text

def default_sent(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
    return text


def add_entity_type_punct_kr_subj_obj(text, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
    TYPE = {'ORG':'단체', 'PER':'사람', 'DAT':'날짜', 'LOC':'위치', 'POH':'기타', 'NOH':'수량'}
    subj_word = text[subj_start:subj_end + 1]
    obj_word = text[obj_start:obj_end + 1]
    
    if subj_start < obj_start:
      new_text = text[:subj_start] + f'@*{TYPE[subj_type]}*{subj_word}@' + text[subj_end + 1:obj_start] + \
                 f'#^{TYPE[obj_type]}^{obj_word}#' + text[obj_end + 1:]
    else:
      new_text = text[:obj_start] + f'#^{TYPE[obj_type]}^{obj_word}#' + text[obj_end + 1:subj_start] + \
                 f'@*{TYPE[subj_type]}*{subj_word}@' + text[subj_end + 1:]
    
    return new_text