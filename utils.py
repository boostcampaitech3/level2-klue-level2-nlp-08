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
mecab = Mecab()
def add_spTok(text):
    for noun in mecab.nouns(text):
        text = text.replace(noun,'[NER]'+noun+'[/NER]')
    return text
TYPE = {'ORG':'단체정보','PER':'사람정보','DAT':'날짜정보','LOC':'위치정보','POH':'기타정보','NOH':'기타 수량표현'}
def add_punct(text, i_start, i_end, i_type,j_start, j_end, j_type):
    if i_start < j_start:
        new_text = text[:i_start] + '*' + text[i_start:i_end + 1] + '[' + TYPE[i_type] + ']*' + text[i_end + 1:j_start] + '*' + \
               text[j_start:j_end + 1] + '[' + TYPE[j_type] + ']*'  + text[j_end + 1:]
        # new_text = text[:i_start] + '*' + '[' + TYPE[i_type] + ']*' + text[i_end + 1:j_start] + '*' + \
        #            '[' + TYPE[j_type] + ']*' + text[j_end + 1:]
    else:
        new_text = text[:j_start] + '*' + text[j_start:j_end + 1] + '[' + TYPE[j_type] + ']*' + text[j_end + 1:i_start] + '*' + \
               text[i_start:i_end + 1] + '[' + TYPE[i_type] + ']*' + text[i_end + 1:]
        # new_text = text[:j_start] + '*' + '[' + TYPE[j_type] + ']*' + text[j_end + 1:i_start] + '*' + \
        #            '[' + TYPE[i_type] + ']*' + text[i_end + 1:]
    return new_text