from sklearn.model_selection import train_test_split

from MyDataset import *
from train import *
from utils import *
from metric import *
from load_data import *
from tokenizing import *
from model import *


def main():
  torch.cuda.empty_cache()
  MODE = "default"
  run_name = "runname setting"
  ##############SEED SETTING###############
  SEED_NUM = 1004
  seed_everything(SEED_NUM)

  print("="*10+f"SEED_NUM : {SEED_NUM}"+"="*10)

  ##############LOAD DATA###############
  entity_tk_type = 'special_token_sentence_with_punct'
  raw_data = load_data(dataset_dir='../dataset/train/cleaned_train.csv',
                       entity_tk_type=entity_tk_type)

  train_label = label_to_num(raw_data['label'].values)
  """
  entity_tk_type

  add_entity_type_punct_star : *entity[TYPE]*  *entity[TYPE]*
  add_entity_type_suffix_kr : *entity[TP]TYPE[/TP]*  * entity[TP]TYPE[/TP]*
  add_entity_type_punct_kr : @*TYPE*entity@  #^TYPE^entity#
  add_entity_type_token : [subj_type]entity[/subj_type]  [obj_type]entity[/obj_type]
  add_entity_token : [SUBJ]entity[/SUBJ]
  add_entity_token_with_type : [SUBJ:type]entity[/SUBJ]
  special_token_sentence : [OBJ] entity [/OBJ]
  special_token_sentence_with_type : [OBJ;obj_type] object_entity [/OBJ;obj_type], [SUB;subj_type] subject_entity [/SUB;subj_type]
  swap_entity_token_with_type : entity --> [SUBJ:type]
  default_sent
  add_entity_type_punct_kr_subj_obj
  special_token_sentence_with_punct
  """
  print('='*10 + f"Preprocessing type : {entity_tk_type}"+'='*10)
  ##############TOKENIZING DATA###############
  tokenizer_name = "klue/roberta-large"
  tokenizer, train_data = tokenizing_data(train_dataset=raw_data, tokenizer_name=tokenizer_name,
                                          MODE="token")

  print("=" * 10 + f"TOKENIZIER_NAME : {tokenizer_name}" + "=" * 10)

  ##############MAKE DATASET###############
  RE_train_dataset = get_dataset(train_data, train_label, change=True)

  valid = False
  valid_size = 0.1
  if valid:
      RE_train_dataset, RE_dev_dataset = train_test_split(RE_train_dataset, test_size=valid_size,
                                                     shuffle=True, stratify=raw_data['label'])
  else:
      RE_dev_dataset = RE_train_dataset

  if valid:
      wandb.init(
          project='KLUE',
          entity='miml',
          name=run_name
      )
  else:
      wandb.init(
          project='KLUE',
          entity='violetto',
          name=run_name
      )

  ##############GET MODEL###############
  MODEL_NAME = "klue/roberta-large"
  model_default = True
  model = get_model(MODEL_NAME=MODEL_NAME, tokenizer=tokenizer, model_default=model_default)

  print("="*10 + f'MODEL_NAME : {MODEL_NAME}' + '='*10)

  ##############GET MODEL###############
  train(RE_train_dataset=RE_train_dataset, RE_dev_dataset=RE_dev_dataset,
        MODE=MODE, run_name = run_name, model = model, tokenizer = tokenizer)

if __name__ == '__main__':
  main()