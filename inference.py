from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader

from MyDataset import RE_Dataset, My_RE_Dataset
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter
from utils import *

from metric import label_to_num
from model import MyRobertaForSequenceClassification, get_model
from tokenizing import tokenized_dataset, get_tokenizer


def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False)
  model.to(device)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(input_ids = data['input_ids'].to(device), token_type_ids=data['token_type_ids'].to(device),
                      OBJ = data['OBJ'].to(device),
                      SUB = data['SUB'].to(device))
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)
    """
    for idx in range(len(prob)):
      if prob[idx][result[idx]] < 0.7 and result[idx]==0:
        tmp_save = prob[idx][result[idx]]
        prob[idx][result[idx]] = 0
        result[idx] = np.argmax(prob[idx])
        prob[idx][result[idx]] = tmp_save
    """

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('./dict_num/dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer, entity_tk_type='add_entity_type_punct_kr'):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir,entity_tk_type)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args, MODE:str = "default"):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = "klue/roberta-large"
  tokenizer = get_tokenizer(tokenizer_name = Tokenizer_NAME, MODE="token")

  ## load test dataset
  test_dataset_dir = "../dataset/test/test_data.csv"

  # csv file name
  file_name = 'submission.csv'

  # sentence preprocessing type
  entity_tk_type = 'special_token_sentence_with_punct'

  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, entity_tk_type)

  Re_test_dataset = My_RE_Dataset(test_dataset, test_label)

  ## load my model
  if MODE=="default":
    ## load my model
    MODEL_NAME = args.model_dir # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    # model.resize_token_embeddings(len(tokenizer))

    model.parameters
    model.to(device)
    print(model.get_input_embeddings())

    ## predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론

  elif MODE=="DJ":
    model = torch.load('./best_model/model.pt')
    pred_answer, output_prob = inference(model, Re_test_dataset, device)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

  # hard voting
  elif MODE=='HV':
    pred_answer_list = []
    output_prob_list = []

    for i in range(1, args.ensemble_num + 1):
      MODEL_NAME = args.model_dir + '_' + str(i)  # model dir.
      model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

      model.to(device)
      temp_pred_answer, temp_output_prob = inference(model, Re_test_dataset, device)  # model에서 class 추론
      pred_answer_list.append(temp_pred_answer)
      output_prob_list.append(temp_output_prob)

    output_prob = []
    pred_answer = []
    for j in range(len(output_prob_list[0])):
      prob = []
      for k in range(30):
        c = 0
        for i in range(args.ensemble_num):
          c += output_prob_list[i][j][k]
        prob.append(c / args.ensemble_num)
      output_prob.append(prob)
      pred_answer.append(np.argmax(prob))

    pred_answer = num_to_label(pred_answer)
  # soft voting
  elif MODE == 'SV':
    pred_answer_list = []
    output_prob_list = []

    for i in range(1, args.ensemble_num + 1):
      MODEL_NAME = args.model_dir + '_' + str(i)  # model dir.
      model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
      model.parameters
      model.to(device)
      temp_pred_answer, temp_output_prob = inference(model, Re_test_dataset, device)  # model에서 class 추론
      pred_answer_list.append(temp_pred_answer)
      output_prob_list.append(temp_output_prob)
    output_prob = []
    pred_answer = []
    for j in range(len(output_prob_list[0])):
      prob = []
      for k in range(30):
        c = 0
        for i in range(args.ensemble_num):
          c += output_prob_list[i][j][k]
        prob.append(c / args.ensemble_num)
      output_prob.append(prob)
      pred_answer.append(np.argmax(prob))

  pred_answer = num_to_label(pred_answer)
    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.


  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('../prediction/'+file_name, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')

if __name__ == '__main__':
  MODE = "SV"

  parser = argparse.ArgumentParser()
  run_time = "runname setting"
  if MODE == "HV" or "SV":
    parser.add_argument('--ensemble_num','-N', type=int, default=3, help='the number of ensemble models')
  # model dir
  parser.add_argument('--model_dir','-M', type=str, default="./best_model/" + run_time, help='inference model name')

  args = parser.parse_args()
  print(args)
  main(args, MODE)