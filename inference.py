from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
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
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  ## load test dataset
  test_dataset_dir = "../dataset/test/test_data.csv"
  # TODO : ../../으로 되어 있는데 test_data.csv 위치 확인!
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset, test_label)

  ## load my model
  if MODE=="default":
    ## load my model
    MODEL_NAME = args.model_dir # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.parameters
    model.to(device)
    print(model.get_input_embeddings())

    ## predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
    pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.

  else:
    if args.ensemble==False:
      MODEL_NAME = args.model_dir  # model dir.
      model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
      model.parameters
      model.to(device)

      pred_answer, output_prob = inference(model, Re_test_dataset, device)  # model에서 class 추론
      pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환
    else:
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
      output_prob = output_prob_list[0]
      pred_answer = []
      for idx in range(len(pred_answer_list[0])):
        # output_prob = output_prob_list[0]
        c = Counter([pred_answer_list[n][idx] for n in range(0, args.ensemble_num)])
        pred_answer.append(c.most_common(1)[0][0])
      pred_answer = num_to_label(pred_answer)

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/RL_typed_punct.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')

if __name__ == '__main__':
  MODE = "default"

  parser = argparse.ArgumentParser()
  run_time = "Not_Setting"
  # model dir
  if MODE == "bolim":
    parser.add_argument('--ensemble', type=bool, default=False)
    parser.add_argument('--ensemble_num', type=int, default=3)
    parser.add_argument('--model_dir', type=str, default="./best_model/"+run_time)
  else:
    parser.add_argument('--model_dir', type=str, default="./best_model/"+run_time)

  args = parser.parse_args()
  print(args)
  main(args, MODE)
  
