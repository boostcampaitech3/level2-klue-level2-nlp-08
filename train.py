import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, \
    AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, \
    RobertaForSequenceClassification, BertTokenizer, BertModel
from load_data import *
import wandb
import random
from transformers import XLNetTokenizer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size = 768, num_classes = 30, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.model = bert_model
        self.dr_rate = dr_rate
        print(self.model)

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids = input_ids, attention_mask = attention_mask)
        if self.dr_rate:
            out = self.dropout(out.pooler_output)
        else:
            out = out.pooler_output
        real_out = self.classifier(out)
        return real_out

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  MODEL_NAME = "skt/kobert-base-v1"
  # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer = XLNetTokenizer.from_pretrained('skt/kobert-base-v1',
                                              sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
  # load dataset
  train_dataset = load_data("../dataset/train/train.csv")
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  train_label = label_to_num(train_dataset['label'].values)
  # dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)

  # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  mode = BertModel.from_pretrained('skt/kobert-base-v1')
  model = BERTClassifier(mode, dr_rate=0.5).to(device)

  optimizer = optim.AdamW(model.parameters(), lr=5e-5)
  loss_fn = nn.CrossEntropyLoss().to(device)
  model.parameters
  model.to(device)

  num_epochs = 20
  train_dataloader = torch.utils.data.DataLoader(RE_train_dataset, batch_size=32, num_workers=5)

  for epoch in range(num_epochs):
      model.train()
      for batch_id, val in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        out = model(input_ids = val['input_ids'].clone().detach().to(device),
                    attention_mask = val['attention_mask'].clone().detach().to(device))

        loss = loss_fn(out, val['labels'].to(device))
        loss.backward()
        optimizer.step()
      test_acc = 0.0

      model.eval()
      for batch_id, val in enumerate(tqdm(train_dataloader)):
              out = model(input_ids=val['input_ids'].clone().detach().to(device),
                          attention_mask=val['attention_mask'].clone().detach().to(device))

              test_acc += calc_accuracy(out, val['labels'].to(device))
      print("Epoch : {}, Accuracy : {}".format(epoch, test_acc/(batch_id+1)))
  """
  wandb.init(
      project="KLUE",
      entity="miml",
      name="dongjin_2_BERT_{KoBERTTOkenizer}"
  )

  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=20,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    #report_to='wandb',
    fp16=True,
    fp16_opt_level="O1",  # "O1" for typical use
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_train_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')
  """
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def seed_setting(random_seed):
  '''
  setting random seed for further reproduction
  :param random_seed:
  :return:
  '''
  os.environ['PYTHONHASHSEED'] = str(random_seed)

  # pytorch, numpy random seed 고정
  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  # CuDNN 고정
  # torch.backends.cudnn.deterministic = True # 고정하면 학습이 느려진다고 합니다.
  torch.backends.cudnn.benchmark = False
  # GPU 난수 생성
  torch.cuda.manual_seed(random_seed)
  # transforms에서 사용하는 random 라이브러리 고정
  random.seed(random_seed)

def main():
  seed_setting(1004)
  train()

if __name__ == '__main__':
  main()
