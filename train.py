from torch.utils.data import DataLoader
from transformers import TrainingArguments

from equip import *

import wandb, os, random

from metric import *
from model import *
from load_data import *
from tokenizing import *
from equip import *

import tqdm

def train():
    MODEL_NAME = "klue/roberta-large"

    train_dataset = load_data("../dataset/train/train.csv")

    tokenizer = get_tokenizer(MODEL_NAME)

    train_tokenized = tokenized_dataset(train_dataset, tokenizer)

    RE_train_dataset = RE_Dataset(train_tokenized, train_dataset['label'])


    RE_train_data, RE_test_data = train_test_split(RE_train_dataset, test_size=0.2,
                                              shuffle=True, stratify=train_dataset['label'])

    model = get_model(MODEL_NAME, tokenizer)

    train_dataloader = DataLoader(RE_train_data, shuffle=True, num_workers=5, batch_size=32)

    device = torch.device("cuda:0")

    optimizer = get_optimizer(model)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 500,
        num_training_steps = 20 * len(train_dataset),
        num_cycles = 4
    )
    criterion = get_loss()
    optimizers = (optimizer, scheduler)

    wandb.init(
        project="KLUE",
        entity="violetto",
        name="dongjin_3_roberta_{Speical_token}"
    )

    for param in model.parameters():
        param.requires_grad = False
        # 파라미터를 학습 대상에서 전부 제외 시킴


    # myModel에서 linear_layer라는 함수를 추가시켰다고 가정하자
    for params in model.classifier.parameters():
        params.requires_grad = True

    epochs = 1
    loss_fn = FocalLoss()

    # config에 저장된 loss_name에 따라 다른 loss 계산


    for epoch in range(epochs):
        model.train()
        for batch_idx, item in enumerate(train_dataloader):
            print(batch_idx, len(train_dataloader))
            labels = item.pop('labels')
            outputs = model(**item)
            optimizer.zero_grad()
            loss = loss_fn(outputs[0], labels)
            loss.backward()
            optimizer.step()

    for param in model.parameters():
        param.requires_grad = True


    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        save_total_limit=5,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=9,  # total number of training epochs
        learning_rate=5e-5,  # learning_rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
       #  warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy='steps',  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        load_best_model_at_end=True,
        metric_for_best_model = "eval_micro f1 score",
        report_to='wandb',
        fp16=True,
        fp16_opt_level="O1"
    )

    trainer = MyTrainer(
        loss = criterion,
        model = model,
        args = training_args,
        train_dataset = RE_train_data,
        eval_dataset = RE_test_data,
        compute_metrics = compute_metrics,
        optimizers = optimizers
    )
    total_steps = math.ceil(len(RE_train_dataset) / training_args.per_device_train_batch_size)
    trainer.create_optimizer_and_scheduler(
        total_steps
    )

    trainer.train()
    model.save_pretrained('./best_model')

def seed_setting(random_seed):
  os.environ['PYTHONHASHSEED'] = str(random_seed)

  # pytorch, numpy random seed 고정
  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  torch.backends.cudnn.benchmark = False
  torch.cuda.manual_seed(random_seed)
  random.seed(random_seed)

def main():
  seed_setting(1004)
  train()

if __name__ == '__main__':
  main()


"""
def train():
    MODEL_NAME = "klue/roberta-large"



    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data("../dataset/train/train.csv")

    train_data, valid_data = train_test_split(train_dataset, test_size=0.1,
                                              shuffle=True, stratify=train_dataset['label'])
    print(train_data.shape)
    print(valid_data.shape)

    train_label = label_to_num(train_data['label'].values)
    valid_label = label_to_num(valid_data['label'].values)

    tokenized_train = tokenized_dataset(train_data, tokenizer)
    tokenized_dev = tokenized_dataset(valid_data, tokenizer)

    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, valid_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.parameters
    model.to(device)

    optimizer = optim.AdamW(model.parameters())
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=500,
                                                                   num_training_steps=20 * len(train_dataset))
    optimizers = (optimizer, scheduler)


    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        save_total_limit=5,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=20,  # total number of training epochs
        learning_rate=5e-5,  # learning_rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy='steps',  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        load_best_model_at_end=True,
        lr_scheduler_type = 'cosine_with_restarts'
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained('./best_model')




def train2():
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      assert str(device)=='cuda:0'

      MODEL_NAME = "klue/roberta-base"
      tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

      ######################## load dataset ############################
      train_dataset = load_data("../dataset/train/train.csv")

      train_data, valid_data = train_test_split(train_dataset, test_size=0.1,
                                                shuffle=True, stratify=train_dataset['label'])

      train_label = label_to_num(train_data['label'].values)
      valid_label = label_to_num(valid_data['label'].values)

      tokenized_train = tokenized_dataset(train_data, tokenizer)
      tokenized_dev = tokenized_dataset(valid_data, tokenizer)

      RE_train_dataset = RE_Dataset(tokenized_train, train_label)
      RE_dev_dataset = RE_Dataset(tokenized_dev, valid_label)

      print("="*10 + "Data_Process_End"+"="*10)

      ########################## Create Model ###########################
      # model = RobertaPreTrainedModel.from_pretrained(MODEL_NAME)

      model_config = AutoConfig.from_pretrained(MODEL_NAME)
      model_config.num_labels = 30

      # myModel = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config).to(device)
      myModel = MyModel(model_config).to(device)

      print("="*10+"Model_End"+"="*10)

      ############### Train ##################

      train_dataloader = DataLoader(RE_train_dataset, batch_size = 32, num_workers=5)
      test_dataloader = DataLoader(RE_dev_dataset, batch_size = 32, num_workers=5,drop_last=True)


      training_args = TrainingArguments(
          output_dir='./results',  # output directory
          save_total_limit=5,  # number of total save model.
          save_steps=500,  # model saving step.
          num_train_epochs=20,  # total number of training epochs
          learning_rate=5e-5,  # learning_rate
          per_device_train_batch_size=32,  # batch size per device during training
          per_device_eval_batch_size=32,  # batch size for evaluation
          warmup_steps=500,  # number of warmup steps for learning rate scheduler
          weight_decay=0.01,  # strength of weight decay
          logging_dir='./logs',  # directory for storing logs
          logging_steps=100,  # log saving step.
          evaluation_strategy='steps',  # evaluation strategy to adopt during training
          # `no`: No evaluation during training.
          # `steps`: Evaluate every `eval_steps`.
          # `epoch`: Evaluate every end of epoch.
          eval_steps=500,  # evaluation step.
          load_best_model_at_end=True,
          # report_to='wandb',
      )

      optimizer = optim.AdamW(myModel.parameters(), lr = 5e-5)
      scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=20*len(train_dataset))
      optimizers = (optimizer, scheduler)
      # MyCallback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0001)

      trainer = Trainer(
          model=myModel,  # the instantiated 🤗 Transformers model to be trained
          args=training_args,  # training arguments, defined above
          train_dataset=RE_train_dataset,  # training dataset
          eval_dataset=RE_dev_dataset,  # evaluation dataset
          compute_metrics=compute_metrics, # define metrics function
          # callbacks=[MyCallback],
          optimizers=optimizers
      )

      # train model
      trainer.train()
      myModel.save_pretrained('./best_model')

      epochs = 30
      for epoch in range(epochs):
          myModel.train()
          for batch_idx, item in enumerate(tqdm(train_dataloader)):
              sentences = item['input_ids'].to(device)
              attention_mask = item['attention_mask'].to(device)
              token_type_ids = item['token_type_ids'].to(device)
              target = item['labels'].to(device)

              pred = myModel.forward(sentences, attention_mask = attention_mask, token_type_ids = token_type_ids)
              logits = pred

              optimizer.zero_grad()
              loss = criterion(logits, target)
              loss.backward()
              optimizer.step()

          with torch.no_grad():
              model.eval()
              acc = 0.0
              for batch_idx, item in enumerate(tqdm(test_dataloader)):
                  sentences = item['input_ids'].to(device)
                  attention_mask = item['attention_mask'].to(device)
                  token_type_ids = item['token_type_ids'].to(device)
                  target = item['labels'].to(device)

                  pred = myModel.forward(sentences, attention_mask = attention_mask, token_type_ids = token_type_ids)
                  acc += calc_accuracy(pred, target)

              print("Epoch : {}, Accuracy : {}".format(epoch, acc/(batch_idx+1)))


      torch.save(myModel.state_dict(), f'./best_model/best.pth')

      print("="*10+"Train End"+"="*10)

      optimizer = optim.AdamW(model.parameters(), lr=5e-5)
      loss_fn = nn.CrossEntropyLoss().to(device)
    
      num_epochs = 20
      train_dataloader = torch.utils.data.DataLoader(RE_train_dataset, batch_size=32, num_workers=5)
      valid_dataloader = torch.utils.data.DataLoader(RE_dev_dataset, batch_size=64, num_workers=5)
    
      best_acc = 0.0
      num = 0
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
          with torch.no_grad():
              for batch_id, val in enumerate(tqdm(valid_dataloader)):
                      out = model(input_ids=val['input_ids'].clone().detach().to(device),
                                  attention_mask=val['attention_mask'].clone().detach().to(device))
    
                      test_acc += calc_accuracy(out, val['labels'].to(device))
              print("Epoch : {}, Accuracy : {}".format(epoch, test_acc/(batch_id+1)))
    
          if best_acc < test_acc: # Model Save
            torch.save(model.state_dict(), f'./best_model/best.pth')
            best_acc = test_acc
            num = 0
          else:
              num = num + 1
    
          if num > 5:
              print("Early Stopping!")
              break
    
    
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