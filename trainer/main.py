import json
from ensemble_train import ensemble_train
from valid_train import valid_train
from train_base import train_base

def main():
  with open('./trainer/config.json', 'r') as f:
    config = json.load(f)

    if config['train_type'] == 'valid_train':
      valid_train(config)
    elif config['train_type'] == 'ensemble_train':
      ensemble_train(config)
    else:
      train_base(config)

if __name__ == '__main__':
    main()