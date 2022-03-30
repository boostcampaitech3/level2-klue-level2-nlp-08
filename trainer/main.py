import json
from trainer.ensemble_train import ensemble_train
from trainer.valid_train import valid_train
from trainer.train_base import train_base

def main():
  with open('config.json', 'r') as f:
    config = json.load(f)

    if config['train_type'] == 'valid_train':
      valid_train(config)
    elif config['train_type'] == 'ensemble_train':
      ensemble_train(config)
    else:
      train_base(config)

if __name__ == '__main__':
    main()