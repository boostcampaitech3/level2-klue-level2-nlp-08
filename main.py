from equip import *
from inference import *
from load_data import *
from metric import *
from model import *
from tokenizing import *
from train import *
from config import *

def seed_setting(random_seed):
  os.environ['PYTHONHASHSEED'] = str(random_seed)

  # pytorch, numpy random seed 고정
  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  torch.backends.cudnn.benchmark = False
  torch.cuda.manual_seed(random_seed)
  random.seed(random_seed)

def main():
  ##############READ CONFIG###############
  config = makeArgument()

  print("=" * 10 + "READ CONFIG END" + "=" * 10)

  ##############SEED SETTING###############
  seed_setting(config['seed'])

  print("="*10+"SEED SETTING END"+"="*10)

  ##############LOAD DATA###############
  raw_data = load_data(config)

  print("=" * 10 + "LOAD DATA END" + "=" * 10)

  ##############TOKENIZING DATA###############
  tokenizer, RE_train_data, RE_test_data = tokenizing_data(train_dataset=raw_data, config = config)

  print("=" * 10 + "TOKENIZING END" + "=" * 10)

  ##############Training###############
  train(config, tokenizer, RE_train_data, RE_test_data)

  print("=" * 10 + "Train END" + "=" * 10)
  return
  ##############INFERENCE###############
  inference_main(TOKENIZER_NAME=config['tokenizer_name'])

  print("=" * 10 + "END" + "=" * 10)

if __name__ == '__main__':
  main()