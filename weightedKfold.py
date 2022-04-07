import os, argparse
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
)
from utils import *

def main(args):
    checkpoints = search(args.path, args.model_name)

    model = []
    for checkpoint in checkpoints:
        model.append(AutoModelForSequenceClassification.from_pretrained(args.path + '/' + checkpoint))

    # 모델의 state_dict 가중치 평균 구하기
    # 1. 가중치 누적합
    for i in range(1,len(model)):
        for param_tensor in model[i].state_dict()[1:]:
            model[0].state_dict()[param_tensor] += model[i].state_dict()[param_tensor]
    # 2. 가중치 평균
    for param_tensor in model[0].state_dict()[1:]:
        model[0].state_dict()[param_tensor] = (model[0].state_dict()[param_tensor]/len(model)).float()

    weighted_model = model[0]
    weighted_model.save_pretrained(f"{args.path}/{args.model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  
    # model dir
    parser.add_argument("--model_name", type=str, default="model_name")
    parser.add_argument("--path", type=str, default="./best_model")
    args = parser.parse_args()
    print(args)
    main(args)
