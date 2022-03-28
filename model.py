import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.init as init

def get_model(MODEL_NAME, tokenizer):
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    origin_roberta = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    """
    for param in origin_roberta.classifier.modules():
        if isinstance(param, nn.Linear):
            init.xavier_normal_(param.weight.data)
            param.bias.data.fill_(0)
    """
    # myModel에서 linear_layer라는 함수를 추가시켰다고 가정하자
    # origin_roberta.classifier = MyRobertaClassificationHead(config=model_config)
    # print(origin_roberta)
    origin_roberta.resize_token_embeddings(len(tokenizer))

    return origin_roberta

class MyRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size*2)
        self.dropout = nn.Dropout(p = 0.5)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.dense2 = nn.Linear(config.hidden_size*2, config.hidden_size)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x