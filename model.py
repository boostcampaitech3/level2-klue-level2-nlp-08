import torch.nn as nn
from promise.dataloader import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, BertPreTrainedModel, \
    RobertaPreTrainedModel
import torch
import torch.nn.init as init
import numpy as np
from time import sleep

from transformers.modeling_outputs import SequenceClassifierOutput

def get_model(MODEL_NAME, tokenizer, model_default=True):
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    if model_default:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    else:
        model = MyRobertaForSequenceClassification(config=model_config)
        model.resize_token_embeddings(len(tokenizer))

    return model


class MyRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = AutoModel.from_pretrained("klue/roberta-large", config=config, add_pooling_layer=False)
        self.classification = MyRobertaClassificationHead(config)
        # self.init_weights()
        # self.reset_parameters()

    def reset_parameters(self):
        for m in self.classification.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)


    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels = None,
            SUB = None,
            OBJ = None,
            return_dict = None

    ):
        # feature_loc = [SUB, OBJ]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               return_dict = return_dict
                               )

        sequence_output = outputs[0]
        # 이 부분을 수정

        logits = self.classification(features=sequence_output, SUB=SUB, OBJ=OBJ)
        loss=None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MyRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size*2)
        self.dropout = nn.Dropout(p = 0.5)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.dense2 = nn.Linear(config.hidden_size*2, config.hidden_size)

    def forward(self, features, SUB, OBJ, **kwargs):
        tmp = None
        for idx2 in range(len(SUB)):
            sub = features[idx2, SUB[idx2], :]
            obj = features[idx2, OBJ[idx2], :]

            tmp1 = torch.cat([sub, obj]).unsqueeze(0)

            if tmp is None:
                tmp = tmp1
            else:
                tmp = torch.cat([tmp, tmp1], dim=0)

        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        tmp = self.dropout(tmp)
        x = self.dense2(tmp)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

if __name__=="__main__":
    MODEL_NAME = "klue/roberta-large"
    get_model(MODEL_NAME=MODEL_NAME, tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME))