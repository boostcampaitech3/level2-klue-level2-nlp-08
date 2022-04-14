import torch.nn as nn
from promise.dataloader import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, BertPreTrainedModel, \
    RobertaPreTrainedModel
import torch
import torch.nn.init as init
import torch.nn.functional as F
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
        # self.lstm = nn.LSTM(1024, 256, batch_first=True, bidirectional=True)
        # self.linear = nn.Linear(256 * 2, self.num_labels)

        self.classification = MyRobertaClassificationHead(config)

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict = return_dict
                            )
        """
        lstm_output, (h, c) = self.lstm(outputs[0])  ## extract the 1st token's embeddings
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        logits = self.linear(hidden.view(-1, 256 * 2))
        """
        sequence_output = outputs[0]

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
        self.dropout = nn.Dropout(p = 0.5)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.dense2 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dense3 = nn.Linear(config.hidden_size*3, config.hidden_size*2)

    def forward(self, features, SUB, OBJ, **kwargs):
        tmp = None
        for idx2 in range(len(SUB)):
            # cls = features[idx2,0,:]
            cls = features[idx2, 0, :]
            sub = features[idx2, SUB[idx2], :]
            obj = features[idx2, OBJ[idx2], :]

            tmp1 = torch.cat([cls, sub])
            tmp1 = torch.cat([tmp1, obj]).unsqueeze(0)
            """
            if SUB[idx2] < OBJ[idx2]:
                tmp1 = torch.cat([sub, obj]).unsqueeze(0)
            else:
                tmp1 = torch.cat([obj, sub]).unsqueeze(0)
            """
            if tmp is None:
                tmp = tmp1
            else:
                tmp = torch.cat([tmp, tmp1], dim=0)

        tmp = self.dropout(tmp)
        x = self.dense3(tmp)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.out_proj(x)

        return x

if __name__=="__main__":
    MODEL_NAME = "klue/roberta-large"
    get_model(MODEL_NAME=MODEL_NAME, tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME))