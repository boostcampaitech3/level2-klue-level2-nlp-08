import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel
import torch

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


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p = 0.5)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MyModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = RobertaModel(config, add_pooling_layer=False)
        self.last_layer = RobertaClassificationHead(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,

    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )

        cls_output = outputs[0]
        logits = self.last_layer(cls_output)
        # logits = nn.tanh(logits)
        return logits
