import torch.nn as nn
from transformers import PreTrainedModel
from transformers import LlamaPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from utils.utils import mean_pooling
from torch.cuda.amp import autocast


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().init()
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        
    def forward(self, x):
        pass

class LlamaClassifier(LlamaPreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink='', use_cls=True):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        self.use_cls = use_cls
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        # init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                root_mask=None):
        with autocast():
            
            outputs = self.bert_encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        return_dict=return_dict,
                                        output_hidden_states=True)
            
            
            emb = self.dropout(outputs['hidden_states'][-1])
            
            emb = mean_pooling(emb, root_mask)
            if self.feat_shrink:
                emb = self.feat_shrink_layer(emb)
                    
        logits = self.classifier(emb)

        loss = self.loss_func(logits, labels)
        
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=emb)

class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink='', use_cls=True):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        self.use_cls = use_cls
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        # init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                root_mask=None):
       
        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
    
        emb = self.dropout(outputs['hidden_states'][-1])
        emb = mean_pooling(emb, root_mask)
        
        if self.feat_shrink:
            emb = self.feat_shrink_layer(emb)
            
        logits = self.classifier(emb)

        loss = self.loss_func(logits, labels)
        
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=emb)

