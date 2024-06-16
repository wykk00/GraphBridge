import torch
import torch.nn as nn
import math


class ReductionModel(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim=64, dropout=0.1):
        super(ReductionModel, self).__init__()
        
        self.dropout = torch.nn.Dropout(p=dropout)
        # reduction model
        self.reduction_model = nn.ModuleList()
        self.key = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Linear(input_dim, hidden_dim)
        
        # classifier
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def inference(self, x, neighbor_embeddings, attention_mask) -> torch.Tensor:
        """
        We only need token score while infering
        """
        query = self.query(x)
        key = self.key(neighbor_embeddings)
        
        key = key.unsqueeze(1)
        
        attention_mask = (1.0 - attention_mask) * -10000.0
        # score [b, s]
        score = query.matmul(key.transpose(1, 2)).squeeze()
        score = score + attention_mask # Neglect the padding token
        score = torch.nn.functional.softmax(score / math.sqrt(x.size(-1)), -1)
        
        return score
    
    def forward(self, x, neighbor_embeddings, attention_mask):
        """
        x : [b, s, d]
        neighbor_embeddings : [b, d]
        """
        query = self.query(x)
        key = self.key(neighbor_embeddings)
            
        key = key.unsqueeze(1)
        
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # score [b, s]
        score = query.matmul(key.transpose(1, 2)).squeeze()
        score = score + attention_mask # Neglect the masked token
        score = torch.nn.functional.softmax(score / math.sqrt(x.size(-1)), -1).unsqueeze(-1)
        
        expand_score = score
        expand_score = self.dropout(expand_score)
        expand_score = score.expand_as(x)
        
        
        pruning_x = x * expand_score
        pruning_x = pruning_x.sum(1)
        
        out = self.classifier(pruning_x)
        return out, score
