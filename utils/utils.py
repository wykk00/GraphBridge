import numpy as np
import torch
import collections
import math
import torch
from utils.dist import is_dist
import os
import random

model_id = {
    'sentencebert': 'sentence-transformers/bert-base-nli-mean-tokens',
    'llama': "./llama/models_hf/7Bf",
    'roberta-base': "FacebookAI/roberta-base",
    'roberta-large': "FacebookAI/roberta-large",
    'bert': 'bert-base-uncased',
    'longformer-base': 'allenai/longformer-base-4096',
}

def _term_frequency(doc):
    # counter
    counter = collections.Counter()
    counter.update(doc)
    return np.array([counter[token] for token in doc], dtype=np.float32)
    
def _inverse_doc_frequency(docs):
    # calulate the document frequency
    df_counter = collections.Counter()
    for doc in docs:
        df_counter.update(set(doc))
        
    # calulate the inverse document frequency (use idf_smooth)
    idf_dict = {k: math.log((len(docs) + 1 )/ (v + 1)) + 1 for k, v in df_counter.items()}
    
    return idf_dict

def tf_idf(docs):
    idf_counter = _inverse_doc_frequency(docs)
    # calculate every token's if-idf for each docs
    tf_idfs = []
    for doc in docs:
        tf = _term_frequency(doc)
        idf = np.array([idf_counter[token] for token in doc])
        tf_idfs.append(tf * idf)
        
    return tf_idfs

def set_seed(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if is_dist():
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    if is_dist():
        gpus = ",".join([str(_) for _ in range(int(os.environ["WORLD_SIZE"]))])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging   
    token_embeddings = model_output # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
