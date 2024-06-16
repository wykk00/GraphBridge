import torch
import transformers
from utils.utils import model_id
import os


class GenerateDataset(torch.utils.data.Dataset):
    """
    Dataset for generating token embeddings
    """
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item
    
    def __len__(self):
        return self.encodings['input_ids'].size(0)
    
class ReductionDataset(torch.utils.data.Dataset):
    """
    Dataset for training reduction module
    """
    def __init__(self, embeddings, attention_mask, neighbor_embeddings, labels):
        self.embeddings = embeddings
        self.attention_mask = attention_mask
        self.neighbor_embeddings = neighbor_embeddings
        self.labels = labels
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.attention_mask[idx], self.neighbor_embeddings[idx], self.labels[idx]
    
    def __len__(self):
        return self.embeddings.size(0)

class NCDataset(torch.utils.data.Dataset):
    """
    Dataset for fine-tuning language models
    """
    def __init__(self, graphs, labels, tokenizer, text, config):
        self.labels = labels
        self.tokenizer = tokenizer        
        self.config = config
        
        self.reduction_tokenizer = transformers.AutoTokenizer.from_pretrained(model_id[self.config.reduction_lm_type])
        
        if self.config.use_reduction:
            reduction_file = os.path.join('out', 'reduction_out', f'{config.reduction_lm_type}', f'{config.dataset}.pt')
            print(f"Loading reduction file : {reduction_file}")
            save_reduction = torch.load(reduction_file)
            self.scores = save_reduction['score']
            self.attention_masks = save_reduction['attention_mask']
            self.encodings = save_reduction['encodings']
        else:
            # tokenizing the text
            self.encodings = [tokenizer.encode(txt, add_special_tokens=False) for txt in text]
            
        self.input_ids = []     
        self.attention_mask = []
        self.root_mask = []
               
        for graph in graphs:
            ids, input_mask, graph_root_mask = self._process(graph)
            self.input_ids.append(torch.tensor(ids))
            self.attention_mask.append(torch.tensor(input_mask))
            self.root_mask.append(torch.tensor(graph_root_mask))
        
        # Padding
        self.input_ids = torch.nn.utils.rnn.pad_sequence(
            self.input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        self.attention_mask = torch.nn.utils.rnn.pad_sequence(
            self.attention_mask, batch_first=True, padding_value=0
        )
        
        self.root_mask = torch.nn.utils.rnn.pad_sequence(
            self.root_mask, batch_first=True, padding_value=0
        )

    def _token_reduction_select(self, original_idx, doc_max_length):
        select_encodings = []
        
        for idx in original_idx:
            score = self.scores[idx]
            attention_mask = self.attention_masks[idx]
            encoding = self.encodings[idx]
            
            valid_token_len = attention_mask.sum().item()
            
            if self.config.lm_type == self.config.reduction_lm_type:  
                        
                # select topk importance tokens
                _, indices = score.topk(doc_max_length - 1 if valid_token_len >= doc_max_length - 1 else valid_token_len)
                
                # keep the tokens original position
                indices, _ = indices.sort() 
                
                select_encodings.append(encoding[indices].tolist())
            
            else:                
                num_tokens = int(doc_max_length * 1.2)
                _, indices = score.topk(num_tokens if valid_token_len >= num_tokens else valid_token_len)
                indices, _ = indices.sort()
                
                # First decode due to different tokenizer
                txt = self.reduction_tokenizer.decode(encoding[indices], skip_special_tokens=True)
                
                # encode using target LMs tokenizer
                enc = self.tokenizer.encode(txt, add_special_tokens=False)
                
                select_encodings.append(enc[:doc_max_length - 1])
                
        return select_encodings
                

    def _process(self, graph):
        origin_idx = graph.original_idx.tolist()
        
        root_origin_idx = graph.original_idx[graph.root_n_index].item()
        origin_idx.remove(root_origin_idx)
        origin_idx.append(root_origin_idx)
        graph.root_n_index = graph.center = len(origin_idx) - 1
        
        doc_max_length = self.config.max_length // graph.num_nodes
        root_mask = []
        ids = []
        
        if self.config.use_reduction:
            _encodings = self._token_reduction_select(origin_idx, doc_max_length)
        
        for i, idx in enumerate(origin_idx):
            
            if self.config.use_reduction:
                # select important tokens via redcution module
                token_ids = _encodings[i]
            else:
                token_ids = self.encodings[idx][:doc_max_length - 1]
            
            # Add [SEP] token between nodes' text
            token_ids.append(self.tokenizer.sep_token_id)
                      
            if i == graph.root_n_index:
                _root_mask = [1 for _ in range(len(token_ids))]
                _root_mask[-1] = 0
            else:
                _root_mask = [0 for _ in range(len(token_ids))]
            
            root_mask.extend(_root_mask)
            
            ids.extend(token_ids)   
            
        input_mask = [1 for _ in range(len(ids))]
            
        return ids, input_mask, root_mask
          
        
    def __getitem__(self, idx):
        item = {'input_ids': self.input_ids[idx], \
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx],
                'root_mask': self.root_mask[idx]}
        
        return item
    
    def __len__(self):
        return len(self.labels)