from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch_geometric
from utils.args import Arguments
from utils.utils import mean_pooling, model_id
from data.load import load_data
from data.dataset import GenerateDataset, ReductionDataset
from models.Reduction import ReductionModel
from tqdm import tqdm
from copy import deepcopy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collect_txt(idx, txt):
    tmp = []
    for i in idx:
        tmp.append(txt[i])
    return tmp


def generate(model, dataset, config, device):
    torch.cuda.set_device(device)
    model = model.cuda()
    generation_batch_size = 192
    
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=generation_batch_size,
            shuffle=False, num_workers=6)
    
    model.eval()
    emb = []
    pooling = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Generating', unit='item'):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)['last_hidden_state']
            pooling_out = mean_pooling(out, batch['attention_mask'])
            emb.append(out.to('cpu'))
            pooling.append(pooling_out.to('cpu'))
    
    emb = torch.cat(emb, dim=0)
    pooling = torch.cat(pooling, dim=0)
    
    print(f"{'#' * 5}Generation Complete{'#' * 5}")
    
    return emb, pooling
    

def do_reduction(model, data_loader, device):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in data_loader:
            embeddings, attention_mask, neighbor_embeddings, _ = batch
            embeddings = embeddings.to(device)
            attention_mask = attention_mask.to(device)
            neighbor_embeddings = neighbor_embeddings.to(device)
            
            with autocast():
                score = model.inference(embeddings, neighbor_embeddings, attention_mask)
            
            output.append(score.cpu())
            
    output = torch.cat(output, dim=0)
    
    return output  

def eval_reduction(model, data_loader, device):
    model.eval()
    correct = 0
    total_num = 0
    
    with torch.no_grad():
        with autocast():
            for batch in data_loader:
                embeddings, attention_mask, neighbor_embeddings, labels = batch
                embeddings = embeddings.to(device)
                attention_mask = attention_mask.to(device)
                neighbor_embeddings = neighbor_embeddings.to(device)
                labels = labels.to(device)
                
                with autocast():
                    out, _ = model(embeddings, neighbor_embeddings, attention_mask) 
                
                pred = out.argmax(dim=1)
                
                correct += (labels == pred).sum().item()
                total_num += labels.size(0)
    
    return 1.0 * correct / total_num
    
def train_reduction(model, optimizer, criterion, config, train_loader, val_loader, test_loader, beta=0.1, device=0):
    cnt = 0
    patience = config.patience
    best_val = 0
    best_test_fromval = 0
        
    tqd = tqdm(range(config.epochs), desc='Training', unit='item')
    scaler = GradScaler()

    for epoch in tqd:
        model.train()
        optimizer.zero_grad()
        
        for batch in train_loader:
            embeddings, attention_mask, neighbor_embeddings, labels = batch
            embeddings = embeddings.to(device)
            attention_mask = attention_mask.to(device)
            neighbor_embeddings = neighbor_embeddings.to(device)
            labels = labels.to(device)
            with autocast():
                out, score = model(embeddings, neighbor_embeddings, attention_mask)
                
                # loss1 cross entropy loss
                loss1 = criterion(out, labels)
                # loss2 regularization loss
                uniform_distribution = attention_mask / attention_mask.sum(1).unsqueeze(-1).expand_as(attention_mask)
                loss2 = torch.nn.functional.kl_div(torch.log(score.squeeze(-1) + 1e-5), torch.log(uniform_distribution + 1e-5), 
                                                reduction="batchmean", log_target=True)
            
                loss = loss1 + beta * loss2
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        train_acc = eval_reduction(model, train_loader, device)
        val_acc = eval_reduction(model, val_loader, device)            
        test_acc = eval_reduction(model, test_loader, device)
        
        tqd.set_description(f"loss : {loss:.3f}, train_acc : {train_acc:.3f}, val_acc : {val_acc:.3f}, test_acc : {test_acc:.3f}")
        
        if val_acc > best_val:
            best_val = val_acc
            best_model = deepcopy(model)
            best_test_fromval = test_acc
            cnt = 0
        if config.earlystop:
            if val_acc <= best_val:
                cnt += 1
                if cnt >= patience:
                    print(f'early stop at epoch {epoch}')
                    break
    
    return best_model, best_test_fromval
            
            
if __name__ == '__main__':
    config = Arguments().parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(config)
    
    # Loading Data
    data, text, num_classes = load_data(config.dataset, use_text=True)
    
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_id[config.reduction_lm_type], add_special_tokens=False)
    bert_model = AutoModel.from_pretrained(model_id[config.reduction_lm_type], output_hidden_states=True, return_dict=True, 
                                        torch_dtype=torch.float16)
    
    encodings = tokenizer(text, add_special_tokens=False, truncation=True,
                                        padding=True, return_tensors="pt", max_length=512)
    
    raw_emb_dir = os.path.join('out', "raw_emb", f"{config.reduction_lm_type}")
    os.makedirs(raw_emb_dir, exist_ok=True)
    raw_emb_path = os.path.join(raw_emb_dir, f"{config.dataset}.pt")
    
    if not os.path.exists(raw_emb_path):
        attention_mask = encodings['attention_mask']
        
        dataset = GenerateDataset(encodings=encodings)
        
        # generating text embedding
        embeddings, pooling_embeddings = generate(bert_model, dataset, config, device)
        
        save_dict = {'embeddings': embeddings, 'attention_mask': attention_mask, 'pooling_embeddings': pooling_embeddings}
        torch.save(save_dict, raw_emb_path)

    else:
        save_dict = torch.load(raw_emb_path)
        embeddings = save_dict['embeddings']
        attention_mask = save_dict['attention_mask']
        pooling_embeddings = save_dict['pooling_embeddings']
    
    # message passing w/o parameters, remove self-loop 
    conv = torch_geometric.nn.SimpleConv(aggr="mean", combine_root=None)
    neighbor_embeddings = pooling_embeddings

    for _ in range(config.hops):
        neighbor_embeddings = conv(neighbor_embeddings, data.edge_index)
    
    
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    
    reduction_model = ReductionModel(num_classes, embeddings.size(-1), hidden_dim=config.hidden_dim).to(device)      
        
    # Training Token reduction model
    optimizer = torch.optim.Adam(reduction_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = ReductionDataset(embeddings, attention_mask, neighbor_embeddings, data.y)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset =  torch.utils.data.Subset(dataset, test_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=6)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=config.eval_batch_size, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=config.eval_batch_size, num_workers=6)
    
    reduction_model, acc = train_reduction(reduction_model, optimizer, criterion, config, train_loader, val_loader, test_loader, beta=config.beta, device=device)
    print(f"# final_acc: {acc*100:.2f}")
    
    # inference
    inference_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=config.eval_batch_size, num_workers=6
    )
    
    # score : [b, s]
    score = do_reduction(reduction_model, inference_loader, device)
    
    # Save reduction score & encodings & attention_mask
    save_reduction = {'score': score, 'encodings': encodings['input_ids'], 'attention_mask' : attention_mask}
    save_dir = os.path.join('out', 'reduction_out', f'{config.reduction_lm_type}')

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{config.dataset}.pt')
    
    torch.save(save_reduction, save_path)