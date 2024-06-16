import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from utils.args import Arguments
from utils.metrics import accuracy
from utils.sampling import prepare_dataloader
from utils.utils import set_seed
from data.load import load_data
from models.GNNs import GNN_Model
import os
from tqdm import tqdm


def train(model, data, train_loader, subgraph_loader, optimizer, split_idx, config, device):
        model = model.to(device)
        patience = config.patience
        best_val = 0
        best_test_fromval = 0
        cnt = 0
        
        tqd = tqdm(range(config.epochs), desc='Training', unit='item')
        for epoch in tqd:
            loss, acc = training_step(model, train_loader, optimizer, split_idx, device)
            
            train_acc, val_acc, test_acc = eval(model, data, subgraph_loader, split_idx, device)

            tqd.set_description(f"loss : {loss:.3f}, train_acc : {train_acc:.3f}, val_acc : {val_acc:.3f}, test_acc : {test_acc:.3f}")
            
            if val_acc > best_val:
                best_val = val_acc
                best_test_fromval = test_acc
                cnt = 0
            if config.earlystop:
                if val_acc <= best_val:
                    cnt += 1
                    if cnt >= patience:
                        print(f'early stop at epoch {epoch}')
                        break
                
        return best_test_fromval

def training_step(model, train_loader, optimizer, split_idx, device,):
    model.train()

    total_loss = total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x.to(device), batch.edge_index.to(device))[: batch.batch_size]
        y = batch.y[: batch.batch_size].squeeze().to(device)
        loss = F.cross_entropy(out, y, label_smoothing=config.label_smoothing)
        loss.backward()
        optimizer.step()

        loss_without_smoothing = F.cross_entropy(out, y)

        total_loss += float(loss_without_smoothing)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / split_idx["train"].size(0)

    return loss, approx_acc

@torch.no_grad()
def eval(model, data, subgraph_loader, split_idx, device):
    model.eval()
    out = model.inference(data.x, device, subgraph_loader)
    y_true = data.y
    
    train_acc = accuracy(out[split_idx['train']], y_true[split_idx['train']])
    val_acc = accuracy(out[split_idx['valid']], y_true[split_idx['valid']])
    test_acc = accuracy(out[split_idx['test']], y_true[split_idx['test']])

    return train_acc, val_acc, test_acc

   

def prepare_model(config):
    assert config.gnn_model_type in ["SAGE", "GCN", "SGC"]
    model_class = GNN_Model[config.gnn_model_type]
    return model_class

if __name__ == '__main__':
    config = Arguments().parse_args()
    print(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    acc_list = []
    
    for run in range(config.runs):
        print(f"{'#' * 5}run : {run + 1}{'#' * 5}")
        random_seed = config.start_seed + run
        set_seed(random_seed)
        
        data, _, num_classes = load_data(config.dataset, use_text=True, seed=run)
        
        # Loading node representations from language models
        bert_x_path = os.path.join('out', 'lm_emb', config.lm_type, f'{config.dataset}.pt')
        if not os.path.exists(bert_x_path):
            assert FileNotFoundError(f"{bert_x_path} Not Found")
        
        bert_x = torch.load(bert_x_path)
        data.x = bert_x
        
        train_idx = data.train_mask.nonzero().squeeze()
        val_idx = data.val_mask.nonzero().squeeze()
        test_idx = data.test_mask.nonzero().squeeze() 
        split_idx = {
            'train': train_idx,
            'valid': val_idx,
            'test': test_idx,
        }
        
        train_loader, subgraph_loader = prepare_dataloader(config, data, split_idx)
    
        model = prepare_model(config)(data.x.size(-1), config.hidden_dim, num_classes, config.layer_num, dropout=config.dropout)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        best_test_acc = train(model, data, train_loader, subgraph_loader, optimizer, split_idx, config, device)
        acc_list.append(best_test_acc)
        print(f"# result for run {run + 1} : {best_test_acc}")
        
        del model
    
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"final_acc: {final_acc * 100:.2f} ± {final_acc_std * 100:.2f}")
    