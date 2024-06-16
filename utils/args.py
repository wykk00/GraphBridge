import argparse

class Arguments:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        
        # Dataset
        self.parser.add_argument('--dataset', type=str, help="dataset name", default='cora')
        # Model configuration
        self.parser.add_argument('--layer_num', type=int, help="the number of model's layers", default=2)
        self.parser.add_argument('--hidden_dim', type=int, help="the hidden dimension", default=64)
        self.parser.add_argument('--dropout', type=float, help="dropout rate", default=0.2)
        self.parser.add_argument('--gnn_model_type', type=str, default="SAGE", help="GNN model type")
        # Training settings
        self.parser.add_argument('--lr', type=float, help="learning rate", default=1e-3)
        self.parser.add_argument('--weight_decay', type=float, help="weight decay", default=5e-5)
        self.parser.add_argument('--epochs', type=int, help="training epochs", default=200)
        self.parser.add_argument('--batch_size', type=int, help="the training batch size", default=256)
        self.parser.add_argument('--eval_batch_size', type=int, help="the eval batch size", default=1500)
        self.parser.add_argument('--runs', type=int, help="runs for training", default=5)
        self.parser.add_argument('--start_seed', type=int, help="start seed", default=42)
        self.parser.add_argument('--beta', type=float, help="weight of loss 2", default=0.1)
        self.parser.add_argument('--label_smoothing', type=float, help="label smoothing rate", default=0.3)
        self.parser.add_argument('--hops', type=int, help="number hops for message passing of token reduction phase", default=2)
        # Early stopping
        self.parser.add_argument('--earlystop', action='store_true', help="earlystop")
        self.parser.add_argument('--patience', type=int, help="the patience of counting", default=20)
        # LoRA configuration
        self.parser.add_argument('--lora', action='store_true', help="use LoRA or not",)      
        self.parser.add_argument('--rank', type=int, help="the rank of LoRA", default=4)
        # LMs configuration
        self.parser.add_argument('--lm_type', type=str, help="the type of lm", default='roberta-base', 
                                 choices=['llama', 'sentencebert', 'roberta-base', 'roberta-large', 'bert', 'longformer-base',])
        self.parser.add_argument('--reduction_lm_type', type=str, default="roberta-base", 
                                 choices=["roberta-base"])
        self.parser.add_argument('--max_length', type=int, help="the max sequence length", default=512)
        self.parser.add_argument('--use_reduction', action='store_true', help="Use reduction algo to select the important tokens")
        # Used for sampling
        self.parser.add_argument('--subsampling', action='store_true', help="subsampling, training with subgraphs")
        self.parser.add_argument('--restart', type=float, help="the restart ratio of random walking", default=0.5)
        self.parser.add_argument('--walk_steps', type=int, help="the steps of random walking", default=64)
        self.parser.add_argument('--k', type=int, help="the hop of neighboors", default=2)
        self.parser.add_argument('--sampler', type=str, help="the choice of sampler, random walk or k-hop sampling", default='rw', 
                                 choices=['rw', 'khop', 'shadow'])
        
        
    def parse_args(self):
        return self.parser.parse_args()
