from transformers import LlamaModel, AutoTokenizer, AutoModel, Trainer, default_data_collator, TrainerCallback, TrainingArguments
from contextlib import nullcontext
import torch
import torch.distributed as dist
import numpy as np
from utils.args import Arguments
from utils.dist import is_dist, set_dist_env
from utils.metrics import accuracy
from utils.peft import create_lora_config
from utils.utils import model_id
from models.LMs import BertClassifier, LlamaClassifier
from data.load import load_data
from data.dataset import NCDataset
from data.sampling import collect_subgraphs
import os
from copy import deepcopy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def collect_txt(idx, txt):
    tmp = []
    for i in idx:
        tmp.append(txt[i])
    return tmp

if __name__ == '__main__':
    if is_dist():
        rank = set_dist_env()
     
    config = Arguments().parse_args()
    print(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = f"tmp"
    epochs = config.epochs
    enable_profiler = False
    
    # Set up profiler
    if enable_profiler:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        
        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler
                
            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()
    
    acc_list = []
    best_val_acc = 0
    
    for run in range(config.runs):
        
        data, text, num_classes = load_data(config.dataset, use_text=True, seed=run)

        # Load model from HuggingFace Hub
        if config.lm_type == 'llama':
            tokenizer = AutoTokenizer.from_pretrained(model_id[config.lm_type], use_fast=False, trust_remote_code=True)
            bert_model = LlamaModel.from_pretrained(model_id[config.lm_type], device_map=rank if is_dist() else device, torch_dtype=torch.float16)
            tokenizer.sep_token = tokenizer.bos_token
            bert_model.config.sep_token_id = bert_model.config.bos_token_id
            tokenizer.pad_token = tokenizer.eos_token
            bert_model.config.pad_token_id = bert_model.config.eos_token_id
        else:
            bert_model = AutoModel.from_pretrained(model_id[config.lm_type], output_hidden_states=True, return_dict=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id[config.lm_type])
        
        train_idx = data.train_mask.nonzero().squeeze()
        val_idx = data.val_mask.nonzero().squeeze()
        test_idx = data.test_mask.nonzero().squeeze() 

        graphs = collect_subgraphs(torch.arange(data.num_nodes), data, walk_steps=config.walk_steps, restart_ratio=config.restart)
        dataset = NCDataset(graphs, data.y, tokenizer, text, config)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset =  torch.utils.data.Subset(dataset, test_idx)
        
        if config.lora:
            bert_model, _ = create_lora_config(bert_model, config.rank)
            if config.lm_type == 'llama':
                for param in bert_model.parameters():
                    if param.requires_grad:
                        param.data = param.data.float()
        
        
        if config.lm_type == 'llama':
            model = LlamaClassifier(bert_model, num_classes)
        else:
            model = BertClassifier(bert_model, num_classes)
        
        if is_dist():
            dist.barrier()
            
        # Define training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            report_to="none",
            bf16=False,  # Use BF16 if available
            dataloader_pin_memory=False,
            # logging strategies
            logging_dir=f"{output_dir}/logs",
            logging_strategy="epoch",
            logging_steps=1,
            save_strategy="no",
            optim="adamw_torch_fused",
            max_steps=total_steps if enable_profiler else -1,
            learning_rate=config.lr,
            num_train_epochs=epochs,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=config.batch_size,
            gradient_checkpointing=False,
            local_rank=rank if is_dist() else -1,
        )

        with profiler:
            # Create Trainer instance
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=default_data_collator,
                callbacks=[profiler_callback] if enable_profiler else [],
            )

            # Start training
            trainer.train()

        # evaluation
        predictions = trainer.predict(dataset)
        train_predictions, train_labels = predictions.predictions[0][train_idx], predictions.label_ids[train_idx]
        val_predictions, val_labels = predictions.predictions[0][val_idx], predictions.label_ids[val_idx]
        test_predictions, test_labels = predictions.predictions[0][test_idx], predictions.label_ids[test_idx]
        
        if not is_dist() or rank == 0:
            # report acc
            train_acc = accuracy(train_predictions, train_labels)
            test_acc = accuracy(test_predictions, test_labels)
            val_acc = accuracy(val_predictions, val_labels)
            
            print(f"# lr : {config.lr}  run : {run} , train acc : {train_acc * 100:.2f}, val acc : {val_acc * 100:.2f} , test acc : {test_acc * 100:.2f}")
            acc_list.append(test_acc)
          
            embedding = predictions.predictions[1]
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_prediction = deepcopy(embedding)
        
        # clear cache
        del predictions
        del trainer
        del model
        del tokenizer
        del bert_model
        
        if is_dist():
            dist.barrier()
        
            
    if not is_dist() or rank == 0:
        final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
        print(f"final_acc: {final_acc * 100:.2f} ± {final_acc_std * 100:.2f}")
        out_dir = os.path.join('out', 'lm_emb', f"{config.lm_type}")
        os.makedirs(out_dir, exist_ok=True)
        
        torch.save(torch.tensor(best_prediction), os.path.join(out_dir, f'{config.dataset}.pt'))