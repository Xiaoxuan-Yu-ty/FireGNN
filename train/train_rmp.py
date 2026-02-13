"""
RMPGNN Model training, HPO with Optuna, and testing.
"""
import copy
import os
from typing import Dict, Tuple
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import gc

import argparse
import pickle
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, BinaryPrecision, BinaryRecall, BinarySpecificity,BinaryAveragePrecision,
)
from tqdm import tqdm
import optuna

# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from utils.helper import (
    networkx_to_hetero_data,
    get_edge_features,
    get_device,
    set_random_seeds
)
from hetero_models.hetero_rmp_models import get_hetero_model

# ===== Training step ==========
def train_epoch(model, 
                data:HeteroData, 
                edge_index_dict: Dict, 
                edge_weight_dict:Dict,
                optimizer: torch.optim.Optimizer, 
                scaler,
                lambdas,
                mask,
                device:torch.device,
                clip_grad_norm: float = 1.0,
                negative_ratio: float = 1.0,
                ) -> Dict:
    
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(): # for memory efficiency
        h_dict, log_probs, relevance_history = model(edge_index_dict, edge_weight_dict)
        
        # 1. classification loss
        labels = torch.as_tensor(data["Patient"].y).long().squeeze()
        train_labels = labels[mask]
         # weight for imbalanced calss
        weights = 1.0/torch.bincount(train_labels)
        weights = weights / weights.sum()
        print('Imbalanced class weights: ',weights)
        loss_cls = F.nll_loss(log_probs[mask], train_labels, weight=weights)

        # 2. link prediction loss
        loss_lp = torch.tensor(0.0, device=device)
        num_edeg_types = 0
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index is None or edge_index.size(1) == 0 or 'Patient' not in edge_type:
                continue
            
            pos_edge_index = edge_index
            pos_scores = model.decode(h_dict, pos_edge_index, edge_type)

            # negative sampling to get negative edge_index
            src_type, _, dst_type = edge_type
            num_neg = max(int(pos_edge_index.size(1)*negative_ratio),1)
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=(data[src_type].num_nodes, data[dst_type].num_nodes),
                num_neg_samples=num_neg
            ).to(device)
            neg_scores = model.decode(h_dict, neg_edge_index, edge_type)

            # loss calculation
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            link_labels = torch.cat([torch.ones_like(pos_scores),
                                     torch.zeros_like(neg_scores)], dim=0)
            loss_lp += F.binary_cross_entropy_with_logits(scores, link_labels)
            num_edeg_types += 1
        
        loss_lp = loss_lp / num_edeg_types

        total_loss = lambdas['cls'] * loss_cls + lambdas['lp'] * loss_lp
    # scaled backward to manage memory. 
    scaler.scale(total_loss).backward
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    result = {
        'total_loss': total_loss.detach.item(),
        'cls_loss': loss_cls.detach.item(),
        'lp_loss': loss_lp.detach.item(),
        'relevance_history': relevance_history
    }
    return result

# lambda Scheduler
class LambdaScheduler:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def get_lambdas(self, epoch):
        # learn KG first
        if epoch < 100:
            return {'cls': 0.1, 'lp': 1.0, 'reg': 0.1}
        
        # focus on classification
        elif epoch < 200:
            # Linear ramp for classification: from 0.1 to 1.0
            cls_val = 0.1 + (0.9 * (epoch - 20) / 80)
            return {'cls': cls_val, 'lp': 0.5, 'reg': 0.1}
        
        # sharpen the edge coefficient
        else:
            return {'cls': 1.0, 'lp': 0.1, 'reg': 0.8}

# ====== Evaluation =====
@torch.no_grad()
def evaluate_link_prediction(model, 
                             data: HeteroData,
                             edge_index_dict:Dict, 
                             h_dict:Dict,
                             device:torch.device,  
                             k=10)->float:
    model.eval()
    # sample edge_types to reduce computation cost
    sampled_edge_types = random.sample([et for et in edge_index_dict.keys() if 'Patient' not in et], k=200)
    hits = 0.0
    count = 0.0
    for edge_type in sampled_edge_types:
        src_type, _, dst_type = edge_type
        pos_edge_index = edge_index_dict[edge_type]
        for idx in range(pos_edge_index.size(1)):
            pos_edge_idx = pos_edge_index[:, idx:idx+1]
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_idx,
                num_nodes=(data[src_type].num_nodes, data[dst_type].num_nodes),
                num_neg_samples=100).to(device)
            combined_edge_index = torch.cat([pos_edge_idx, neg_edge_index], dim=1)
            scores = model.decode(h_dict, combined_edge_index, edge_type)
            pos_score = scores[0]
            rank = (scores > pos_score).sum().item() + 1
            if rank <= k:
                hits += 1
            count += 1
    if count == 0:
        return 0.0      
    
    return hits/count

@torch.no_grad()
def evaluate(model, data, edge_index_dict, edge_weight_dict, mask,device)->Dict:
    
    # classification metrics
    f1 = BinaryF1Score() # input: preds[N], targets[N]
    auroc = BinaryAUROC() 
    recall = BinaryRecall()
    precision = BinaryPrecision()
    specificity = BinarySpecificity()
    auprc = BinaryAveragePrecision()

    model.eval()
    h_dict, log_probs, relevance_history = model(edge_index_dict, edge_weight_dict)

    # 1. classification performance
    y = torch.as_tensor(data['Patient'].y).long().squeeze()[mask]
    preds = log_probs.argmax(dim=1)[mask] # -> [N] of 0 or 1, hard classification
    correct = (preds == y).sum().item()
    acc = correct/mask.sum().item()
    f1_score = f1(preds,y)
    recall_score = recall(preds,y)
    precision_score = precision(preds,y)
    specificity_score = specificity(preds,y)

    preds_exp = torch.exp(log_probs)[:,1] # -> [N] of (0,1) probabilities of disease class
    auroc_score = auroc(preds_exp,y)
    auprc_score = auprc(preds_exp,y)


    # 2. link prediction performance
    lp_hits = evaluate_link_prediction(model, data, edge_index_dict, h_dict,device)

    metrics = {
        'accuracy': acc,
        'f1_score': f1_score,
        'recall (tp/(tp+fn))': recall_score,
        'precision (tp/(tp+fp))': precision_score,
        'specificity (tn/(tn+fp))': specificity_score,
        'auroc':auprc_score,
        'auprc':auprc_score,
        'hits@10':lp_hits
    }
    return metrics


# ===== HPO =====
def objective(
        trial:optuna.trial.Trial,
        model_type:str, 
        data:HeteroData,
        device:torch.device,
        epochs:int,
        patientce:float,
        output_dir:str,
        num_negatives:int
)->float:
    
    # set hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    in_channels = trial.suggest_categorical(
        "in_channels", [16, 32, 64, 128]
    )
    hidden_channels = trial.suggest_categorical(
        "hidden_channels", [32, 64, 128, 256]
    )
    out_channels = trial.suggest_categorical("out_channels", [16, 32, 64, 128])
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.6)
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    num_layers = trial.suggest_categorical("GNN layers", [2,3])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    negative_ratio = trial.suggest_float("negative_ratio", 1.0, 5.0)

    # initialize model, optimizer, scheduler
    model = get_hetero_model(model_type=model_type,
                             data=data,
                             in_channels=in_channels,
                             hidden_channels=hidden_channels,
                             out_channels=out_channels,
                             num_layers=num_layers,
                             heads = heads,
                             dropout_rate = dropout_rate
                             ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler()
    lambdas = LambdaScheduler(total_epochs=epochs)

    # prepare model forward input
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    initial_relevance_dict = {nt: data[nt].relevance for nt in data.node_types}
    edge_weight_dict = {}
    for edge_type in data.edge_types:
        if 'Patient' in edge_type:
            ew = data[edge_type].edge_weight
        else:
            ew = None
        edge_weight_dict[edge_type] = ew
    # initialize relevance_dict: turn relevances to learnable parmeters and Patient=0
    model.initialize_relevances(initial_relevance_dict, data)

    best_val_metric = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    epochs_history = []
    validation = []
    # ----- start HPO -----
    try:
        for epoch in tqdm(range(epochs), desc=f"Trial {trial.number}", leave=False):
            current_lambdas = lambdas.get_lambdas(epoch)
            # train with cls, lp losses
            epoch_result = train_epoch(model,
                                    data,
                                    edge_index_dict,
                                    edge_weight_dict,
                                    optimizer,
                                    scaler,
                                    current_lambdas,
                                    data['Patient'].train_mask,
                                    device,
                                    negative_ratio=negative_ratio)
            epochs_history.append(epoch_result)
            # valiate with accuracy
            val_metrics = evaluate(model,
                                    data,
                                    edge_index_dict,
                                    edge_weight_dict,
                                    data['Patient'].val_mask,
                                    device)
            validation.append(val_metrics)
            scheduler.step(val_metrics['f1_score'])

            if val_metrics['f1_score'] > best_val_metric:
                best_val_metric = val_metrics['f1_score']
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            trial.report(best_val_metric, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if epochs_no_improve >= patientce:
                break
        # training done, store metrics and history
        model.load_state_dict(best_state)
        cpu_best_state = {k: v.cpu() for k,v in best_state.items()}
        
        torch.save(cpu_best_state, os.path.join(output_dir, f"model_weight_trial{trial.number}.pt"))
        torch.save(epochs_history, os.path.join(output_dir, f"train_result_trial{trial.number}.pt"))
        torch.save(validation, os.path.join(output_dir,f"val_metrics_trail{trial.number}.pt"))

        return best_val_metric
    
    finally:
        del model
        del optimizer
        del scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

def test_model(model_type:str,
               data:HeteroData,
               best_params:Dict,
               epochs:int,
               device:torch.device,
               output_dir:str,
               )->Dict:
    """Retrain model with the best hyperparameters on Train + Val data and Evaluate on Test data."""
    model = get_hetero_model(model_type=model_type,
                             data=data,
                             in_channels=best_params['in_channels'],
                             hidden_channels=best_params['hidden_channels'],
                             out_channels=best_params['out_channels'],
                             num_layers=best_params['num_layers'],
                             heads = best_params['heads'],
                             dropout_rate = best_params['dropout_rate']
                             ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    scaler = torch.cuda.amp.GradScaler()
    lambdas = LambdaScheduler(total_epochs=epochs)

    # prepare model forward input
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    initial_relevance_dict = {nt: data[nt].relevance for nt in data.node_types}
    edge_weight_dict = {}
    for edge_type in data.edge_types:
        if 'Patient' in edge_type:
            ew = data[edge_type].edge_weight
        else:
            ew = None
        edge_weight_dict[edge_type] = ew
    # initialize relevance_dict: turn relevances to learnable parmeters and Patient=0
    model.initialize_relevances(initial_relevance_dict, data)
    # retrain
    epochs_history = []
    for epoch in tqdm(range(epochs), desc="Final training with best HP"):
        current_lambdas = lambdas.get_lambdas(epoch)
        epoch_result = train_epoch(model,
                                    data,
                                    edge_index_dict,
                                    edge_weight_dict,
                                    optimizer,
                                    scaler,
                                    current_lambdas,
                                    data['Patient'].train_mask,
                                    device,
                                    negative_ratio=best_params['negative_ratio'])
        epochs_history.append(epoch_result)
    
    test_metrics = evaluate(model, data, edge_index_dict,edge_weight_dict, data['Patient'].test_maks, device)
    
    return test_metrics


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rmp_gat',
                        choices=['base_gat', 'rmp_gat'])
    parser.add_argument('--patient_graph', type=str, default="../AD/data/patient_kg.pkl")
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()
    return args

def main():
    args = parse()
    set_random_seeds(42)
    device = get_device()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    # 1. load graph, expression data and design
    patient_kg_path = args.patient_graph
    with open(patient_kg_path, 'rb') as f:
        G = pickle.load(f) # patient_kg
    
    # 2. convert netwrokx graph to HeteroData 
    data, new_node_mappings = networkx_to_hetero_data(G)
    data.to(device)
    
    # 3. HPO object
    storage = args.storage if args.storage else None
    study = optuna.create_study(
        storage=storage,
        study_name=f"{args.model} HPO",
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(
            trial=trial,
            data=data,
            model_type=args.model,
            device=device,
            epochs=args.epochs,
            patientce=50,
            output_dir=output_dir,
            num_negatives=100
        ),
        n_trials=args.trials,
        show_progress_bar=True,
    )
    # save best trial
    best_trial = study.best_trial
    with open(os.path.join(output_dir, "best_trial.pkl"), "wb") as file:
        pickle.dump(best_trial, file)
    
    # test model
    test_metrics = test_model(args.model,
                              data,
                              best_trial.params,
                              args.epochs,
                              device,
                              output_dir)

    print("\n=======================================================")
    print(f"Best trial {best_trial.number} validation f1_score: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"Test metrics: ")
    for k, v in test_metrics.items():
        print(f"    {k}: {v}")
    print("=======================================================")

if __name__ == "__main__":
    main()
    

    

    