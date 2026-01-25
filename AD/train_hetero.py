"""
Training functions for Rule-enhanced HeteroFireGNN models.
"""
import argparse
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

from tqdm import trange
# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from helper import (
    networkx_to_hetero_data,
    compute_rule_features,
    get_device,
    set_random_seeds
)
from hetero_models import get_hetero_model

# Training Step
# --------------------------------------------------------------------
def train_step(model, data, rule_features, optimizer, lambdas):
    model.train()
    optimizer.zero_grad()
    #x_dict = {nt:data[nt].x for nt in data.node_types}
    edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
    
    # 1. forward Pass
    h_dict, class_out, r = model(edge_index_dict, rule_features)
    
    # 2. classification Loss 
    y = torch.as_tensor(data['Patient'].y).long().squeeze()
    mask = data['Patient'].train_mask
    loss_cls = F.nll_loss(class_out[mask], y[mask])
    
    # 3. link prediction loss
    loss_lp = 0
    for edge_type in edge_index_dict.keys():
        pos_edge_index = edge_index_dict[edge_type]
        
        # Positive Scores
        pos_scores = model.decode(h_dict, pos_edge_index, edge_type)
        
        # Negative Sampling: Randomly shuffle destination nodes
        neg_edge_index = pos_edge_index.clone()
        neg_edge_index[1] = neg_edge_index[1][torch.randperm(neg_edge_index.size(1))]
        neg_scores = model.decode(h_dict, neg_edge_index, edge_type)
        
        # Ranking Loss: Encourage pos_scores > neg_scores
        loss_lp += -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()
    
    loss_lp = torch.tensor(loss_lp / len(data.edge_types)) # to prevent link prediction loss becomes too huge

    # 4. regularizer to ensure rule activations to be 'decisive' (near 0 or 1, not 0.5)
    loss_reg = torch.mean(r * (1 - r)) 
    loss_reg = torch.mean(r * (1 - r)) if r is not None else torch.tensor(0.0).to(loss_lp.device)

    # 5. Combined Total Loss
    total_loss = (lambdas['cls'] * loss_cls + 
                  lambdas['lp'] * loss_lp + 
                  lambdas['reg'] * loss_reg)
    print(total_loss.grad_fn)
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_cls.item(), loss_lp.item(), loss_reg.item()

# lambda Scheduler
class LambdaScheduler:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def get_lambdas(self, epoch):
        # Phase 1: Warming up the KG (Epoch 0-20)
        if epoch < 20:
            return {'cls': 0.1, 'lp': 1.0, 'reg': 0.0}
        
        # Phase 2: Shift focus to Classification (Epoch 21-100)
        elif epoch < 100:
            # Linear ramp for classification: from 0.1 to 1.0
            cls_val = 0.1 + (0.9 * (epoch - 20) / 80)
            return {'cls': cls_val, 'lp': 0.1, 'reg': 0.005}
        
        # Phase 3: Sharpen the Symbolic Rules (Epoch 100+)
        else:
            return {'cls': 1.0, 'lp': 0.05, 'reg': 0.02}

# Evaluation (on classification + link prediction)
# ----------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, rule_features, mask):
    model.eval()

    # 1. classfication performance
    edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
    h_dict, class_out, r = model(edge_index_dict, rule_features)
    
    y = torch.as_tensor(data['Patient'].y).long().squeeze()
    preds = class_out.argmax(dim=1)
    correct = (preds[mask] == y[mask]).sum().item()
    acc = correct / mask.sum().item()

    # 2. link prediction performance
    sample_edge_type = list(edge_index_dict.keys())[0]
    lp_hits = evaluate_link_prediction(model, data, h_dict, sample_edge_type, k=10)

    return acc, lp_hits, r.detach() if r is not None else None

@torch.no_grad()
def evaluate_link_prediction(model, data, h_dict, edge_type, k=10):
    model.eval()
    u_type, rel, v_type = edge_type
    edge_index = data[edge_type].edge_index
    
    # sample a subset for speed 
    num_edges = edge_index.size(1)
    indices = torch.randperm(num_edges)[:500] 
    edge_index = edge_index[:, indices]
    
    # 1. positive scores
    pos_scores = model.decode(h_dict, edge_index, edge_type)
    
    # 2. negative scores (ranking against random nodes)
    # for each positive edge, compare it against many random nodes
    num_neg = 100
    hits = 0
    for i in range(edge_index.size(1)):
        src_node = edge_index[0, i]
        true_dst = edge_index[1, i]
        
        # sample 100 random negative destination nodes
        neg_dst = torch.randint(0, data[v_type].num_nodes, (num_neg,), device=edge_index.device)
        
        # build neg_edge_index for this specific source
        neg_edges = torch.stack([torch.full((num_neg,), src_node, device=edge_index.device), neg_dst])
        
        neg_scores = model.decode(h_dict, neg_edges, edge_type)
        
        # check if positive score is in the top K among (1 positive + 100 negatives)
        combined_scores = torch.cat([pos_scores[i].unsqueeze(0), neg_scores])
        _, top_indices = torch.topk(combined_scores, k=k)
        
        if 0 in top_indices: # 0 is the index of the positive score
            hits += 1
            
    return hits / edge_index.size(1)

# Training Loop
# ----------------------------------------------------------
def train(model, data, optimizer, lambdas, epochs, rule_features, device):
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_hits@k": [],
        "val_hits@k": [],
        "train_cls_loss": [],
        "train_lp_loss": [],
        "rule_loss": [],
        'train_rule':[],
        'val_rule':[]
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in trange(epochs, desc="Training"):
        
        # train step
        current_lambdas = lambdas.get_lambdas(epoch)
        total_loss, cls_loss, lp_loss, r_loss = train_step(model, data, rule_features, optimizer, current_lambdas)

        # ---- Evaluation ----
        train_acc, train_hits, train_r = evaluate(model, data, rule_features,data['Patient'].train_mask)
        val_acc,  val_hits, val_r = evaluate(model, data, rule_features,data['Patient'].val_mask)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history['train_hits@k'].append(train_hits)
        history['val_hits@k'].append(val_hits)
        history["train_cls_loss"].append(cls_loss)
        history["train_lp_loss"].append(lp_loss)
        history['rule_loss'].append(r_loss)
        history["train_rule"].append(train_r)
        history["val_rule"].append(val_r)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_state, history

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['base_gat', 'rule_gat', 'hgt'])
    parser.add_argument('--patient_graph', type=str, default="./data/KG/patient_kg.pkl")
    
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=200)
    

    args = parser.parse_args()
    return args

def main():
    args = parse()
    set_random_seeds(42)
    device = get_device()

    patient_kg_path = args.patient_graph

    # 1. load graph, expression data and design
    with open(patient_kg_path, 'rb') as f:
        G = pickle.load(f) # patient_kg
    
    # 2. convert netwrokx graph to HeteroData 
    data, new_node_mappings = networkx_to_hetero_data(G)
    data.to(device)
    # 3. compute relevance_rule_features
    relevance_features = compute_rule_features(data)

    # 4. define model
    model = get_hetero_model(
        model_type=args.model,
        data=data,
        hidden_channels=128,
        out_channels=2,
        heads=2,
        dropout_rate=0.5,
        num_rules=2
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # Usage in your training loop:
    scheduler = LambdaScheduler(total_epochs=args.epochs)

    # train_loop
    best_state, history = train(
                                model,
                                data,
                                optimizer,
                                scheduler,
                                args.epochs,
                                relevance_features,
                                device)
    
    # save best_state and history metrics to output_dir/model
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    model_state_path = os.path.join(output_dir, 'best_state.pt')
    history_path = os.path.join(output_dir, 'history.pt')
    torch.save(best_state, model_state_path)
    torch.save(history, history_path)
    print('Training history')
    for k,v in history.items():
        print(f"{k} : {v}")
        print(f"{k} : Min = {min(v)}, Max = {max(v)}\n")

if __name__=="__main__":
     main()