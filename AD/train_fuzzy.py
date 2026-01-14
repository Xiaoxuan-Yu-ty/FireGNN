#!/usr/bin/env python3
"""
Training script for fuzzy rule-enhanced FireGNN models.
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import sys

# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import (
    load_graph,
    prepare_pytorch_geometric_data,
    get_device,
    set_random_seeds
)

from fuzzy_models.fuzzy_models import get_fuzzy_model

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out, fuzzy_rules = model(
        data.x,
        data.edge_index,
        edge_attr=data.edge_attr,
        topo_features=data.topo_features
    )
    preds = out.argmax(dim=1)
    correct = (preds[mask] == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    loss = F.nll_loss(out[mask], data.y[mask]).item()
    return acc, loss, preds, fuzzy_rules


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train(model, data, optimizer, epochs, device):
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": []
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in trange(epochs, desc="Training"):
        model.train()
        optimizer.zero_grad()

        out, _ = model(
            data.x,
            data.edge_index,
            edge_attr=data.edge_attr,
            topo_features=data.topo_features
        )

        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # ---- Evaluation ----
        train_acc, train_loss, _, _ = evaluate(model, data, data.train_mask)
        val_acc, val_loss, _, _ = evaluate(model, data, data.val_mask)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_state, history


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['gcn', 'gat', 'gin'])
    parser.add_argument('--graph_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=200)
    
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_rules', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_random_seeds(args.seed)
    device = get_device()

    # -----------------------------------------------------
    # Load graph & data
    # -----------------------------------------------------
    G = load_graph(args.graph_file)
    data = prepare_pytorch_geometric_data(G)
    data = data.to(device)

    in_channels = data.x.size(1)
    out_channels = int(data.y.max().item() + 1)

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = get_fuzzy_model(
        model_type=args.model,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=out_channels,
        num_rules=args.num_rules
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    best_state, history = train(
        model=model,
        data=data,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device
    )

    # Load best model
    model.load_state_dict(best_state)

    # -----------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------
    test_acc, test_loss, test_preds, fuzzy_rules = evaluate(
        model, data, data.test_mask
    )

    print(f"\nTest Accuracy: {test_acc:.4f}")

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    save_dir = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.model}_fuzzy"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Model
    torch.save(best_state, os.path.join(save_dir, "model.pt"))

    # Predictions
    torch.save({
        "preds": test_preds.cpu(),
        "labels": data.y.cpu(),
        "test_mask": data.test_mask.cpu()
    }, os.path.join(save_dir, "predictions.pt"))

    # Fuzzy rule activations
    if fuzzy_rules is not None:
        torch.save(
            fuzzy_rules.cpu(),
            os.path.join(save_dir, "fuzzy_rules.pt")
        )

    # Learned fuzzy parameters
    fuzzy_params = {
        "centers": getattr(model.fuzzy_layer, "centers", None),
        "sigmas": getattr(model.fuzzy_layer, "log_sigmas", None),
        "rule_weights": getattr(model.fuzzy_layer, "rule_weights", None)
    }
    torch.save(fuzzy_params, os.path.join(save_dir, "fuzzy_params.pt"))

    # Metrics
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({
            "test_acc": test_acc,
            "test_loss": test_loss,
            "history": history
        }, f, indent=4)

    print(f"Saved results to: {save_dir}")


if __name__ == "__main__":
    main()
