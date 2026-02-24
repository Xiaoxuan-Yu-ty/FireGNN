#!/usr/bin/env python3
"""
Main script to run the entire FireGNN pipeline.
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout[-500:])
        if e.stderr:
            print("Stderr:", e.stderr[-500:])
        return False

def main():
    parser = argparse.ArgumentParser(description='Run FireGNN pipeline')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['organcmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organsmnist'],
                       help='Dataset to use')
    parser.add_argument('--models', type=str, nargs='+', default=['gcn', 'gat', 'gin'],
                       choices=['gcn', 'gat', 'gin'],
                       help='Models to train')
    parser.add_argument('--build_graph', action='store_true',
                       help='Build graph from scratch (if not provided, use existing)')
    parser.add_argument('--train_baselines', action='store_true',
                       help='Train baseline models')
    parser.add_argument('--train_fuzzy', action='store_true',
                       help='Train fuzzy rule-enhanced models')
    parser.add_argument('--train_auxiliary', action='store_true',
                       help='Train auxiliary task models')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if graph exists
    graph_file = f"datasets/G_{args.dataset.capitalize()}_inductive.gpickle"
    if not os.path.exists(graph_file):
        print(f"Graph file {graph_file} not found!")
        if args.build_graph:
            print("Building graph from scratch...")
            cmd = f"python data_processing/build_graphs.py --dataset {args.dataset} --output_dir datasets"
            if not run_command(cmd, f"Building graph for {args.dataset}"):
                print("Failed to build graph. Exiting.")
                return
        else:
            print("Please use --build_graph to create the graph, or ensure the graph file exists.")
            return
    
    print(f"Using graph file: {graph_file}")
    
    # Train baseline models
    if args.train_baselines:
        for model in args.models:
            cmd = f"python models/train_baseline.py --model {model} --dataset {args.dataset} --graph_file {graph_file} --output_dir {args.output_dir} --epochs {args.epochs} --n_folds {args.n_folds}"
            if not run_command(cmd, f"Training {model.upper()} baseline"):
                print(f"Failed to train {model} baseline. Continuing...")
    
    # Train fuzzy models
    if args.train_fuzzy:
        for model in args.models:
            cmd = f"python fuzzy_models/train_fuzzy.py --model {model} --dataset {args.dataset} --graph_file {graph_file} --output_dir {args.output_dir} --epochs {args.epochs} --n_folds {args.n_folds}"
            if not run_command(cmd, f"Training {model.upper()} fuzzy"):
                print(f"Failed to train {model} fuzzy. Continuing...")
    
    # Train auxiliary models
    if args.train_auxiliary:
        for model in args.models:
            cmd = f"python auxiliary_models/train_auxiliary.py --model {model} --dataset {args.dataset} --graph_file {graph_file} --output_dir {args.output_dir} --epochs {args.epochs} --n_folds {args.n_folds}"
            if not run_command(cmd, f"Training {model.upper()} auxiliary"):
                print(f"Failed to train {model} auxiliary. Continuing...")
    
    print(f"\n{'='*60}")
    print("Pipeline completed!")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main() 