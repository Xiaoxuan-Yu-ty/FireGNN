#!/usr/bin/env python3
"""
Pipeline to run FireGNN & ML models on different dataset.
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
    parser = argparse.ArgumentParser(description='Run Pilot Study pipeline')
    parser.add_argument('--dataset', type=str, default="Bloodmnist",
                       choices=['Composite', 'KGRules_Composite', 'KGRules_Expression','NormExpression','RawExpression','Bloodmnist'],
                       help='Dataset to use')
    parser.add_argument('--k', type=int, default=20, help='Number of clusters in K-NN')
    parser.add_argument('--models', type=str, nargs='+', default=['gcn', 'gat', 'gin'],
                       choices=['gcn', 'gat', 'gin'],
                       help='Models to train')
    #parser.add_argument('--build_graph', action='store_true',
    #                    help='Build graph from scratch (if not provided, use existing)')
    parser.add_argument('--train_baselines', action='store_true',
                       help='Train baseline models')
    parser.add_argument('--train_fuzzy', action='store_true',
                       help='Train fuzzy models')
    parser.add_argument("--train_ml", 
                        action='store_true', 
                        help='Train classic machine learning models.')
    # parser.add_argument('--train_auxiliary', action='store_true',
    #                    help='Train auxiliary task models')
    parser.add_argument('--output_dir', type=str, default='results/three_classes/no_label_leakage',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of trials for HPO')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # run the pipline for each k-graph
    for i in range(2,args.k):
        # Build graph from scratch is graph file is not provided
        graph_file = f"datasets/three_classes/no_label_leakage/G_{args.dataset}_k{i}.pkl"
            
        print(f"Using graph file: {graph_file}")

        # Train baseline models
        if args.train_baselines:
            for model in args.models:
                cmd = f"python train/train_baseline.py --model {model} --dataset {args.dataset} --graph_file {graph_file} --k {i} --output_dir {args.output_dir} --epochs {args.epochs}"
                if not run_command(cmd, f"Training {model.upper()} baseline"):
                    print(f"Failed to train {model} baseline. Continuing...")
        
        # Train fuzzy models
        if args.train_fuzzy:
            for model in ['gcn', 'gat', 'gin','paper_gcn', 'fuzzy_only']:
                cmd = f"python train/train_fuzzy.py --model {model} --dataset {args.dataset} --graph_file {graph_file} --k {i} --output_dir {args.output_dir} --epochs {args.epochs}"
                if not run_command(cmd, f"Training {model.upper()} fuzzy"):
                    print(f"Failed to train {model} fuzzy. Continuing...")
        
        # Train ML models
        if args.train_ml:
            cmd = f"python train/train_ml.py --dataset {args.dataset}  --output_dir {args.output_dir} --n_trials {args.n_trials}"
            if not run_command(cmd, f"Training ML models"):
                print(f"Failed to train ML model. Continuing...")
    
    print(f"\n{'='*60}")
    print("Pipeline completed!")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main() 