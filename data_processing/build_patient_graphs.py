#!/usr/bin/env python3
"""
Build graphs from Patient Expression Data and KGE for FireGNN framework.
"""

import argparse
from collections import defaultdict
import json
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import gzip
import struct
from torchvision import transforms as T, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import networkx as nx

# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

print(base_dir)
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import (
    load_graph,
    build_knn_graph_from_features,
    prepare_pytorch_geometric_data,
    create_fuzzy_rules,
    get_device,
    save_graph
)


# split train, val, test data
# # create masks
def create_train_val_test_masks(features, train_ratio=0.8, val_ratio=0.1):
    total_size = len(features)
    train_size, val_size, test_size = train_ratio*total_size, val_ratio* total_size, (1-train_ratio - val_ratio)*total_size

    train_mask = [i < train_size for i in range(total_size)]
    val_mask = [train_size <= i < train_size + val_size for i in range(total_size)]
    test_mask = [i >= train_size + val_size for i in range(total_size)]
    
    return train_mask, val_mask, test_mask

# add masks to nodes
def add_mask_to_graph(graph, features, output):
    train_mask, val_mask, test_mask = create_train_val_test_masks(features)

    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['train'] = train_mask[i]
        graph.nodes[node]['val'] = val_mask[i]
        graph.nodes[node]['test'] = test_mask[i]
    # save graph
    save_graph(graph, output)

    return graph

def build_and_save_patient_graph(composite_embed_path, expression_path, design_path, k:int, output_dir:str):
    
    # get labels: 3 classes
    design = pd.read_csv(design_path, sep='\t', index_col=0)
    design['Old_Target'] = design['Old_Target'].map({'Control': 0, 'Disease': 1})
    labels = design['Old_Target'].to_numpy()

    # composite features
    embed_data = torch.load(composite_embed_path)
    composite_features = torch.stack([embed_data[idx] for idx in embed_data.keys()]).cpu().tolist()
    
    # expression features
    exp_data = pd.read_csv(expression_path, index_col=0)
    exp_data = exp_data.T
    # raw expression features
    exp_raw_features = exp_data.to_numpy()
    # normalized expression features
    exp_norm = (exp_data - exp_data.min())/(exp_data.max()-exp_data.min())
    exp_norm_features = exp_norm.to_numpy()

    # build graph for 3 types of data
    graph_info = {}
    for i in range(2,k):
        graph_info[i]=defaultdict(dict)
        graph_composite = build_knn_graph_from_features(features=composite_features,
                                                        labels=labels,
                                                        k=i,
                                                        add_label_edges=True,
                                                        rewire_edges=True,
                                                        )
        print("The Number of Connected Components:", nx.number_connected_components(graph_composite))
        graph_info[i]['Composite'] = [nx.number_connected_components(graph_composite),
                                        nx.number_of_nodes(graph_composite),
                                        nx.number_of_edges(graph_composite)]
        
        graph_normexp = build_knn_graph_from_features(features=exp_norm_features,
                                                        labels=labels,
                                                        k=i,
                                                        add_label_edges=True,
                                                        rewire_edges=True,
                                                        )
        print("The Number of Connected Components:", nx.number_connected_components(graph_normexp))
        graph_info[i]['NormExpression'] = [nx.number_connected_components(graph_normexp),
                                            nx.number_of_nodes(graph_normexp),
                                            nx.number_of_edges(graph_normexp)]
        
        graph_rawexp = build_knn_graph_from_features(features=exp_raw_features,
                                                        labels=labels,
                                                        k=i,
                                                        add_label_edges=True,
                                                        rewire_edges=True,
                                                        )
        print("The Number of Connected Components:", nx.number_connected_components(graph_rawexp))
        graph_info[i]['RawExpression'] = [nx.number_connected_components(graph_rawexp),
                                            nx.number_of_nodes(graph_rawexp),
                                            nx.number_of_edges(graph_rawexp)]
        
        # add masks and save graph
        os.makedirs(output_dir, exist_ok=True)
        graph_composite = add_mask_to_graph(graph_composite, composite_features,os.path.join(output_dir, f"G_Composite_k{i}.pkl"))
        graph_normexp = add_mask_to_graph(graph_normexp, exp_norm_features,os.path.join(output_dir, f"G_NormExpression_k{i}.pkl"))
        graph_rawexp = add_mask_to_graph(graph_rawexp, exp_raw_features,os.path.join(output_dir, f"G_RawExpression_k{i}.pkl"))
    # save info
    with open(os.path.join(output_dir, 'gragh_patient_metrics.json'),'w') as f:
        json.dump(graph_info, f, indent=4)
    return graph_info

def rebuild_morpho_graphs(graph_path:str, dataset:str, k:int, output_dir:str):
    
    G = load_graph(graph_path)
    
    features = []
    labels = []
    train_mask = []
    val_mask = []
    test_mask = []
    for node_id, attrs in G.nodes(data=True):
        x = attrs['x']
        features.append(x)
        y = attrs['y']
        labels.append(y)
        train = attrs['train']
        train_mask.append(train)
        val = attrs['val']
        val_mask.append(val)
        test = attrs['test']
        test_mask.append(test)
    
    graph_info = {}
    for i in range(1,k):
        graph_info[i] = defaultdict(dict)
        graph = build_knn_graph_from_features(features=features, 
                                                            labels=labels, 
                                                            k=i,
                                                            add_label_edges=False,
                                                            rewire_edges=False,)
        print("The Number of Connected Components:", nx.number_connected_components(graph))
        graph_info[i][dataset] = [nx.number_connected_components(graph),
                                            nx.number_of_nodes(graph),
                                            nx.number_of_edges(graph)]
        # add masks
        for j, node in enumerate(graph.nodes()):
            graph.nodes[node]['train'] = train_mask[j]
            graph.nodes[node]['val'] = val_mask[j]
            graph.nodes[node]['test'] = test_mask[j]
        # save graph
        save_graph(graph, os.path.join(output_dir, f"G_{dataset}_k{i}.pkl"))
    
    filename = os.path.join(output_dir, 'graph_metrics.json')
    with open(filename, 'w') as f:
        json.dump(graph_info, f, indent=4)
    return

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_type", type=str, required=True, choices=['patient', 'morpho'], help='Which type of graph to build.')
    parser.add_argument("--dataset", type=str, default='Bloodmnist', choices=['Bloodmnist','Organcmnist'], help='Dataset to rebuild graph with different k')
    parser.add_argument("--morpho_path", type=str, default="../datasets/G_Bloodmnist_inductive.gpickle")
    parser.add_argument("--exp_path", type=str, default="../AD/data/adni_gene_cleaned.csv")
    parser.add_argument("--labels_path", type=str, default="../AD/data/design_with_real_target.tsv")
    parser.add_argument("--kge_path", type=str, default="../AD/data/composite_embed.pt")
    parser.add_argument("--k", type=int, default=20, help="Number of k graphs to build with k in K-NN clustering")
    parser.add_argument("--output_dir", type=str, default="../datasets/two_classes/label_leakage")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.graph_type == 'patient':
        build_and_save_patient_graph(composite_embed_path=args.kge_path,
                                expression_path=args.exp_path,
                                design_path=args.labels_path,
                                k=args.k,
                                output_dir=args.output_dir)
    elif args.graph_type == 'morpho':
        rebuild_morpho_graphs(graph_path=args.morpho_path,
                                k=args.k, 
                                dataset=args.dataset,
                                output_dir=args.output_dir)
    else:
        print("Invalid graph_type, please choose in ['patient', 'morpho']")
    
    
if __name__=="__main__":
    main()
    