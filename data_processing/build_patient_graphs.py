#!/usr/bin/env python3
"""
Build graphs from Patient Expression Data and KGE for FireGNN framework.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gzip
import struct
from torchvision import transforms as T, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

print(base_dir)
sys.path.append(os.path.dirname(base_dir))
from utils.graph_utils import (
    build_knn_graph_from_features,
    prepare_pytorch_geometric_data,
    create_fuzzy_rules,
    get_device,
    save_graph
)


# split train, val, test data

# create masks
def create_train_val_test_masks(features, train_ratio=0.8, val_ratio=0.1):
    total_size = len(features)
    train_size, val_size, test_size = train_ratio*total_size, val_ratio* total_size, (1-train_ratio - val_ratio)*total_size

    train_mask = [i < train_size for i in range(total_size)]
    val_mask = [train_size <= i < train_size + val_size for i in range(total_size)]
    test_mask = [i >= train_size + val_size for i in range(total_size)]
    
    return train_mask, val_mask, test_mask

# add masks to nodes
def add_mask_to_graph(graph, output):
    train_mask, val_mask, test_mask = create_train_val_test_masks(features)

    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['train'] = train_mask[i]
        graph.nodes[node]['val'] = val_mask[i]
        graph.nodes[node]['test'] = test_mask[i]
    # save graph
    save_graph(graph, output)

    return graph

def build_and_save_patient_graph():
    pass