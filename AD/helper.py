"""
Utility functions for HeteroFireGNN framework.
Provide functions for graph generation, HeteroData conversion,
topological features computtaion and conditioned rules computation.
"""
import pickle
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
import scipy.stats
from tqdm import tqdm
import pandas as pd
import re
from typing import Any, Dict, Tuple
from torch_geometric.data import HeteroData

def extract_hgnc(node_id):
    match = re.search(r'HGNC:"([^"]+)"\)', node_id)
    return match.group(1) if match else ""

def add_patient_to_kg(kg:nx.MultiDiGraph, exp:pd.DataFrame, labels: list, output_filename:str):
    """Add patients to KG with gene-expression-value as edge_weight, 
    create edges between patients and all overlapping HGNC-proteins .

    Args:
        kg (nx.MultiDiGraph): knowledge graph
        exp (pd.DataFrame): patient gene expression data (num_patients x num_genes)
        labels (list): target labels for the patients.
        output_filename(str): save graph

    Returns:
        nx.MultiDiGraph: A new Knowledge Graph with patient nodes
    """
    exp_genes = list(exp.columns)
    print('The number of genes in Gene Expression data:',len(exp_genes))
    patients = list(exp.index)
    print('The number of patients:', len(patients))

    # add patients to kg
    G = kg.copy()
    kg_proteins = [node for node in G.nodes(data=True) if node[1]['label'] == 'Protein']
    print('The number of KG proteins: ',len(kg_proteins))

    mapped_nodes = set()
    for i in tqdm(range(len(patients)), desc='Add patients to KG'):
        head = patients[i]
        target = labels[i]
        if head not in G.nodes:
            G.add_node(head, label='Patient', y=target)

        for node,attr in kg_proteins:
            
            gene_name = extract_hgnc(node) # only link to HGNC proteins

            if gene_name in exp_genes:
                j = exp_genes.index(gene_name)
                #print(head)
                #print(gene_name)
                exp_value = exp.iloc[i,j]
                #print(exp_value)
                G.add_edge(head, node, 'express', edge_weight = exp_value)
                G.add_edge(node, head, 'rev_express', edge_weight = exp_value)
                mapped_nodes.add(node)
    print(f'The number of mapped protein-nodes is {len(mapped_nodes)}')
    
    # save graph
    with open(output_filename, 'wb') as f:
        pickle.dump(G, f)
    print('-------------------------- Done ------------------------------')
    return G, mapped_nodes

