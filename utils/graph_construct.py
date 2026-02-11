# Construct patient-KG from provided KG and patient data
import argparse
import sys
import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from typing import Any, Dict

import torch
# Add parent directory to path for imports
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
sys.path.append(os.path.dirname(base_dir))

from utils.helper import (
    extract_hgnc,
    add_patient_to_kg,
    add_attributes_to_graph,
    get_node_relevance,
)

def construct_patient_graph(kg_path:str, 
                            exp_path: str, 
                            labels_path: str, 
                            kge_path:str,
                            kge_node_mappings_path: str,
                            output:str) -> nx.MultiDiGraph:
    # load data
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)
    exp = pd.read_csv(exp_path, index_col=0)
    exp = exp.transpose()

    design = pd.read_csv(labels_path, sep='\t', index_col=0)
    design['Target'] = design['Target'].map({'Control':0, 'Disease':1})
    labels = design['Target'].to_list()

    # add patients to kg
    G, mapped_nodes = add_patient_to_kg(kg=kg, 
                          exp=exp, 
                          labels=labels
                          )
    
    # get node_relevances
    kge = torch.load(kge_path)
    with open(kge_node_mappings_path, 'rb') as f:
        node_mappings = pickle.load(f)
    
    node_relevances = get_node_relevance(kg=kg, 
                                         kg_embed=kge, 
                                         node_mappings=node_mappings, 
                                         method='cosine similarity')
    # add attributes to graph
    G = add_attributes_to_graph(G=G,
                                node_relevances=node_relevances,
                                patient_labels=labels,
                                output_filename=output)
    
    return G, mapped_nodes

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_path", type=str, default="../AD/data/ad_network_with_reverse_edges.pkl")
    parser.add_argument("--exp_path", type=str, default="../AD/data/adni_gene_cleaned.csv")
    parser.add_argument("--labels_path", type=str, default="../AD/data/adni_targets.tsv")
    parser.add_argument("--kge_path", type=str, default="../AD/data/hgt_KGembeddings.pt")
    parser.add_argument("--node_mappings", type=str, default="../AD/data/KGE_node_mappings.pkl")
    parser.add_argument("--output", type=str, default="../AD/data/patient_kg.pkl")
    args = parser.parse_args()

    construct_patient_graph(kg_path=args.kg_path,
                            exp_path=args.exp_path,
                            labels_path=args.labels_path,
                            kge_path=args.kge_path,
                            kge_node_mappings_path=args.node_mappings,
                            output=args.output)
    
if __name__=="__main__":
    main()
    