"""
Relevance fuzzy rule-enhanced HeteroGNN models modified from FireGNN framework.
Combines HeteroGNN architectures with trainable disease_relevance_fuzzy rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import HeteroConv, GATConv, HGTConv
from torch_scatter import scatter_sum

# Rule propogate at every layer

# FireGNN style models, message gate at last layer
class FuzzyRuleLayer(nn.Module):
    def __init__(self, num_rules=5):
        super().__init__()
        # alpha: sharpness of the rule, theta: threshold for "significance"
        self.theta = nn.Parameter(torch.zeros(num_rules))
        self.alpha = nn.Parameter(torch.ones(num_rules))

    def forward(self, relevance_features):
        # relevance_features shape: [num_patients, num_rules]
        # Eq (3) from paper: r = sigmoid(alpha * (f - theta))
        r = torch.sigmoid(self.alpha * (relevance_features - self.theta))
        return r

class HeteroFireGAT(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, heads, dropout_rate, num_rules=5):
        super().__init__()

        self.data = data
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # 1. Neural Component
        # Use HeteroConv to handle different edge types differently
        self.embeddings = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(num_nodes, hidden_channels)
            for node_type, num_nodes in {nt: data[nt].num_nodes for nt in data.node_types}.items()
        })
        self.conv1 = (HeteroConv({
            edge_type: GATConv((-1, -1), hidden_channels, heads, add_self_loops=False)
            for edge_type in data.edge_types
        }))
        self.conv2 = (HeteroConv({
            edge_type: GATConv(hidden_channels*heads, hidden_channels, heads=1, add_self_loops=False)
            for edge_type in data.edge_types
        }))
        
        # 2. Symbolic Component: Fuzzy Rules for Patients
        self.rule_layer = FuzzyRuleLayer(num_rules=num_rules)
        self.gate = nn.Linear(num_rules, 1)

        # 3. classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

        # 3. Link Prediction Head 
        self.rel_weights = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.ones(hidden_channels))
            for edge_type in data.edge_types
        })
       
    def forward(self,edge_index_dict, patient_rule_features):
        """_summary_

        Args:
            edge_index_dict (_type_): _description_
            patient_rule_features (_type_): _description_

        Returns:
            _type_: _description_
        """
        # --- Neural Phase ---
        # Pass messages across the whole KG
        x_dict = {
                node_type: embedding.weight
                for node_type, embedding in self.embeddings.items()
            }

        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.elu(v) for key, v in h_dict.items()}
        h_dict = {key: F.dropout(v, p=self.dropout_rate, training=self.training) for key, v in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)
        # Target the patient embeddings
        h_patient = h_dict['Patient']
        h_protein = h_dict['Protein']
        
        # --- Symbolic Phase ---
        # apply FireGNN Fuzzy Logic: r = sigmoid(alpha * (f - theta))
        # this checks if the patient's neighborhood satisfies disease logic
        r_symbolic = self.rule_layer(patient_rule_features)
        
        # gives each patient a score (learnt from relevance features) 
        # due to trained by classification, disease_patients have high score while controls have low.
        r_gate = torch.sigmoid(self.gate(r_symbolic))
        
        pat_idx, prot_idx = edge_index_dict[('Patient', 'express', 'Protein')]
        # expand r_gate to the same dimension of protein_embedding
        edge_gates = r_gate[pat_idx] 

        # here the protein_embeddings are scaled by learned rule_gate score;
        gated_messages = h_protein[prot_idx] * edge_gates
        
        # aggregate gated messages
        selective_message = scatter_sum(gated_messages, pat_idx, dim=0, dim_size=h_patient.size(0))
        
        # --- Fusion ---
        # Combine global GNN context (h_patient) and selective context
        combined = torch.cat([h_patient, selective_message], dim=1)
        out = self.classifier(combined)

        return h_dict, F.log_softmax(out, dim=1), r_symbolic
    
    def decode(self, h_dict, edge_index, edge_type):
        """Link Prediction Decoder: Predicts probability of edges"""
        u_type, rel, v_type = edge_type
        src, dst = edge_index
        
        h_src = h_dict[u_type][src]
        h_dst = h_dict[v_type][dst]
        
        rel_key = "__".join(edge_type)
        rel_w = self.rel_weights[rel_key]
        return (h_src * rel_w * h_dst).sum(dim=-1)

class HeteroGAT(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, heads, dropout_rate, num_rules=5):
        super().__init__()

        self.data = data
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # 1. Neural Component
        # Use HeteroConv to handle different edge types differently
        self.embeddings = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(num_nodes, hidden_channels)
            for node_type, num_nodes in {nt: data[nt].num_nodes for nt in data.node_types}.items()
        })
        self.conv1 = (HeteroConv({
            edge_type: GATConv((-1, -1), hidden_channels, heads, add_self_loops=False)
            for edge_type in data.edge_types
        }))
        self.conv2 = (HeteroConv({
            edge_type: GATConv(hidden_channels*heads, hidden_channels, heads=1, concat=False, add_self_loops=False)
            for edge_type in data.edge_types
        }))
        
        # 2. Link Prediction Head 
        self.rel_weights = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.ones(hidden_channels))
            for edge_type in data.edge_types
        })
        
        # 3. classifier
        self.classifier = self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self,edge_index_dict):
        # --- Neural Phase ---
        # Pass messages across the whole KG
        x_dict = {
                node_type: embedding.weight
                for node_type, embedding in self.embeddings.items()
            }

        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.elu(v) for key, v in h_dict.items()}
        h_dict = {key: F.dropout(v, p=self.dropout_rate, training=self.training) for key, v in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)
        
        # classification
        pat_embedding = h_dict['Patient']
        out = self.classifier(pat_embedding)
        
        return h_dict, F.log_softmax(out, dim=1), None
    
    def decode(self, h_dict, edge_index, edge_type):
        """Link Prediction Decoder: Predicts probability of edges"""
        u_type, rel, v_type = edge_type
        src, dst = edge_index
        
        h_src = h_dict[u_type][src]
        h_dst = h_dict[v_type][dst]
        
        rel_key = "__".join(edge_type)
        rel_w = self.rel_weights[rel_key]
        return (h_src * rel_w * h_dst).sum(dim=-1)

class HeteroFireGCN(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, heads, dropout_rate, num_rules=5):
        super().__init__()

class HeteroFireHGT(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, heads, dropout_rate, num_rules=5):
        super().__init__()

def get_hetero_model(model_type, data, hidden_channels, out_channels, **kwargs):
    """
    Factory function to create hetero rule-enhanced fuzzy models.
    
    Args:
        model_type: Type of model ('base_gat', 'rule_gat', 'hgt')
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Initialized fuzzy model
    """
    model_type = model_type.lower()
    
    if model_type == 'base_gat':
        return HeteroGAT(data, hidden_channels, out_channels, **kwargs)
    elif model_type == 'rule_gat':
        return HeteroFireGAT(data,hidden_channels, out_channels, **kwargs)
    elif model_type == 'hgt':
        return HeteroFireHGT(data, hidden_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 