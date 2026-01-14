"""
Fuzzy rule-enhanced GNN models for FireGNN framework.
Combines GNN architectures with trainable fuzzy rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import Data


class FuzzyRuleLayer(nn.Module):
    """
    Trainable fuzzy rule layer using Gaussian membership functions.
    """
    
    def __init__(self, num_features, num_rules):
        super(FuzzyRuleLayer, self).__init__()
        self.num_rules = num_rules
        self.num_features = num_features
        
        # Learnable rule centers and widths
        self.centers = nn.Parameter(torch.randn(num_rules, num_features))
        self.log_sigmas = nn.Parameter(torch.zeros(num_rules, num_features))
        
        # Learnable rule weights
        self.rule_weights = nn.Parameter(torch.ones(num_rules))
        
    def forward(self, x):
        """
        Forward pass through fuzzy rule layer.
        
        Args:
            x: Input features [N, num_features]
            
        Returns:
            torch.Tensor: Fuzzy rule activations [N, num_rules]
        """
        # Expand dimensions for broadcasting
        x_e = x.unsqueeze(1)                 # [N, 1, num_features]
        c = self.centers.unsqueeze(0)        # [1, num_rules, num_features]
        s = torch.exp(self.log_sigmas).unsqueeze(0)  # [1, num_rules, num_features]
        
        # Compute Gaussian membership functions
        gauss = torch.exp(-((x_e - c)**2) / (2*s**2))
        
        # Product of membership functions (AND operation)
        rule_activations = gauss.prod(dim=2)  # [N, num_rules]
        
        # Apply rule weights
        rule_activations = rule_activations * torch.sigmoid(self.rule_weights)
        
        return rule_activations


class FuzzyGCN(nn.Module):
    """
    GCN with fuzzy rule enhancement.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_rules=10, num_layers=2, dropout=0.5):
        super(FuzzyGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_rules = num_rules
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Fuzzy rule layer
        self.fuzzy_layer = FuzzyRuleLayer(6, num_rules)  # 6 topological features
        
        # Rule integration layers: h(u)' = W [hu, ru] + b
        self.rule_integration = nn.Linear(hidden_channels + num_rules, hidden_channels)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None, topo_features=None):
        # GCN forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_attr)
        
        # Generate fuzzy rules
        if topo_features is not None:
            fuzzy_rules = self.fuzzy_layer(topo_features)
            
            # Integrate fuzzy rules with GNN embeddings
            combined = torch.cat([x, fuzzy_rules], dim=1) # project rule to embedding dimension
            x = F.relu(self.rule_integration(combined)) # gating and fusion
        
        # Final classification
        out = self.classifier(x)
        return F.log_softmax(out, dim=1), fuzzy_rules if topo_features is not None else None


class FuzzyGAT(nn.Module):
    """
    GAT with fuzzy rule enhancement.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_rules=10, num_layers=2, heads=8, dropout=0.5, negative_slope=0.2):
        super(FuzzyGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_rules = num_rules
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                                 dropout=dropout, negative_slope=negative_slope))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=heads, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=1, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Fuzzy rule layer
        self.fuzzy_layer = FuzzyRuleLayer(6, num_rules)  # 6 topological features
        
        # Rule integration layers
        self.rule_integration = nn.Linear(hidden_channels + num_rules, hidden_channels)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None, topo_features=None):
        # GAT forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_attr)
        
        # Generate fuzzy rules
        if topo_features is not None:
            fuzzy_rules = self.fuzzy_layer(topo_features)
            
            # Integrate fuzzy rules with GNN embeddings
            combined = torch.cat([x, fuzzy_rules], dim=1)
            x = F.relu(self.rule_integration(combined))
        
        # Final classification
        out = self.classifier(x)
        return F.log_softmax(out, dim=1), fuzzy_rules if topo_features is not None else None


class MLP(nn.Module):
    """
    Multi-layer perceptron for GIN.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.lins[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
        
        x = self.lins[-1](x)
        return x


class FuzzyGIN(nn.Module):
    """
    GIN with fuzzy rule enhancement.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_rules=10, num_layers=2, dropout=0.5):
        super(FuzzyGIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_rules = num_rules
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(MLP(in_channels, hidden_channels, hidden_channels)))
        
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        
        if num_layers > 1:
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Fuzzy rule layer
        self.fuzzy_layer = FuzzyRuleLayer(6, num_rules)  # 6 topological features
        
        # Rule integration layers
        self.rule_integration = nn.Linear(hidden_channels + num_rules, hidden_channels)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None, topo_features=None):
        # GIN forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_attr)
        
        # Generate fuzzy rules
        if topo_features is not None:
            fuzzy_rules = self.fuzzy_layer(topo_features)
            
            # Integrate fuzzy rules with GNN embeddings
            combined = torch.cat([x, fuzzy_rules], dim=1)
            x = F.relu(self.rule_integration(combined))
        
        # Final classification
        out = self.classifier(x)
        return F.log_softmax(out, dim=1), fuzzy_rules if topo_features is not None else None


def get_fuzzy_model(model_type, in_channels, hidden_channels, out_channels, **kwargs):
    """
    Factory function to create fuzzy rule-enhanced models.
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'gin')
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Initialized fuzzy model
    """
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return FuzzyGCN(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_type == 'gat':
        return FuzzyGAT(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_type == 'gin':
        return FuzzyGIN(in_channels, hidden_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 