"""
جاسازی ویژگی‌های عددی
"""

import torch
import torch.nn as nn
import numpy as np


class PiecewiseLinearEncoding(nn.Module):
    """Piecewise Linear Encoding"""
    def __init__(self, num_features, num_bins=10, embedding_dim=32):
        super().__init__()
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        
        self.breakpoints = nn.Parameter(torch.linspace(0, 1, num_bins + 1))
        self.weights = nn.Parameter(torch.randn(num_features, num_bins, embedding_dim))
    
    def forward(self, x):
        batch_size, num_features = x.shape
        
        # نرمال‌سازی
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        # محاسبه ضرایب
        coeffs = []
        for i in range(self.num_bins):
            left = self.breakpoints[i]
            right = self.breakpoints[i + 1]
            mask = torch.sigmoid((x_norm - left) * 10) * torch.sigmoid((right - x_norm) * 10)
            coeffs.append(mask.unsqueeze(-1))
        
        coeffs = torch.stack(coeffs, dim=2)
        
        # Embedding
        embedding = torch.einsum('bfk, fke -> bfe', coeffs, self.weights)
        return embedding


class PeriodicEncoding(nn.Module):
    """Periodic Activations"""
    def __init__(self, num_features, embedding_dim=32, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.frequencies = nn.Parameter(torch.randn(num_features, num_frequencies))
        self.phases = nn.Parameter(torch.randn(num_features, num_frequencies))
        self.proj = nn.Linear(num_features * num_frequencies * 2, embedding_dim)
    
    def forward(self, x):
        batch_size, num_features = x.shape
        
        x_exp = x.unsqueeze(-1)
        freq = self.frequencies.unsqueeze(0)
        phase = self.phases.unsqueeze(0)
        
        sin_comp = torch.sin(2 * np.pi * freq * x_exp + phase)
        cos_comp = torch.cos(2 * np.pi * freq * x_exp + phase)
        
        features = torch.cat([sin_comp, cos_comp], dim=-1)
        features = features.reshape(batch_size, -1)
        
        return self.proj(features).unsqueeze(1)


class BucketEmbedding(nn.Module):
    """Bucket Embedding"""
    def __init__(self, num_features, num_buckets=20, embedding_dim=32):
        super().__init__()
        self.num_buckets = num_buckets
        
        self.boundaries = nn.Parameter(torch.linspace(0, 1, num_buckets - 1))
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_buckets, embedding_dim) for _ in range(num_features)
        ])
    
    def forward(self, x):
        batch_size, num_features = x.shape
        
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        bounds = torch.sigmoid(self.boundaries)
        bounds = torch.cat([torch.tensor([0.0]).to(x.device), bounds, torch.tensor([1.0]).to(x.device)])
        
        indices = torch.bucketize(x_norm, bounds) - 1
        indices = torch.clamp(indices, 0, self.num_buckets - 1)
        
        embeddings = []
        for i in range(num_features):
            emb = self.embeddings[i](indices[:, i])
            embeddings.append(emb)
        
        return torch.stack(embeddings, dim=1)


class TabTransformerWithNumEmbedding(nn.Module):
    """TabTransformer با جاسازی عددی"""
    def __init__(self, num_categorical, num_continuous, categories,
                 num_embedding_type='ple', embedding_dim=32, num_heads=4,
                 num_layers=3, hidden_dims=(128, 64), dropout=0.2, output_dim=1):
        super().__init__()
        
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        self.num_embedding_type = num_embedding_type
        
        # Embedding دسته‌ای
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat, embedding_dim) for cat in categories
        ])
        
        # جاسازی عددی
        if num_continuous > 0:
            if num_embedding_type == 'ple':
                self.num_embedding = PiecewiseLinearEncoding(num_continuous, embedding_dim=embedding_dim)
            elif num_embedding_type == 'periodic':
                self.num_embedding = PeriodicEncoding(num_continuous, embedding_dim=embedding_dim)
            elif num_embedding_type == 'bucket':
                self.num_embedding = BucketEmbedding(num_continuous, embedding_dim=embedding_dim)
            else:
                self.num_proj = nn.Linear(num_continuous, embedding_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=embedding_dim*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # MLP
        mlp_input = embedding_dim * num_categorical + (embedding_dim if num_continuous > 0 else 0)
        
        mlp = []
        prev = mlp_input
        for h in hidden_dims:
            mlp.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        mlp.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*mlp)
    
    def forward(self, x_cat, x_cont):
        # Embedding دسته‌ای
        cat_emb = []
        for i in range(self.num_categorical):
            cat_emb.append(self.cat_embeddings[i](x_cat[:, i]))
        cat_emb = torch.stack(cat_emb, dim=1)
        
        # جاسازی عددی
        if self.num_continuous > 0:
            if hasattr(self, 'num_embedding'):
                num_emb = self.num_embedding(x_cont)
            else:
                num_emb = self.num_proj(x_cont).unsqueeze(1)
        else:
            num_emb = torch.empty(x_cat.size(0), 0, cat_emb.size(2)).to(x_cat.device)
        
        # ترکیب
        all_emb = torch.cat([cat_emb, num_emb], dim=1)
        
        # Transformer
        transformed = self.transformer(all_emb)
        flat = transformed.reshape(transformed.size(0), -1)
        
        return self.mlp(flat).squeeze()
