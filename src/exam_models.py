"""
مدل‌های PyTorch برای داده‌های کنکور
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


class ExamMLP(nn.Module):
    """شبکه عصبی ساده"""
    def __init__(self, input_dim, hidden_dims=(128, 64), num_classes=1, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class TabTransformer(nn.Module):
    """TabTransformer"""
    def __init__(self, num_categorical, num_continuous, categories,
                 embedding_dim=32, num_heads=4, num_layers=3,
                 hidden_dims=(128, 64), dropout=0.2, output_dim=1):
        super().__init__()
        
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        
        # Embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat, embedding_dim) for cat in categories
        ])
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # پروجکشن ویژگی‌های عددی
        if num_continuous > 0:
            self.cont_proj = nn.Linear(num_continuous, embedding_dim)
        
        # MLP نهایی
        mlp_input = embedding_dim * num_categorical + (embedding_dim if num_continuous > 0 else 0)
        
        mlp_layers = []
        prev_dim = mlp_input
        for h_dim in hidden_dims:
            mlp_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h_dim
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x_cat, x_cont):
        # Embedding
        embedded = []
        for i in range(self.num_categorical):
            emb = self.embeddings[i](x_cat[:, i])
            embedded.append(emb)
        embedded = torch.stack(embedded, dim=1)
        
        # Transformer
        transformed = self.transformer(embedded)
        transformed_flat = transformed.reshape(transformed.size(0), -1)
        
        # ترکیب
        if self.num_continuous > 0:
            cont_emb = self.cont_proj(x_cont)
            combined = torch.cat([transformed_flat, cont_emb], dim=1)
        else:
            combined = transformed_flat
        
        return self.mlp(combined).squeeze()


class ExamDataset(Dataset):
    """Dataset برای مدل‌های عادی"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabTransformerDataset(Dataset):
    """Dataset برای TabTransformer"""
    def __init__(self, X_cat, X_cont, y):
        self.X_cat = torch.LongTensor(X_cat)
        self.X_cont = torch.FloatTensor(X_cont)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X_cat)
    
    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_cont[idx], self.y[idx]


class EarlyStopping:
    """Early stopping"""
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("  🛑 Early stopping")
