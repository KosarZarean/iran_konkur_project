"""
مدل‌های PyTorch برای داده‌های کنکور ایران
شامل: MLP، TabTransformer و کلاس‌های Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import math


class ExamDataset(Dataset):
    """
    Dataset برای مدل‌های معمولی (MLP)
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        if len(y.shape) == 1:
            self.y = torch.FloatTensor(y)
        else:
            self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabTransformerDataset(Dataset):
    """
    Dataset برای مدل TabTransformer
    """
    def __init__(self, X_cat, X_cont, y):
        self.X_cat = torch.LongTensor(X_cat)
        self.X_cont = torch.FloatTensor(X_cont)
        if len(y.shape) == 1:
            self.y = torch.FloatTensor(y)
        else:
            self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X_cat)
    
    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_cont[idx], self.y[idx]


class ExamMLP(nn.Module):
    """
    شبکه عصبی چندلایه (MLP) برای داده‌های کنکور
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1, dropout=0.2):
        super(ExamMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class TabTransformer(nn.Module):
    """
    پیاده‌سازی TabTransformer
    """
    def __init__(self, num_categorical, num_continuous, categories,
                 embedding_dim=32, num_heads=4, num_layers=3,
                 mlp_hidden_dims=[128, 64], transformer_dropout=0.1,
                 mlp_dropout=0.2, output_dim=1):
        super(TabTransformer, self).__init__()
        
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        
        # Embedding برای ویژگی‌های دسته‌ای
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat, embedding_dim) for cat in categories
        ])
        
        # لایه‌های Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=transformer_dropout,
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
        for h_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(mlp_dropout)
            ])
            prev_dim = h_dim
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # مقداردهی اولیه
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def forward(self, x_cat, x_cont):
        # Embedding ویژگی‌های دسته‌ای
        embedded = []
        for i in range(self.num_categorical):
            emb = self.embeddings[i](x_cat[:, i])
            embedded.append(emb)
        embedded = torch.stack(embedded, dim=1)
        
        # Transformer
        transformed = self.transformer(embedded)
        transformed_flat = transformed.reshape(transformed.size(0), -1)
        
        # ترکیب با ویژگی‌های عددی
        if self.num_continuous > 0:
            cont_emb = self.cont_proj(x_cont)
            combined = torch.cat([transformed_flat, cont_emb], dim=1)
        else:
            combined = transformed_flat
        
        return self.mlp(combined).squeeze()


class EarlyStopping:
    """
    Early stopping برای جلوگیری از overfitting
    """
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0
    
    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.best_epoch = epoch
        elif score - self.best_score > self.min_delta:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"      🛑 Early stopping در epoch {epoch}")
                if self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
    
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0
