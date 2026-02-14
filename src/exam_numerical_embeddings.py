"""
جاسازی ویژگی‌های عددی برای TabTransformer
بر اساس مقاله: On Embeddings for Numerical Features in Tabular Deep Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PiecewiseLinearEncoding(nn.Module):
    """
    Piecewise Linear Encoding (PLE)
    تبدیل مقادیر عددی به ترکیب خطی قطعه‌ای
    """
    def __init__(self, num_features, num_bins=10, embedding_dim=32, temperature=0.1):
        super(PiecewiseLinearEncoding, self).__init__()
        
        self.num_features = num_features
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # نقاط شکست قابل یادگیری
        self.breakpoints = nn.Parameter(torch.linspace(0, 1, num_bins + 1))
        
        # وزن‌های خطی برای هر قطعه
        self.linear_weights = nn.Parameter(
            torch.randn(num_features, num_bins, embedding_dim) * 0.01
        )
    
    def forward(self, x):
        """
        پارامترها:
        -----------
        x : torch.Tensor
            ویژگی‌های عددی با شکل (batch_size, num_features)
        
        Returns:
        --------
        torch.Tensor
            embedding با شکل (batch_size, num_features, embedding_dim)
        """
        batch_size, num_features = x.shape
        
        # نرمال‌سازی به بازه [0, 1]
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        # محاسبه فاصله تا هر نقطه شکست
        breakpoints = self.breakpoints.view(1, 1, -1)
        distances = torch.abs(x_norm.unsqueeze(-1) - breakpoints)
        
        # محاسبه membership در هر قطعه با softmax
        memberships = F.softmax(-distances[..., :-1] / self.temperature, dim=-1)
        
        # محاسبه embedding
        embedding = torch.einsum('bfk, fkd -> bfd', memberships, self.linear_weights)
        
        return embedding


class PeriodicEncoding(nn.Module):
    """
    Periodic Activations
    استفاده از توابع دوره‌ای sin و cos
    """
    def __init__(self, num_features, embedding_dim=32, num_frequencies=8):
        super(PeriodicEncoding, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies
        
        # فرکانس‌های قابل یادگیری
        self.frequencies = nn.Parameter(
            torch.exp(torch.linspace(math.log(1), math.log(10), num_frequencies))
        )
        
        # لایه ترکیب
        self.combine = nn.Linear(num_features * num_frequencies * 2, embedding_dim)
    
    def forward(self, x):
        """
        پارامترها:
        -----------
        x : torch.Tensor
            ویژگی‌های عددی با شکل (batch_size, num_features)
        
        Returns:
        --------
        torch.Tensor
            embedding با شکل (batch_size, 1, embedding_dim)
        """
        batch_size, num_features = x.shape
        
        # نرمال‌سازی
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        # گسترش ابعاد
        x_expanded = x_norm.unsqueeze(-1)
        
        # محاسبه توابع sin و cos
        freq = self.frequencies.view(1, 1, -1)
        
        sin_comp = torch.sin(2 * np.pi * freq * x_expanded)
        cos_comp = torch.cos(2 * np.pi * freq * x_expanded)
        
        # ترکیب
        periodic_features = torch.stack([sin_comp, cos_comp], dim=-1)
        periodic_flat = periodic_features.reshape(batch_size, -1)
        
        # projection
        embedding = self.combine(periodic_flat)
        
        return embedding.unsqueeze(1)


class BucketEmbedding(nn.Module):
    """
    Bucket Embedding
    تقسیم مقادیر عددی به سطل‌های مجزا
    """
    def __init__(self, num_features, num_buckets=20, embedding_dim=32):
        super(BucketEmbedding, self).__init__()
        
        self.num_features = num_features
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        
        # مرزهای سطل‌ها
        self.register_buffer('boundaries', torch.linspace(0, 1, num_buckets - 1))
        
        # embedding برای هر ویژگی
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_buckets, embedding_dim) for _ in range(num_features)
        ])
    
    def forward(self, x):
        """
        پارامترها:
        -----------
        x : torch.Tensor
            ویژگی‌های عددی با شکل (batch_size, num_features)
        
        Returns:
        --------
        torch.Tensor
            embedding با شکل (batch_size, num_features, embedding_dim)
        """
        batch_size, num_features = x.shape
        
        # نرمال‌سازی
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        # تعیین سطل
        boundaries = torch.cat([
            torch.tensor([0.0]).to(x.device),
            self.boundaries,
            torch.tensor([1.0]).to(x.device)
        ])
        
        bucket_indices = torch.bucketize(x_norm, boundaries) - 1
        bucket_indices = torch.clamp(bucket_indices, 0, self.num_buckets - 1)
        
        # دریافت embedding
        embeddings = []
        for i in range(num_features):
            emb = self.embeddings[i](bucket_indices[:, i])
            embeddings.append(emb)
        
        return torch.stack(embeddings, dim=1)


class TabTransformerWithNumEmbedding(nn.Module):
    """
    TabTransformer با قابلیت جاسازی عددی
    """
    def __init__(self, num_categorical, num_continuous, categories,
                 num_embedding_type='ple',
                 embedding_dim=32, num_heads=4, num_layers=3,
                 mlp_hidden_dims=[128, 64],
                 transformer_dropout=0.1, mlp_dropout=0.2,
                 output_dim=1):
        super(TabTransformerWithNumEmbedding, self).__init__()
        
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        self.num_embedding_type = num_embedding_type
        self.embedding_dim = embedding_dim
        
        # Embedding برای ویژگی‌های دسته‌ای
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat, embedding_dim) for cat in categories
        ])
        
        # جاسازی برای ویژگی‌های عددی
        if num_continuous > 0:
            if num_embedding_type == 'ple':
                self.num_embedding = PiecewiseLinearEncoding(
                    num_continuous, num_bins=10, embedding_dim=embedding_dim
                )
            elif num_embedding_type == 'periodic':
                self.num_embedding = PeriodicEncoding(
                    num_continuous, embedding_dim=embedding_dim
                )
            elif num_embedding_type == 'bucket':
                self.num_embedding = BucketEmbedding(
                    num_continuous, num_buckets=20, embedding_dim=embedding_dim
                )
            else:
                self.num_proj = nn.Linear(num_continuous, embedding_dim)
        
        # لایه‌های Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # محاسبه بعد خروجی Transformer
        # هر ویژگی دسته‌ای یک embedding به اندازه embedding_dim دارد
        # هر ویژگی عددی یک embedding به اندازه embedding_dim دارد (بعد از جاسازی)
        transformer_output_dim = embedding_dim * (num_categorical + num_continuous)
        print(f"📐 بعد خروجی Transformer: {transformer_output_dim}")
        
        # MLP نهایی
        mlp_layers = []
        prev_dim = transformer_output_dim
        
        for i, hidden_dim in enumerate(mlp_hidden_dims):
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(mlp_dropout))
            prev_dim = hidden_dim
            print(f"   لایه {i+1}: {prev_dim} -> {hidden_dim}")
        
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        print(f"   لایه خروجی: {prev_dim} -> {output_dim}")
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        print(f"✅ مدل TabTransformerWithNumEmbedding ساخته شد")
        print(f"   ویژگی‌های دسته‌ای: {num_categorical}, ویژگی‌های عددی: {num_continuous}")
        print(f"   ابعاد embedding: {embedding_dim}")
        print(f"   بعد کل: {transformer_output_dim}")
    
    def forward(self, x_cat, x_cont):
        batch_size = x_cat.shape[0]
        
        # Embedding ویژگی‌های دسته‌ای
        cat_embedded = []
        for i in range(self.num_categorical):
            emb = self.cat_embeddings[i](x_cat[:, i])
            cat_embedded.append(emb)
        
        cat_embedded = torch.stack(cat_embedded, dim=1)  # (batch, num_cat, emb_dim)
        
        # جاسازی ویژگی‌های عددی
        if self.num_continuous > 0:
            if hasattr(self, 'num_embedding'):
                # روش‌های جاسازی پیشرفته
                num_embedded = self.num_embedding(x_cont)  # (batch, num_cont, emb_dim)
            else:
                # projection ساده
                num_embedded = self.num_proj(x_cont).unsqueeze(1)  # (batch, 1, emb_dim)
                # اگر num_continuous > 1، باید تکرار شود
                if self.num_continuous > 1:
                    num_embedded = num_embedded.repeat(1, self.num_continuous, 1)
        else:
            num_embedded = torch.empty(batch_size, 0, self.embedding_dim).to(x_cat.device)
        
        # ترکیب همه embeddings
        all_embeddings = torch.cat([cat_embedded, num_embedded], dim=1)
        # all_embeddings shape: (batch, num_cat + num_cont, emb_dim)
        
        # عبور از Transformer
        transformed = self.transformer(all_embeddings)
        # transformed shape: (batch, num_cat + num_cont, emb_dim)
        
        # Flatten کردن
        flattened = transformed.reshape(batch_size, -1)
        # flattened shape: (batch, (num_cat + num_cont) * emb_dim)
        
        # MLP نهایی
        output = self.mlp(flattened)
        
        return output.squeeze() if output.shape[1] == 1 else output
