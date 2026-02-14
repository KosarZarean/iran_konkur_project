"""
جاسازی ویژگی‌های عددی برای TabTransformer
بر اساس مقاله: On Embeddings for Numerical Features in Tabular Deep Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================
# روش‌های جاسازی عددی
# ============================================

class PiecewiseLinearEncoding(nn.Module):
    """
    Piecewise Linear Encoding (PLE)
    تبدیل مقادیر عددی به ترکیب خطی قطعه‌ای
    """
    def __init__(self, num_features, num_bins=10, embedding_dim=32, 
                 temperature=0.1, learnable_breaks=True):
        """
        پارامترها:
        -----------
        num_features : int
            تعداد ویژگی‌های عددی
        num_bins : int
            تعداد قطعات
        embedding_dim : int
            بعد embedding
        temperature : float
            دما برای softmax
        learnable_breaks : bool
            یادگیری نقاط شکست
        """
        super(PiecewiseLinearEncoding, self).__init__()
        
        self.num_features = num_features
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # نقاط شکست (بین 0 و 1)
        if learnable_breaks:
            # نقاط شکست قابل یادگیری
            self.breakpoints = nn.Parameter(torch.linspace(0, 1, num_bins + 1))
        else:
            # نقاط شکست ثابت
            self.register_buffer('breakpoints', torch.linspace(0, 1, num_bins + 1))
        
        # وزن‌های خطی برای هر قطعه
        self.linear_weights = nn.Parameter(
            torch.randn(num_features, num_bins, embedding_dim) * 0.01
        )
        self.linear_biases = nn.Parameter(
            torch.zeros(num_features, num_bins, embedding_dim)
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
        # شکل: (batch_size, num_features, num_bins + 1)
        breakpoints = self.breakpoints.view(1, 1, -1)
        distances = torch.abs(x_norm.unsqueeze(-1) - breakpoints)
        
        # محاسبه membership در هر قطعه با softmax
        # شکل: (batch_size, num_features, num_bins)
        memberships = F.softmax(-distances[..., :-1] / self.temperature, dim=-1)
        
        # محاسبه embedding
        # memberships: (b, f, bins) @ weights: (f, bins, d) -> (b, f, d)
        embedding = torch.einsum('bfk, fkd -> bfd', memberships, self.linear_weights)
        
        return embedding


class PeriodicEncoding(nn.Module):
    """
    Periodic Activations
    استفاده از توابع دوره‌ای sin و cos
    """
    def __init__(self, num_features, embedding_dim=32, num_frequencies=8,
                 min_freq=1, max_freq=10, trainable=True):
        """
        پارامترها:
        -----------
        num_features : int
            تعداد ویژگی‌های عددی
        embedding_dim : int
            بعد embedding
        num_frequencies : int
            تعداد فرکانس‌ها
        min_freq : float
            حداقل فرکانس
        max_freq : float
            حداکثر فرکانس
        trainable : bool
            یادگیری فرکانس‌ها
        """
        super(PeriodicEncoding, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies
        
        if trainable:
            # فرکانس‌های قابل یادگیری
            self.frequencies = nn.Parameter(
                torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), num_frequencies))
            )
            self.phases = nn.Parameter(
                torch.randn(num_features, num_frequencies) * 0.1
            )
        else:
            # فرکانس‌های ثابت
            self.register_buffer(
                'frequencies',
                torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), num_frequencies))
            )
            self.register_buffer(
                'phases',
                torch.zeros(num_features, num_frequencies)
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
        x_expanded = x_norm.unsqueeze(-1)  # (b, f, 1)
        
        # محاسبه توابع sin و cos
        # freq: (num_freq) -> (1, 1, num_freq)
        freq = self.frequencies.view(1, 1, -1)
        # phase: (f, num_freq) -> (1, f, num_freq)
        phase = self.phases.unsqueeze(0)
        
        sin_comp = torch.sin(2 * np.pi * freq * x_expanded + phase)
        cos_comp = torch.cos(2 * np.pi * freq * x_expanded + phase)
        
        # ترکیب
        # (b, f, num_freq, 2) -> (b, f * num_freq * 2)
        periodic_features = torch.stack([sin_comp, cos_comp], dim=-1)
        periodic_flat = periodic_features.reshape(batch_size, -1)
        
        # projection
        embedding = self.combine(periodic_flat)
        
        return embedding.unsqueeze(1)  # (b, 1, d)


class BucketEmbedding(nn.Module):
    """
    Bucket Embedding
    تقسیم مقادیر عددی به سطل‌های مجزا
    """
    def __init__(self, num_features, num_buckets=20, embedding_dim=32,
                 strategy='linear', learnable=True):
        """
        پارامترها:
        -----------
        num_features : int
            تعداد ویژگی‌های عددی
        num_buckets : int
            تعداد سطل‌ها
        embedding_dim : int
            بعد embedding
        strategy : str
            استراتژی تقسیم: 'linear', 'quantile', 'log'
        learnable : bool
            یادگیری مرز سطل‌ها
        """
        super(BucketEmbedding, self).__init__()
        
        self.num_features = num_features
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.strategy = strategy
        
        if learnable:
            # مرزهای قابل یادگیری
            self.boundaries = nn.Parameter(torch.linspace(0, 1, num_buckets - 1))
        else:
            # مرزهای ثابت
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
        if self.strategy == 'quantile':
            # استفاده از quantile (نیاز به داده آموزش)
            # اینجا از min-max استفاده می‌کنیم
            x_min = x.min(dim=0, keepdim=True)[0]
            x_max = x.max(dim=0, keepdim=True)[0]
            x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        elif self.strategy == 'log':
            # تبدیل لگاریتمی
            x_norm = torch.log1p(x - x.min())
            x_norm = x_norm / x_norm.max()
        else:  # linear
            x_min = x.min(dim=0, keepdim=True)[0]
            x_max = x.max(dim=0, keepdim=True)[0]
            x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        # تعیین سطل
        boundaries = torch.sigmoid(self.boundaries)
        boundaries = torch.cat([
            torch.tensor([0.0]).to(x.device),
            boundaries,
            torch.tensor([1.0]).to(x.device)
        ])
        
        # پیدا کردن اندیس سطل
        # bucketize مقادیر را به بازه‌ها تقسیم می‌کند
        bucket_indices = torch.bucketize(x_norm, boundaries) - 1
        bucket_indices = torch.clamp(bucket_indices, 0, self.num_buckets - 1)
        
        # دریافت embedding
        embeddings = []
        for i in range(num_features):
            emb = self.embeddings[i](bucket_indices[:, i])
            embeddings.append(emb)
        
        return torch.stack(embeddings, dim=1)  # (b, f, d)


# ============================================
# مدل ترکیبی TabTransformer با جاسازی عددی
# ============================================

class TabTransformerWithNumEmbedding(nn.Module):
    """
    TabTransformer با قابلیت جاسازی عددی
    ترکیب مقاله‌های TabTransformer و Numerical Embeddings
    """
    
    def __init__(self, num_categorical, num_continuous, categories,
                 num_embedding_type='ple',  # 'ple', 'periodic', 'bucket', 'none'
                 embedding_dim=32, num_heads=4, num_layers=3,
                 mlp_hidden_dims=[128, 64], mlp_dropout=0.2,
                 transformer_dropout=0.1, output_dim=1):
        """
        پارامترها:
        -----------
        num_categorical : int
            تعداد ویژگی‌های دسته‌ای
        num_continuous : int
            تعداد ویژگی‌های عددی
        categories : list
            تعداد مقادیر یکتا برای هر ویژگی دسته‌ای
        num_embedding_type : str
            نوع جاسازی عددی: 'ple', 'periodic', 'bucket', 'none'
        embedding_dim : int
            بعد embedding
        num_heads : int
            تعداد headهای attention
        num_layers : int
            تعداد لایه‌های Transformer
        mlp_hidden_dims : list
            ابعاد لایه‌های پنهان MLP
        mlp_dropout : float
            نرخ Dropout در MLP
        transformer_dropout : float
            نرخ Dropout در Transformer
        output_dim : int
            بعد خروجی
        """
        super(TabTransformerWithNumEmbedding, self).__init__()
        
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        self.num_embedding_type = num_embedding_type
        
        # 1. Embedding برای ویژگی‌های دسته‌ای
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat, embedding_dim) for cat in categories
        ])
        
        # 2. جاسازی برای ویژگی‌های عددی
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
            else:  # 'none' - projection ساده
                self.num_projection = nn.Linear(num_continuous, embedding_dim)
        
        # 3. لایه‌های Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. MLP نهایی
        total_embeddings = num_categorical + num_continuous
        mlp_input_dim = embedding_dim * total_embeddings
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout)
            ])
            prev_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # مقداردهی اولیه
        self._init_weights()
    
    def _init_weights(self):
        """مقداردهی اولیه وزن‌ها"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x_cat, x_cont):
        """
        پیش‌برد داده در مدل
        
        پارامترها:
        -----------
        x_cat : torch.Tensor
            ویژگی‌های دسته‌ای
        x_cont : torch.Tensor
            ویژگی‌های عددی
        """
        batch_size = x_cat.shape[0]
        
        # 1. Embedding ویژگی‌های دسته‌ای
        cat_embedded = []
        for i in range(self.num_categorical):
            emb = self.cat_embeddings[i](x_cat[:, i])
            cat_embedded.append(emb)
        
        cat_embedded = torch.stack(cat_embedded, dim=1)  # (b, num_cat, d)
        
        # 2. پردازش ویژگی‌های عددی
        if self.num_continuous > 0:
            if hasattr(self, 'num_embedding'):
                # استفاده از جاسازی پیشرفته
                num_embedded = self.num_embedding(x_cont)
            else:
                # projection ساده
                num_embedded = self.num_projection(x_cont).unsqueeze(1)
        else:
            num_embedded = torch.empty(batch_size, 0, cat_embedded.size(2)).to(x_cat.device)
        
        # 3. ترکیب همه embeddings
        all_embeddings = torch.cat([cat_embedded, num_embedded], dim=1)
        
        # 4. عبور از Transformer
        transformed = self.transformer(all_embeddings)
        flattened = transformed.reshape(batch_size, -1)
        
        # 5. MLP نهایی
        output = self.mlp(flattened)
        
        return output.squeeze() if output.shape[1] == 1 else output


# ============================================
# توابع کمکی
# ============================================

def create_numerical_embedding_layer(num_features, method='ple', **kwargs):
    """
    ایجاد لایه جاسازی عددی
    
    پارامترها:
    -----------
    num_features : int
        تعداد ویژگی‌ها
    method : str
        روش جاسازی
    **kwargs : dict
        پارامترهای اضافی
    
    Returns:
    --------
    nn.Module
        لایه جاسازی
    """
    if method == 'ple':
        return PiecewiseLinearEncoding(num_features, **kwargs)
    elif method == 'periodic':
        return PeriodicEncoding(num_features, **kwargs)
    elif method == 'bucket':
        return BucketEmbedding(num_features, **kwargs)
    else:
        return nn.Linear(num_features, kwargs.get('embedding_dim', 32))


def test_numerical_embeddings():
    """تست روش‌های جاسازی عددی"""
    print("🧪 تست Numerical Embeddings")
    print("="*60)
    
    batch_size = 32
    num_features = 5
    x = torch.randn(batch_size, num_features)
    
    methods = ['ple', 'periodic', 'bucket']
    
    for method in methods:
        print(f"\n📌 تست {method}:")
        
        if method == 'ple':
            embed_layer = PiecewiseLinearEncoding(num_features, num_bins=10, embedding_dim=16)
        elif method == 'periodic':
            embed_layer = PeriodicEncoding(num_features, embedding_dim=16)
        else:  # bucket
            embed_layer = BucketEmbedding(num_features, num_buckets=20, embedding_dim=16)
        
        output = embed_layer(x)
        print(f"   ورودی shape: {x.shape}")
        print(f"   خروجی shape: {output.shape}")
        print(f"   تعداد پارامترها: {sum(p.numel() for p in embed_layer.parameters()):,}")
    
    print("\n✅ همه تست‌ها با موفقیت انجام شد")


if __name__ == "__main__":
    test_numerical_embeddings()
