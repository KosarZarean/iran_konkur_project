"""
مدل‌های PyTorch برای داده‌های کنکور ایران
شامل: MLP، TabTransformer، Regressor و کلاس‌های Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import math


# ============================================
# کلاس‌های Dataset
# ============================================

class ExamDataset(Dataset):
    """
    Dataset برای مدل‌های معمولی (MLP, Regressor)
    """
    def __init__(self, X, y):
        """
        پارامترها:
        -----------
        X : array-like
            ویژگی‌ها
        y : array-like
            برچسب‌ها
        """
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
        """
        پارامترها:
        -----------
        X_cat : array-like
            ویژگی‌های دسته‌ای
        X_cont : array-like
            ویژگی‌های عددی
        y : array-like
            برچسب‌ها
        """
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


# ============================================
# مدل‌های پایه
# ============================================

class ExamMLP(nn.Module):
    """
    شبکه عصبی چندلایه (MLP) برای داده‌های کنکور
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1, 
                 dropout=0.2, activation='relu', use_batch_norm=True):
        """
        پارامترها:
        -----------
        input_dim : int
            بعد ورودی
        hidden_dims : list
            ابعاد لایه‌های پنهان
        output_dim : int
            بعد خروجی (1 برای رگرسیون، >1 برای طبقه‌بندی)
        dropout : float
            نرخ Dropout
        activation : str
            تابع فعال‌ساز ('relu', 'leaky_relu', 'elu', 'tanh')
        use_batch_norm : bool
            استفاده از Batch Normalization
        """
        super(ExamMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        
        # انتخاب تابع فعال‌ساز
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # ساخت لایه‌ها
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # لایه خطی
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # تابع فعال‌ساز
            layers.append(self.activation)
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # لایه خروجی
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # مقداردهی اولیه وزن‌ها
        self._init_weights()
    
    def _init_weights(self):
        """مقداردهی اولیه وزن‌ها با روش Kaiming"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """پیش‌برد داده در شبکه"""
        return self.model(x)
    
    def get_feature_importance(self, feature_names=None):
        """
        محاسبه اهمیت ویژگی‌ها بر اساس وزن‌های لایه اول
        
        پارامترها:
        -----------
        feature_names : list
            نام ویژگی‌ها
        
        Returns:
        --------
        dict
            اهمیت هر ویژگی
        """
        # وزن‌های لایه اول
        first_layer = self.model[0]
        weights = first_layer.weight.data.cpu().numpy()
        
        # محاسبه اهمیت (میانگین قدر مطلق وزن‌ها)
        importance = np.mean(np.abs(weights), axis=0)
        
        if feature_names is not None:
            if len(feature_names) != len(importance):
                print(f"⚠️ تعداد نام ویژگی‌ها ({len(feature_names)}) با تعداد ویژگی‌ها ({len(importance)}) مطابقت ندارد")
                return dict(enumerate(importance))
            
            # مرتب‌سازی بر اساس اهمیت
            sorted_idx = np.argsort(importance)[::-1]
            return {feature_names[i]: importance[i] for i in sorted_idx}
        else:
            return dict(enumerate(importance))


class ExamRegressor(nn.Module):
    """
    مدل رگرسیون ساده برای پیش‌بینی رتبه
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super(ExamRegressor, self).__init__()
        
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
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


# ============================================
# مدل TabTransformer
# ============================================

class TransformerBlock(nn.Module):
    """
    یک بلوک Transformer شامل Self-Attention و Feed-Forward
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Self-Attention with residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-Forward with residual connection
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class TabTransformer(nn.Module):
    """
    پیاده‌سازی کامل TabTransformer
    بر اساس مقاله: TabTransformer: Tabular Data Modeling Using Contextual Embeddings
    """
    def __init__(self, num_categorical, num_continuous, categories,
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
            لیست تعداد مقادیر یکتا برای هر ویژگی دسته‌ای
        embedding_dim : int
            بعد embedding
        num_heads : int
            تعداد headهای attention
        num_layers : int
            تعداد لایه‌های Transformer
        mlp_hidden_dims : list
            ابعاد لایه‌های پنهان MLP نهایی
        mlp_dropout : float
            نرخ Dropout در MLP
        transformer_dropout : float
            نرخ Dropout در Transformer
        output_dim : int
            بعد خروجی (1 برای رگرسیون)
        """
        super(TabTransformer, self).__init__()
        
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        self.categories = categories
        self.embedding_dim = embedding_dim
        
        # 1. لایه‌های Embedding برای ویژگی‌های دسته‌ای
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat, embedding_dim) for cat in categories
        ])
        
        # 2. لایه‌های Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, 
                           dim_feedforward=embedding_dim*4, 
                           dropout=transformer_dropout)
            for _ in range(num_layers)
        ])
        
        # 3. لایه Projection برای ویژگی‌های عددی
        if num_continuous > 0:
            self.cont_projection = nn.Sequential(
                nn.Linear(num_continuous, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(mlp_dropout)
            )
        
        # 4. MLP نهایی
        mlp_input_dim = embedding_dim * num_categorical + (embedding_dim if num_continuous > 0 else 0)
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
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
            ویژگی‌های دسته‌ای با شکل (batch_size, num_categorical)
        x_cont : torch.Tensor
            ویژگی‌های عددی با شکل (batch_size, num_continuous)
        
        Returns:
        --------
        torch.Tensor
            خروجی مدل
        """
        batch_size = x_cat.shape[0]
        
        # 1. Embedding ویژگی‌های دسته‌ای
        cat_embedded = []
        for i in range(self.num_categorical):
            emb = self.cat_embeddings[i](x_cat[:, i])
            cat_embedded.append(emb)
        
        # شکل: (batch_size, num_categorical, embedding_dim)
        cat_embedded = torch.stack(cat_embedded, dim=1)
        
        # 2. عبور از لایه‌های Transformer
        transformer_out = cat_embedded
        for transformer in self.transformer_blocks:
            transformer_out = transformer(transformer_out)
        
        # 3. Flatten کردن خروجی Transformer
        transformer_flat = transformer_out.reshape(batch_size, -1)
        
        # 4. پردازش ویژگی‌های عددی
        if self.num_continuous > 0:
            cont_embedded = self.cont_projection(x_cont)
            # ترکیب با خروجی Transformer
            combined = torch.cat([transformer_flat, cont_embedded], dim=1)
        else:
            combined = transformer_flat
        
        # 5. MLP نهایی
        output = self.mlp(combined)
        
        return output.squeeze() if output.shape[1] == 1 else output
    
    def get_attention_weights(self, x_cat):
        """
        دریافت وزن‌های attention برای تفسیرپذیری
        
        پارامترها:
        -----------
        x_cat : torch.Tensor
            ویژگی‌های دسته‌ای
        
        Returns:
        --------
        list
            وزن‌های attention هر لایه
        """
        self.eval()
        attention_weights = []
        
        with torch.no_grad():
            # Embedding
            cat_embedded = []
            for i in range(self.num_categorical):
                emb = self.cat_embeddings[i](x_cat[:, i])
                cat_embedded.append(emb)
            
            cat_embedded = torch.stack(cat_embedded, dim=1)
            
            # جمع‌آوری وزن‌های attention از هر لایه
            x = cat_embedded
            for transformer in self.transformer_blocks:
                # دسترسی به وزن‌های attention (در صورت نیاز)
                if hasattr(transformer.self_attn, 'get_attention_weights'):
                    attn_weights = transformer.self_attn.get_attention_weights(x)
                    attention_weights.append(attn_weights)
                
                x = transformer(x)
        
        return attention_weights


# ============================================
# مدل‌های کمکی
# ============================================

class EarlyStopping:
    """
    Early stopping برای جلوگیری از overfitting
    """
    def __init__(self, patience=10, min_delta=0.001, verbose=True, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0
    
    def __call__(self, score, model, epoch):
        """
        بررسی early stopping
        
        پارامترها:
        -----------
        score : float
            امتیاز validation (بیشتر بهتر است)
        model : nn.Module
            مدل
        epoch : int
            شماره epoch
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.best_epoch = epoch
            if self.verbose:
                print(f"      🏆 بهترین امتیاز اولیه: {self.best_score:.4f}")
        
        elif score - self.best_score > self.min_delta:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"      📈 بهبود امتیاز به: {self.best_score:.4f}")
        
        else:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"      ⏳ عدم بهبود برای {self.counter}/{self.patience} epoch")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"      🛑 توقف زودهنگام در epoch {epoch}")
                
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    if self.verbose:
                        print(f"      🔄 بارگذاری بهترین مدل از epoch {self.best_epoch}")
    
    def reset(self):
        """بازنشانی"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0


class ModelUtils:
    """
    توابع کمکی برای مدل‌ها
    """
    
    @staticmethod
    def count_parameters(model):
        """
        شمارش تعداد پارامترهای مدل
        
        پارامترها:
        -----------
        model : nn.Module
            مدل
        
        Returns:
        --------
        dict
            تعداد پارامترها
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    @staticmethod
    def get_device():
        """
        دریافت دستگاه مناسب برای اجرا
        
        Returns:
        --------
        torch.device
            دستگاه (cuda/cpu)
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("⚠️ CUDA not available, using CPU")
        
        return device


# ============================================
# تابع تست
# ============================================

def test_models():
    """تست مدل‌ها با داده نمونه"""
    print("🧪 تست مدل‌های PyTorch")
    print("="*60)
    
    device = ModelUtils.get_device()
    
    # داده نمونه
    batch_size = 32
    n_cat = 3
    n_cont = 5
    categories = [10, 5, 8]  # 10, 5, 8 کلاس برای هر ویژگی دسته‌ای
    
    x_cat = torch.randint(0, 10, (batch_size, n_cat))
    x_cont = torch.randn(batch_size, n_cont)
    
    # 1. تست MLP
    print("\n1️⃣ تست ExamMLP:")
    mlp = ExamMLP(input_dim=10, hidden_dims=[64, 32], output_dim=1)
    mlp.to(device)
    output = mlp(x_cont.to(device))
    params = ModelUtils.count_parameters(mlp)
    print(f"   خروجی shape: {output.shape}")
    print(f"   تعداد پارامترها: {params['total']:,}")
    
    # 2. تست TabTransformer
    print("\n2️⃣ تست TabTransformer:")
    tab = TabTransformer(
        num_categorical=n_cat,
        num_continuous=n_cont,
        categories=categories,
        embedding_dim=32,
        num_heads=4,
        num_layers=3
    )
    tab.to(device)
    output = tab(x_cat.to(device), x_cont.to(device))
    params = ModelUtils.count_parameters(tab)
    print(f"   خروجی shape: {output.shape}")
    print(f"   تعداد پارامترها: {params['total']:,}")
    
    # 3. تست EarlyStopping
    print("\n3️⃣ تست EarlyStopping:")
    early_stopping = EarlyStopping(patience=5, verbose=True)
    
    # شبیه‌سازی امتیازها
    scores = [0.8, 0.82, 0.81, 0.83, 0.82, 0.82, 0.81, 0.80]
    for i, score in enumerate(scores):
        early_stopping(score, mlp, i)
        if early_stopping.early_stop:
            break
    
    print("\n✅ همه تست‌ها با موفقیت انجام شد")


if __name__ == "__main__":
    test_models()
