"""
مرحله ۲: مدل TabTransformer برای داده‌های کنکور
===================================================
این ماژول شامل پیاده‌سازی مدل TabTransformer است.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# تنظیمات برای reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================================================
# کلاس‌های کمکی
# ==================================================

class DataManager:
    """مدیریت داده‌های کنکور"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path or Path('../data/konkur_data.csv')
        self.data = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.target_col = 'rank'
        
    def load_and_preprocess(self):
        """بارگذاری و پیش‌پردازش داده‌ها"""
        print("📂 در حال بارگذاری داده‌ها...")
        
        # بارگذاری داده
        self.data = pd.read_csv(self.data_path)
        print(f"✅ داده‌ها بارگذاری شدند: {self.data.shape}")
        
        # حذف ستون‌های غیرضروری
        cols_to_drop = ['Unnamed: 0', 'id']  # اضافه کردن ستون‌های غیرضروری
        existing_cols = [col for col in cols_to_drop if col in self.data.columns]
        if existing_cols:
            self.data = self.data.drop(columns=existing_cols)
        
        # شناسایی ستون‌های عددی و دسته‌ای
        self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in self.numerical_cols:
            self.numerical_cols.remove(self.target_col)
            
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        print(f"📊 ویژگی‌های عددی: {len(self.numerical_cols)}")
        print(f"📊 ویژگی‌های دسته‌ای: {len(self.categorical_cols)}")
        
        # پاکسازی داده‌ها
        self._clean_data()
        
        return self.data
    
    def _clean_data(self):
        """پاکسازی اولیه داده‌ها"""
        # حذف ردیف‌های با مقادیر گمشده
        initial_len = len(self.data)
        self.data = self.data.dropna()
        print(f"🧹 {initial_len - len(self.data)} ردیف با مقادیر گمشده حذف شدند")
        
        # حذف outliers در رتبه (مثلاً رتبه‌های بالای 20000)
        if self.target_col in self.data.columns:
            outliers = len(self.data[self.data[self.target_col] > 20000])
            self.data = self.data[self.data[self.target_col] <= 20000]
            print(f"🧹 {outliers} ردیف outlier حذف شدند")
    
    def prepare_data(self, cat_embed_dims=None):
        """آماده‌سازی داده برای TabTransformer"""
        if cat_embed_dims is None:
            cat_embed_dims = {}
            
        # انکد کردن ویژگی‌های دسته‌ای
        self.label_encoders = {}
        for col in self.categorical_cols:
            if col in self.data.columns:
                self.label_encoders[col] = LabelEncoder()
                self.data[col] = self.label_encoders[col].fit_transform(self.data[col].astype(str))
        
        # استانداردسازی ویژگی‌های عددی
        self.scaler = StandardScaler()
        if self.numerical_cols:
            self.data[self.numerical_cols] = self.scaler.fit_transform(self.data[self.numerical_cols])
        
        # محاسبه ابعاد embedding برای هر ویژگی دسته‌ای
        self.cat_embed_dims = {}
        for col in self.categorical_cols:
            if col in self.data.columns:
                n_categories = len(self.data[col].unique())
                # فرمول پیشنهادی: min(50, (n_categories + 1) // 2)
                embed_dim = cat_embed_dims.get(col, min(50, (n_categories + 1) // 2))
                self.cat_embed_dims[col] = (n_categories, embed_dim)
        
        return self.data


# ==================================================
# معماری TabTransformer
# ==================================================

class TransformerBlock(nn.Module):
    """بلوک ترنسفورمر"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class TabTransformer(nn.Module):
    """
    مدل TabTransformer برای داده‌های جدولی
    ترکیبی از embedding برای ویژگی‌های دسته‌ای و ترنسفورمر
    """
    
    def __init__(self, 
                 cat_embed_dims,      # لیست (تعداد_دسته‌ها, بعد_embedding) برای هر ویژگی دسته‌ای
                 numerical_dim,        # تعداد ویژگی‌های عددی
                 transformer_dim=64,   # بعد خروجی ترنسفورمر
                 n_heads=8,            # تعداد سرهای attention
                 n_layers=6,           # تعداد لایه‌های ترنسفورمر
                 ff_dim=128,            # بعد لایه feed-forward
                 dropout=0.1,
                 task='regression'):   # regression یا classification
        super().__init__()
        
        self.task = task
        self.numerical_dim = numerical_dim
        self.transformer_dim = transformer_dim
        
        # Embedding لایه‌ها برای ویژگی‌های دسته‌ای
        self.cat_embeddings = nn.ModuleList()
        for n_cat, embed_dim in cat_embed_dims:
            self.cat_embeddings.append(
                nn.Embedding(n_cat, embed_dim)
            )
        
        # پروجکشن embedding‌های دسته‌ای به فضای ترنسفورمر
        total_cat_dim = sum(embed_dim for _, embed_dim in cat_embed_dims)
        self.cat_proj = nn.Linear(total_cat_dim, transformer_dim) if cat_embed_dims else None
        
        # پروجکشن ویژگی‌های عددی
        if numerical_dim > 0:
            self.num_proj = nn.Linear(numerical_dim, transformer_dim)
        
        # Positional encoding (اختیاری برای ویژگی‌های دسته‌ای)
        self.pos_encoding = nn.Parameter(torch.randn(1, max(1, len(cat_embed_dims)), transformer_dim))
        
        # لایه‌های ترنسفورمر
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # لایه‌های خروجی
        self.output_layer = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, 1 if task == 'regression' else 2)  # برای طبقه‌بندی دودویی
        )
        
    def forward(self, categorical, numerical=None):
        """
        Args:
            categorical: tensor با shape (batch_size, n_cat_features)
            numerical: tensor با shape (batch_size, n_num_features) یا None
        """
        batch_size = categorical.shape[0]
        
        # Embedding ویژگی‌های دسته‌ای
        cat_embeds = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            cat_embeds.append(emb_layer(categorical[:, i]))
        
        if cat_embeds:
            # Concatenate همه embedding‌ها
            cat_combined = torch.cat(cat_embeds, dim=1)  # (batch_size, total_cat_dim)
            
            # پروجکشن به فضای ترنسفورمر
            cat_features = self.cat_proj(cat_combined).unsqueeze(1)  # (batch_size, 1, transformer_dim)
            
            # اضافه کردن positional encoding
            cat_features = cat_features + self.pos_encoding[:, :1, :]
            
            x = cat_features
        else:
            x = None
        
        # اضافه کردن ویژگی‌های عددی
        if numerical is not None and self.numerical_dim > 0:
            num_features = self.num_proj(numerical).unsqueeze(1)  # (batch_size, 1, transformer_dim)
            if x is None:
                x = num_features
            else:
                x = torch.cat([x, num_features], dim=1)  # (batch_size, n_tokens, transformer_dim)
        
        # عبور از لایه‌های ترنسفورمر
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # گرفتن میانگین روی تمام token‌ها
        x = x.mean(dim=1)  # (batch_size, transformer_dim)
        
        # لایه خروجی
        output = self.output_layer(x)
        
        if self.task == 'regression':
            return output.squeeze(-1)
        else:
            return output


# ==================================================
# آموزش و ارزیابی
# ==================================================

def create_data_loaders(X_cat, X_num, y, train_idx, val_idx, test_idx, batch_size=128):
    """ایجاد DataLoader برای آموزش"""
    
    # تبدیل به tensor
    X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
    X_num_tensor = torch.tensor(X_num, dtype=torch.float32) if X_num is not None else None
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # داده‌های آموزش
    train_cat = X_cat_tensor[train_idx]
    train_num = X_num_tensor[train_idx] if X_num is not None else None
    train_y = y_tensor[train_idx]
    
    # داده‌های اعتبارسنجی
    val_cat = X_cat_tensor[val_idx]
    val_num = X_num_tensor[val_idx] if X_num is not None else None
    val_y = y_tensor[val_idx]
    
    # داده‌های آزمایش
    test_cat = X_cat_tensor[test_idx]
    test_num = X_num_tensor[test_idx] if X_num is not None else None
    test_y = y_tensor[test_idx]
    
    # ایجاد Dataset
    if X_num is not None:
        train_dataset = TensorDataset(train_cat, train_num, train_y)
        val_dataset = TensorDataset(val_cat, val_num, val_y)
        test_dataset = TensorDataset(test_cat, test_num, test_y)
    else:
        train_dataset = TensorDataset(train_cat, train_y)
        val_dataset = TensorDataset(val_cat, val_y)
        test_dataset = TensorDataset(test_cat, test_y)
    
    # ایجاد DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """یک epoch آموزش"""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        if len(batch) == 3:
            cat, num, y = batch
            cat, num, y = cat.to(device), num.to(device), y.to(device)
        else:
            cat, y = batch
            cat, y = cat.to(device), y.to(device)
            num = None
        
        optimizer.zero_grad()
        output = model(cat, num)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """ارزیابی بر روی داده‌های اعتبارسنجی"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                cat, num, y = batch
                cat, num, y = cat.to(device), num.to(device), y.to(device)
            else:
                cat, y = batch
                cat, y = cat.to(device), y.to(device)
                num = None
            
            output = model(cat, num)
            loss = criterion(output, y)
            
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    return total_loss / len(val_loader), rmse, r2


def train_tabtransformer(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda', patience=10):
    """آموزش کامل مدل TabTransformer"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_r2': []}
    
    print("🚀 شروع آموزش TabTransformer...")
    
    for epoch in range(epochs):
        # آموزش
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # اعتبارسنجی
        val_loss, val_rmse, val_r2 = validate(model, val_loader, criterion, device)
        
        # به‌روزرسانی scheduler
        scheduler.step(val_loss)
        
        # ذخیره تاریخچه
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        
        # چاپ پیشرفت
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | RMSE: {val_rmse:.2f} | R²: {val_r2:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # ذخیره بهترین مدل
            torch.save(model.state_dict(), 'best_tabtransformer.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️ Early stopping در epoch {epoch+1}")
                break
    
    # بارگذاری بهترین مدل
    model.load_state_dict(torch.load('best_tabtransformer.pt'))
    
    return model, history


# ==================================================
# اجرای اصلی مرحله ۲
# ==================================================

def run_stage2():
    """تابع اصلی اجرای مرحله ۲"""
    
    print("\n" + "="*70)
    print("🎯 مرحله ۲: مدل TabTransformer")
    print("="*70 + "\n")
    
    # ایجاد پوشه‌های خروجی
    Path("results/stage2").mkdir(parents=True, exist_ok=True)
    Path("plots/stage2").mkdir(parents=True, exist_ok=True)
    
    # 1. بارگذاری و آماده‌سازی داده‌ها
    print("📁 مرحله ۱: آماده‌سازی داده‌ها")
    print("-" * 50)
    
    data_manager = DataManager()
    data = data_manager.load_and_preprocess()
    
    # آماده‌سازی داده برای TabTransformer
    cat_embed_dims_config = {}  # می‌توانید ابعاد دلخواه تنظیم کنید
    data = data_manager.prepare_data(cat_embed_dims_config)
    
    print(f"\n✅ داده‌ها آماده شدند:")
    print(f"   - تعداد نمونه‌ها: {len(data)}")
    print(f"   - ویژگی‌های عددی: {len(data_manager.numerical_cols)}")
    print(f"   - ویژگی‌های دسته‌ای: {len(data_manager.categorical_cols)}")
    
    # 2. آماده‌سازی ویژگی‌ها و هدف
    print("\n" + "📊 مرحله ۲: آماده‌سازی ویژگی‌ها")
    print("-" * 50)
    
    # ویژگی‌های دسته‌ای
    if data_manager.categorical_cols:
        X_cat = data[data_manager.categorical_cols].values.astype(np.int64)
        cat_embed_dims = [(len(data[col].unique()), 
                          data_manager.cat_embed_dims[col][1]) 
                         for col in data_manager.categorical_cols]
    else:
        X_cat = np.zeros((len(data), 0))
        cat_embed_dims = []
    
    # ویژگی‌های عددی
    if data_manager.numerical_cols:
        X_num = data[data_manager.numerical_cols].values.astype(np.float32)
        numerical_dim = len(data_manager.numerical_cols)
    else:
        X_num = None
        numerical_dim = 0
    
    # هدف
    y = data[data_manager.target_col].values.astype(np.float32)
    
    print(f"   X_cat shape: {X_cat.shape}")
    print(f"   X_num shape: {X_num.shape if X_num is not None else 'None'}")
    print(f"   y shape: {y.shape}")
    
    # 3. تقسیم داده‌ها
    print("\n" + "✂️ مرحله ۳: تقسیم داده‌ها")
    print("-" * 50)
    
    # تقسیم 70-15-15
    n = len(data)
    train_idx = np.random.choice(n, int(0.7 * n), replace=False)
    remaining = np.setdiff1d(np.arange(n), train_idx)
    val_idx = np.random.choice(remaining, int(0.5 * len(remaining)), replace=False)
    test_idx = np.setdiff1d(remaining, val_idx)
    
    print(f"   Train: {len(train_idx)} samples")
    print(f"   Val: {len(val_idx)} samples")
    print(f"   Test: {len(test_idx)} samples")
    
    # 4. ایجاد DataLoader
    print("\n" + "🔄 مرحله ۴: ایجاد DataLoader")
    print("-" * 50)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        X_cat, X_num, y, train_idx, val_idx, test_idx, batch_size=128
    )
    
    print("✅ DataLoaderها ایجاد شدند")
    
    # 5. ایجاد مدل
    print("\n" + "🏗️ مرحله ۵: ایجاد مدل TabTransformer")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   دستگاه: {device}")
    
    model = TabTransformer(
        cat_embed_dims=cat_embed_dims,
        numerical_dim=numerical_dim,
        transformer_dim=64,
        n_heads=8,
        n_layers=4,
        ff_dim=128,
        dropout=0.1,
        task='regression'
    ).to(device)
    
    print(f"\n📋 معماری مدل:")
    print(f"   - تعداد پارامترها: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - لایه‌های ترنسفورمر: 4")
    print(f"   - بعد ترنسفورمر: 64")
    print(f"   - تعداد سرهای attention: 8")
    
    # 6. آموزش مدل
    print("\n" + "🚀 مرحله ۶: آموزش مدل")
    print("-" * 50)
    
    start_time = time.time()
    model, history = train_tabtransformer(
        model, train_loader, val_loader,
        epochs=50, lr=1e-3, device=device, patience=10
    )
    training_time = time.time() - start_time
    
    print(f"\n✅ آموزش کامل شد. زمان: {training_time:.2f} ثانیه")
    
    # 7. ارزیابی نهایی
    print("\n" + "📊 مرحله ۷: ارزیابی نهایی")
    print("-" * 50)
    
    criterion = nn.MSELoss()
    test_loss, test_rmse, test_r2 = validate(model, test_loader, criterion, device)
    
    print(f"\n📈 نتایج نهایی روی داده‌های آزمایش:")
    print(f"   RMSE: {test_rmse:.2f}")
    print(f"   R²: {test_r2:.4f}")
    
    # 8. رسم نمودارهای آموزش
    print("\n" + "📈 مرحله ۸: رسم نمودارها")
    print("-" * 50)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0].plot(history['val_loss'], label='Val Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RMSE
    axes[1].plot(history['val_rmse'], color='orange', alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Validation RMSE')
    axes[1].grid(True, alpha=0.3)
    
    # R²
    axes[2].plot(history['val_r2'], color='green', alpha=0.8)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('R²')
    axes[2].set_title('Validation R²')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/stage2/training_history.jpg', dpi=100, bbox_inches='tight')
    plt.show()
    print("📊 نمودار در plots/stage2/training_history.jpg ذخیره شد")
    
    # 9. ذخیره نتایج
    print("\n" + "💾 مرحله ۹: ذخیره نتایج")
    print("-" * 50)
    
    results = {
        'stage': 2,
        'model': 'TabTransformer',
        'timestamp': str(datetime.now()),
        'data_info': {
            'n_samples': len(data),
            'n_categorical': len(data_manager.categorical_cols),
            'n_numerical': len(data_manager.numerical_cols)
        },
        'model_params': {
            'transformer_dim': 64,
            'n_heads': 8,
            'n_layers': 4,
            'ff_dim': 128,
            'dropout': 0.1,
            'total_params': sum(p.numel() for p in model.parameters())
        },
        'training': {
            'epochs': len(history['train_loss']),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_rmse': min(history['val_rmse']),
            'best_val_r2': max(history['val_r2']),
            'training_time_seconds': training_time
        },
        'test_results': {
            'rmse': float(test_rmse),
            'r2': float(test_r2),
            'loss': float(test_loss)
        }
    }
    
    # ذخیره JSON
    with open('results/stage2/tabtransformer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ذخیره CSV نتایج
    results_df = pd.DataFrame({
        'metric': ['RMSE', 'R²', 'Train Loss', 'Val Loss', 'Training Time (s)'],
        'value': [test_rmse, test_r2, history['train_loss'][-1], history['val_loss'][-1], training_time]
    })
    results_df.to_csv('results/stage2/tabtransformer_results.csv', index=False)
    
    print("✅ نتایج در results/stage2/ ذخیره شدند")
    
    # 10. گزارش نهایی
    print("\n" + "="*70)
    print("📊 گزارش نهایی مرحله ۲")
    print("="*70)
    print(f"تاریخ: {datetime.now()}")
    print("\n📈 نتایج TabTransformer:")
    print(f"   RMSE: {test_rmse:.2f}")
    print(f"   R²: {test_r2:.4f}")
    print(f"   زمان آموزش: {training_time:.2f} ثانیه")
    print("\n✅ مرحله ۲ با موفقیت به پایان رسید!")
    print("="*70 + "\n")
    
    return model, history, results


# ==================================================
# اجرای مستقیم
# ==================================================

if __name__ == "__main__":
    run_stage2()
