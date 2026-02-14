#!/usr/bin/env python3
"""
مرحله ۲: TabTransformer
رفع شده - مشکل None در داده‌ها برطرف شده
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# اضافه کردن مسیر src
sys.path.append('src')

from exam_data_manager import ExamDataManager
from exam_models import TabTransformer
from exam_trainer import ExamTrainer


def run_stage2(data_path='data/iran_exam.csv'):
    print("\n" + "="*70)
    print("🎯 مرحله ۲: TabTransformer")
    print("="*70)
    
    # ایجاد پوشه‌های خروجی
    os.makedirs('results/stage2', exist_ok=True)
    os.makedirs('plots/stage2', exist_ok=True)
    os.makedirs('models/stage2', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # ============================================
    # ۱. بارگذاری داده
    # ============================================
    print("\n📊 مرحله ۲-۱: بارگذاری داده‌ها...")
    data_manager = ExamDataManager()
    df = data_manager.load_and_prepare_data(data_path, 'regression')
    
    # ============================================
    # ۲. آماده‌سازی برای TabTransformer
    # ============================================
    print("\n🔄 مرحله ۲-۲: آماده‌سازی داده برای TabTransformer...")
    X_cat, X_cont, y = data_manager.prepare_for_tabtransformer()
    
    print(f"\n   ✅ X_cat shape: {X_cat.shape}")
    print(f"   ✅ X_cont shape: {X_cont.shape}")
    print(f"   ✅ y shape: {y.shape}")
    
    # ============================================
    # ۳. تقسیم داده (با نمونه‌گیری برای تست سریع)
    # ============================================
    print("\n✂️ مرحله ۲-۳: تقسیم داده‌ها...")
    
    # برای تست سریع، فقط ۱۰۰۰۰ نمونه اول را استفاده می‌کنیم
    n_samples = min(10000, len(y))
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    print(f"   - آموزش: {len(train_idx)} نمونه")
    print(f"   - اعتبارسنجی: {len(val_idx)} نمونه")
    print(f"   - آزمایش: {len(test_idx)} نمونه")
    
    # ============================================
    # ۴. ساخت مدل
    # ============================================
    print("\n🏗️ مرحله ۲-۴: ساخت معماری TabTransformer...")
    model = TabTransformer(
        num_categorical=X_cat.shape[1],
        num_continuous=X_cont.shape[1],
        categories=data_manager.categories,
        embedding_dim=32,
        num_heads=4,
        num_layers=3,
        mlp_hidden_dims=[128, 64],
        mlp_dropout=0.2,
        transformer_dropout=0.1,
        output_dim=1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ تعداد کل پارامترها: {total_params:,}")
    
    # ============================================
    # ۵. ایجاد DataLoader (رفع مشکل اصلی)
    # ============================================
    print("\n📦 مرحله ۲-۵: ایجاد DataLoader...")
    
    trainer = ExamTrainer(model, model_type='tabtransformer', model_name='tabtransformer')
    
    # ارسال داده‌ها به trainer
    trainer.create_dataloaders(
        X_train=None, y_train=None,  # برای tabtransformer اینها استفاده نمی‌شوند
        X_val=None, y_val=None,
        X_cat_train=X_cat[train_idx],
        X_cont_train=X_cont[train_idx],
        y_train=y[train_idx],
        X_cat_val=X_cat[val_idx],
        X_cont_val=X_cont[val_idx],
        y_val=y[val_idx],
        batch_size=64
    )
    
    # ============================================
    # ۶. آموزش مدل
    # ============================================
    print("\n🚀 مرحله ۲-۶: آموزش مدل...")
    trainer.train(
        epochs=30,  # تعداد کمتر برای تست سریع
        lr=0.001,
        weight_decay=1e-5,
        patience=10,
        verbose=True
    )
    
    # ============================================
    # ۷. رسم نمودار آموزش
    # ============================================
    print("\n📈 مرحله ۲-۷: رسم نمودارهای آموزش...")
    trainer.plot_history('plots/stage2/training_history.jpg')
    
    # ============================================
    # ۸. ارزیابی مدل
    # ============================================
    print("\n🧪 مرحله ۲-۸: ارزیابی مدل...")
    results = trainer.evaluate(
        X_test=None, y_test=None,  # برای tabtransformer
        X_cat_test=X_cat[test_idx],
        X_cont_test=X_cont[test_idx],
        y_test=y[test_idx]
    )
    
    # ============================================
    # ۹. ذخیره مدل
    # ============================================
    print("\n💾 مرحله ۲-۹: ذخیره مدل...")
    trainer.save_model('tabtransformer_model.pt')
    
    # ============================================
    # ۱۰. ذخیره نتایج
    # ============================================
    results_df = pd.DataFrame([{
        'Model': 'TabTransformer',
        'RMSE': results['rmse'],
        'R2': results['r2']
    }])
    results_df.to_csv('results/stage2/tabtransformer_results.csv', index=False, encoding='utf-8-sig')
    
    # ============================================
    # ۱۱. ایجاد گزارش
    # ============================================
    report = f"""
{'='*70}
📊 گزارش مرحله ۲: TabTransformer
{'='*70}
تاریخ: {datetime.now()}

📊 اطلاعات داده:
   تعداد کل نمونه‌ها: {len(y)}
   نمونه‌های استفاده شده: {n_samples}
   ویژگی‌های دسته‌ای: {X_cat.shape[1]}
   ویژگی‌های عددی: {X_cont.shape[1]}

🏗️ معماری مدل:
   embedding_dim: 32
   num_heads: 4
   num_layers: 3
   mlp_hidden: [128, 64]
   dropout: 0.2
   تعداد پارامترها: {total_params:,}

📈 نتایج آموزش:
   بهترین RMSE آموزش: {min(trainer.history['train_rmse']):.2f}
   بهترین RMSE اعتبارسنجی: {min(trainer.history['val_rmse']):.2f}

🎯 نتایج نهایی روی داده تست:
   RMSE: {results['rmse']:.2f}
   R²: {results['r2']:.4f}

📁 فایل‌های ذخیره شده:
   - مدل: models/stage2/tabtransformer_model.pt
   - نتایج: results/stage2/tabtransformer_results.csv
   - نمودار: plots/stage2/training_history.jpg
{'='*70}
"""
    
    with open('reports/stage2_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    return results, trainer, report


if __name__ == "__main__":
    run_stage2()
