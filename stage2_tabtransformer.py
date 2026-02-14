#!/usr/bin/env python3
"""
مرحله ۲: TabTransformer
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.append('src')
from exam_data_manager import ExamDataManager
from exam_models import TabTransformer
from exam_trainer import ExamTrainer


def run_stage2(data_path='data/iran_exam.csv'):
    print("\n" + "="*70)
    print("🎯 مرحله ۲: TabTransformer")
    print("="*70)
    
    os.makedirs('results/stage2', exist_ok=True)
    os.makedirs('plots/stage2', exist_ok=True)
    os.makedirs('models/stage2', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # بارگذاری داده
    data_manager = ExamDataManager()
    df = data_manager.load_and_prepare_data(data_path, 'regression')
    
    # آماده‌سازی برای TabTransformer
    print("\n🔄 آماده‌سازی داده برای TabTransformer...")
    X_cat, X_cont, y = data_manager.prepare_for_tabtransformer()
    
    # تقسیم داده
    n = len(y)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n*0.7)]
    val_idx = indices[int(n*0.7):int(n*0.85)]
    test_idx = indices[int(n*0.85):]
    
    print(f"\n📊 تقسیم داده:")
    print(f"   آموزش: {len(train_idx)} نمونه")
    print(f"   اعتبارسنجی: {len(val_idx)} نمونه")
    print(f"   آزمایش: {len(test_idx)} نمونه")
    
    # ساخت مدل
    print("\n🏗️ ساخت مدل TabTransformer...")
    model = TabTransformer(
        num_categorical=X_cat.shape[1],
        num_continuous=X_cont.shape[1],
        categories=data_manager.categories,
        embedding_dim=32,
        num_heads=4,
        num_layers=3,
        mlp_hidden_dims=[128, 64],
        dropout=0.2,
        output_dim=1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   تعداد پارامترها: {total_params:,}")
    
    # آموزش
    print("\n🚀 شروع آموزش...")
    trainer = ExamTrainer(model, model_type='tabtransformer', model_name='tabtransformer_stage2')
    
    # ایجاد DataLoader - مشکل اینجا بود: y_train تکراری بود
    trainer.create_dataloaders(
        X_train=None, 
        y_train=None,
        X_val=None, 
        y_val=None,
        X_cat_train=X_cat[train_idx], 
        X_cont_train=X_cont[train_idx],
        X_cat_val=X_cat[val_idx], 
        X_cont_val=X_cont[val_idx],
        y_train=y[train_idx],      # ✅ اینجا فقط یک بار y_train
        y_val=y[val_idx],           # ✅ اینجا فقط یک بار y_val
        batch_size=64
    )
    
    trainer.train(epochs=50, lr=0.001, task_type='regression', patience=10)
    trainer.plot_history('plots/stage2/training_history.jpg')
    
    # ارزیابی
    print("\n📊 ارزیابی مدل...")
    results = trainer.evaluate(
        X_test=None, 
        y_test=None,
        X_cat_test=X_cat[test_idx], 
        X_cont_test=X_cont[test_idx],
        y_test=y[test_idx]
    )
    
    # ذخیره مدل
    torch.save(model.state_dict(), 'models/stage2/tabtransformer_model.pt')
    print(f"💾 مدل در models/stage2/tabtransformer_model.pt ذخیره شد")
    
    # ذخیره نتایج
    results_df = pd.DataFrame([{
        'Model': 'TabTransformer',
        'RMSE': results['rmse'],
        'R2': results['r2'],
        'Parameters': total_params
    }])
    results_df.to_csv('results/stage2/tabtransformer_results.csv', index=False, encoding='utf-8-sig')
    
    # گزارش
    report = f"""
{'='*70}
📊 گزارش مرحله ۲
{'='*70}
تاریخ: {datetime.now()}

📊 نتایج نهایی:
   RMSE: {results['rmse']:.2f}
   R²: {results['r2']:.4f}

📈 نمودار آموزش: plots/stage2/training_history.jpg
🤖 مدل ذخیره شده: models/stage2/tabtransformer_model.pt
📊 نتایج: results/stage2/tabtransformer_results.csv
{'='*70}
"""
    
    with open('reports/stage2_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    return results, trainer, report


if __name__ == "__main__":
    run_stage2()
