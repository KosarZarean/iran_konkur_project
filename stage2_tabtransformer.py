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
    """اجرای مرحله ۲"""
    print("\n" + "="*70)
    print("🎯 مرحله ۲: TabTransformer")
    print("="*70)
    
    os.makedirs('results/stage2', exist_ok=True)
    os.makedirs('plots/stage2', exist_ok=True)
    os.makedirs('models/stage2', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 1. بارگذاری داده
    print("\n📊 مرحله ۲-۱: بارگذاری داده‌ها...")
    data_manager = ExamDataManager()
    df = data_manager.load_and_prepare_data(data_path, 'regression')
    
    # 2. آماده‌سازی
    print("\n🔄 مرحله ۲-۲: آماده‌سازی داده...")
    X_cat, X_cont, y = data_manager.prepare_for_tabtransformer()
    
    # 3. تقسیم داده
    print("\n✂️ مرحله ۲-۳: تقسیم داده‌ها...")
    n = len(y)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n*0.7)]
    val_idx = indices[int(n*0.7):int(n*0.85)]
    test_idx = indices[int(n*0.85):]
    
    # 4. ساخت مدل
    print("\n🏗️ مرحله ۲-۴: ساخت مدل...")
    model = TabTransformer(
        num_categorical=X_cat.shape[1],
        num_continuous=X_cont.shape[1],
        categories=data_manager.categories,
        embedding_dim=32,
        num_heads=4,
        num_layers=3,
        mlp_hidden_dims=[128, 64],
        output_dim=1
    )
    
    # 5. آموزش
    print("\n🚀 مرحله ۲-۵: آموزش...")
    trainer = ExamTrainer(model, model_type='tabtransformer', model_name='tabtransformer_stage2')
    
    trainer.create_dataloaders(
        X_cat_train=X_cat[train_idx],
        X_cont_train=X_cont[train_idx],
        y_train=y[train_idx],
        X_cat_val=X_cat[val_idx],
        X_cont_val=X_cont[val_idx],
        y_val=y[val_idx],
        batch_size=64
    )
    
    trainer.train(epochs=50, lr=0.001, patience=10)
    trainer.plot_history('plots/stage2/training_history.jpg')
    
    # 6. ارزیابی
    print("\n📊 مرحله ۲-۶: ارزیابی...")
    results = trainer.evaluate(
        X_cat_test=X_cat[test_idx],
        X_cont_test=X_cont[test_idx],
        y_test=y[test_idx]
    )
    
    # 7. ذخیره
    results_df = pd.DataFrame([{
        'Model': 'TabTransformer',
        'RMSE': results['rmse'],
        'R2': results['r2'],
        'MAE': results['mae']
    }])
    results_df.to_csv('results/stage2/tabtransformer_results.csv', index=False)
    torch.save(model.state_dict(), 'models/stage2/tabtransformer_model.pt')
    
    print(f"\n✅ نتایج: RMSE={results['rmse']:.2f}, R²={results['r2']:.4f}")
    
    return results


if __name__ == "__main__":
    run_stage2()
