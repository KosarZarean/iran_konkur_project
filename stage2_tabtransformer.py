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
    
    # آماده‌سازی
    X_cat, X_cont, y = data_manager.prepare_for_tabtransformer()
    
    # تقسیم داده
    n = len(y)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n*0.7)]
    val_idx = indices[int(n*0.7):int(n*0.85)]
    test_idx = indices[int(n*0.85):]
    
    # ساخت مدل
    model = TabTransformer(
        num_categorical=X_cat.shape[1],
        num_continuous=X_cont.shape[1],
        categories=data_manager.categories
    )
    
    # آموزش
    trainer = ExamTrainer(model, model_type='tabtransformer')
    trainer.create_dataloaders(
        None, None, None, None,
        X_cat[train_idx], X_cont[train_idx],
        X_cat[val_idx], X_cont[val_idx]
    )
    
    trainer.train(epochs=50)
    trainer.plot_history('plots/stage2/training_history.jpg')
    
    # ارزیابی
    results = trainer.evaluate(None, None, X_cat[test_idx], X_cont[test_idx])
    
    # ذخیره
    torch.save(model.state_dict(), 'models/stage2/tabtransformer_model.pt')
    
    pd.DataFrame([results]).to_csv('results/stage2/tabtransformer_results.csv', index=False)
    
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
{'='*70}
"""
    
    with open('reports/stage2_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    return results


if __name__ == "__main__":
    run_stage2()
