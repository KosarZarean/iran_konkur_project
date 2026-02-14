#!/usr/bin/env python3
"""
مرحله ۳: جاسازی عددی
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append('src')
from exam_data_manager import ExamDataManager
from exam_numerical_embeddings import TabTransformerWithNumEmbedding
from exam_trainer import ExamTrainer


def run_stage3(data_path='data/iran_exam.csv'):
    print("\n" + "="*70)
    print("🎯 مرحله ۳: جاسازی عددی")
    print("="*70)
    
    os.makedirs('results/stage3', exist_ok=True)
    os.makedirs('plots/stage3', exist_ok=True)
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
    
    # آزمایش روش‌ها
    methods = ['ple', 'periodic', 'bucket']
    method_names = {'ple': 'Piecewise Linear', 'periodic': 'Periodic', 'bucket': 'Bucket'}
    
    results_list = []
    
    for method in methods:
        print(f"\n📌 روش: {method_names[method]}")
        
        model = TabTransformerWithNumEmbedding(
            num_categorical=X_cat.shape[1],
            num_continuous=X_cont.shape[1],
            categories=data_manager.categories,
            num_embedding_type=method
        )
        
        trainer = ExamTrainer(model, model_type='tabtransformer')
        trainer.create_dataloaders(
            None, None, None, None,
            X_cat[train_idx], X_cont[train_idx],
            X_cat[val_idx], X_cont[val_idx]
        )
        
        trainer.train(epochs=30)
        trainer.plot_history(f'plots/stage3/{method}_history.jpg')
        
        res = trainer.evaluate(None, None, X_cat[test_idx], X_cont[test_idx])
        res['method'] = method_names[method]
        results_list.append(res)
        
        print(f"   ✅ RMSE: {res['rmse']:.2f}, R²: {res['r2']:.4f}")
    
    # ذخیره نتایج
    df_results = pd.DataFrame([{
        'Method': r['method'],
        'RMSE': r['rmse'],
        'R2': r['r2']
    } for r in results_list])
    
    df_results.to_csv('results/stage3/embeddings_results.csv', index=False)
    
    # رسم نمودار مقایسه
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(df_results['Method'], df_results['RMSE'], color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(df_results['Method'], df_results['R2'], color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('R²')
    plt.title('R² Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/stage3/embeddings_comparison.jpg', dpi=300)
    plt.show()
    
    # گزارش
    best = df_results.loc[df_results['RMSE'].idxmin()]
    report = f"""
{'='*70}
📊 گزارش مرحله ۳
{'='*70}
تاریخ: {datetime.now()}

📊 بهترین روش: {best['Method']}
   RMSE: {best['RMSE']:.2f}
   R²: {best['R2']:.4f}

📈 نتایج کامل:
{df_results.to_string()}

📊 نمودارها در: plots/stage3/
{'='*70}
"""
    
    with open('reports/stage3_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    return df_results


if __name__ == "__main__":
    run_stage3()
