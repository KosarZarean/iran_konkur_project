#!/usr/bin/env python3
"""
مرحله ۳: جاسازی عددی
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append('src')
from exam_data_manager import ExamDataManager
from exam_numerical_embeddings import TabTransformerWithNumEmbedding
from exam_trainer import ExamTrainer


def run_stage3(data_path='data/iran_exam.csv'):
    """اجرای مرحله ۳"""
    print("\n" + "="*70)
    print("🎯 مرحله ۳: جاسازی عددی")
    print("="*70)
    
    os.makedirs('results/stage3', exist_ok=True)
    os.makedirs('plots/stage3', exist_ok=True)
    os.makedirs('models/stage3', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 1. بارگذاری داده
    print("\n📊 مرحله ۳-۱: بارگذاری داده‌ها...")
    data_manager = ExamDataManager()
    df = data_manager.load_and_prepare_data(data_path, 'regression')
    
    # 2. آماده‌سازی
    print("\n🔄 مرحله ۳-۲: آماده‌سازی داده...")
    X_cat, X_cont, y = data_manager.prepare_for_tabtransformer()
    
    # 3. تقسیم داده
    print("\n✂️ مرحله ۳-۳: تقسیم داده‌ها...")
    n = len(y)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n*0.7)]
    val_idx = indices[int(n*0.7):int(n*0.85)]
    test_idx = indices[int(n*0.85):]
    
    # 4. آزمایش روش‌های مختلف جاسازی
    methods = ['ple', 'periodic', 'bucket']
    method_names = {
        'ple': 'Piecewise Linear',
        'periodic': 'Periodic',
        'bucket': 'Bucket'
    }
    
    results_list = []
    
    print("\n🧪 مرحله ۳-۴: آزمایش روش‌های جاسازی عددی")
    print("-" * 60)
    
    for method in methods:
        print(f"\n📌 روش: {method_names[method]}")
        
        # ساخت مدل با جاسازی عددی
        model = TabTransformerWithNumEmbedding(
            num_categorical=X_cat.shape[1],
            num_continuous=X_cont.shape[1],
            categories=data_manager.categories,
            num_embedding_type=method,
            embedding_dim=32,
            num_heads=4,
            num_layers=3,
            mlp_hidden_dims=[128, 64],
            output_dim=1
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   تعداد پارامترها: {total_params:,}")
        
        # ایجاد Trainer
        trainer = ExamTrainer(
            model=model,
            model_type='tabtransformer',
            model_name=f'tabtransformer_{method}',
            save_dir='models/stage3'
        )
        
        # ایجاد DataLoader
        print("\n   📦 ایجاد DataLoader...")
        trainer.create_dataloaders(
            X_cat_train=X_cat[train_idx],
            X_cont_train=X_cont[train_idx],
            y_train=y[train_idx],
            X_cat_val=X_cat[val_idx],
            X_cont_val=X_cont[val_idx],
            y_val=y[val_idx],
            batch_size=64
        )
        
        # آموزش مدل
        print("   🚀 آموزش مدل...")
        trainer.train(epochs=30, lr=0.001, patience=8, verbose=False)
        
        # ارزیابی مدل
        print("   📊 ارزیابی مدل...")
        results = trainer.evaluate(
            X_cat_test=X_cat[test_idx],
            X_cont_test=X_cont[test_idx],
            y_test=y[test_idx]
        )
        
        results['method'] = method
        results['method_name'] = method_names[method]
        results['parameters'] = total_params
        results_list.append(results)
        
        print(f"   ✅ RMSE: {results['rmse']:.2f}, R²: {results['r2']:.4f}")
        
        # رسم نمودار آموزش
        trainer.plot_history(f'plots/stage3/{method}_history.jpg')
    
    # 5. ذخیره نتایج
    print("\n💾 مرحله ۳-۵: ذخیره نتایج...")
    results_df = pd.DataFrame([{
        'Method': r['method_name'],
        'RMSE': r['rmse'],
        'R2': r['r2'],
        'MAE': r['mae'],
        'Parameters': r['parameters']
    } for r in results_list])
    
    results_df.to_csv('results/stage3/embeddings_results.csv', index=False, encoding='utf-8-sig')
    
    # 6. رسم نمودار مقایسه
    print("\n📈 مرحله ۳-۶: رسم نمودار مقایسه...")
    plot_comparison(results_list)
    
    # 7. گزارش
    print("\n📝 مرحله ۳-۷: ایجاد گزارش...")
    report = generate_report(results_list, data_manager)
    
    with open('reports/stage3_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*70)
    print("✅ مرحله ۳ با موفقیت کامل شد!")
    print("="*70)
    print(report)
    
    return results_list


def plot_comparison(results_list):
    """رسم نمودار مقایسه روش‌های جاسازی"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    methods = [r['method_name'] for r in results_list]
    rmse_values = [r['rmse'] for r in results_list]
    r2_values = [r['r2'] for r in results_list]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # نمودار RMSE
    bars1 = axes[0].bar(methods, rmse_values, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Embedding Method', fontsize=12)
    axes[0].set_ylabel('RMSE (lower is better)', fontsize=12)
    axes[0].set_title('RMSE Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, rmse_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # نمودار R²
    bars2 = axes[1].bar(methods, r2_values, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Embedding Method', fontsize=12)
    axes[1].set_ylabel('R² (higher is better)', fontsize=12)
    axes[1].set_title('R² Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, r2_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Numerical Embeddings Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('plots/stage3/embeddings_comparison.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def generate_report(results_list, data_manager):
    """ایجاد گزارش مرحله ۳"""
    report = []
    report.append("="*70)
    report.append("📊 گزارش مرحله ۳: جاسازی عددی")
    report.append("="*70)
    report.append(f"تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # اطلاعات داده
    report.append("📋 اطلاعات داده:")
    report.append(f"  - تعداد کل نمونه‌ها: {len(data_manager.df):,}")
    report.append(f"  - تعداد ویژگی‌های عددی: {data_manager.X_cont.shape[1]}")
    report.append("")
    
    # نتایج روش‌ها
    report.append("📊 مقایسه روش‌های جاسازی عددی:")
    report.append("-" * 60)
    
    best_method = min(results_list, key=lambda x: x['rmse'])
    
    for r in results_list:
        report.append(f"\n📌 روش: {r['method_name']}")
        report.append(f"   - RMSE: {r['rmse']:.2f}")
        report.append(f"   - R²: {r['r2']:.4f}")
        report.append(f"   - MAE: {r['mae']:.2f}")
        report.append(f"   - تعداد پارامترها: {r['parameters']:,}")
    
    report.append("")
    report.append("-" * 60)
    report.append(f"\n🏆 بهترین روش: {best_method['method_name']}")
    report.append(f"   - RMSE: {best_method['rmse']:.2f}")
    report.append(f"   - R²: {best_method['r2']:.4f}")
    
    report.append("\n" + "="*70)
    
    return "\n".join(report)


if __name__ == "__main__":
    run_stage3()
