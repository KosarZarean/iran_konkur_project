#!/usr/bin/env python3
"""
مرحله ۱: مدل‌های پایه
"""

import os
import sys
import pandas as pd
from datetime import datetime

sys.path.append('src')
from exam_data_manager import ExamDataManager
from exam_baseline import BaselineModels


def run_stage1(data_path='data/iran_exam.csv'):
    print("\n" + "="*70)
    print("🎯 مرحله ۱: مدل‌های پایه و سنتی")
    print("="*70)
    
    os.makedirs('results/stage1', exist_ok=True)
    os.makedirs('plots/stage1', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # بارگذاری داده
    data_manager = ExamDataManager()
    df = data_manager.load_and_prepare_data(data_path, 'regression')
    
    # آماده‌سازی
    data_manager.prepare_for_traditional_models()
    data_manager.create_train_val_test_split()
    
    # آموزش
    baseline = BaselineModels(
        data_manager.X_train, data_manager.y_train,
        data_manager.X_val, data_manager.y_val,
        data_manager.X_test, data_manager.y_test
    )
    
    baseline.define_models()
    results = baseline.train_and_evaluate()
    
    # ذخیره
    results.to_csv('results/stage1/baseline_results.csv', index=False)
    baseline.plot_comparison('plots/stage1/baseline_comparison.jpg')
    
    # گزارش
    best = results.loc[results['Test RMSE'].idxmin()]
    report = f"""
{'='*70}
📊 گزارش مرحله ۱
{'='*70}
تاریخ: {datetime.now()}

📊 بهترین مدل: {best['Model']}
   RMSE: {best['Test RMSE']:.2f}
   R²: {best['Test R2']:.4f}
   زمان: {best['Time (s)']:.2f} ثانیه

📈 نتایج کامل در: results/stage1/baseline_results.csv
📊 نمودار در: plots/stage1/baseline_comparison.jpg
{'='*70}
"""
    
    with open('reports/stage1_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    return results


if __name__ == "__main__":
    run_stage1()
