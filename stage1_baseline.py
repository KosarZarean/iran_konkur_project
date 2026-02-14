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
    """اجرای مرحله ۱"""
    print("\n" + "="*70)
    print("🎯 مرحله ۱: مدل‌های پایه")
    print("="*70)
    
    # ایجاد پوشه‌ها
    os.makedirs('results/stage1', exist_ok=True)
    os.makedirs('plots/stage1', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 1. بارگذاری داده
    print("\n📊 مرحله ۱-۱: بارگذاری داده‌ها...")
    data_manager = ExamDataManager()
    df = data_manager.load_and_prepare_data(data_path, 'regression')
    
    # 2. آماده‌سازی
    print("\n🔄 مرحله ۱-۲: آماده‌سازی ویژگی‌ها...")
    data_manager.prepare_for_traditional_models()
    data_manager.create_train_val_test_split()
    
    # 3. آموزش
    print("\n🤖 مرحله ۱-۳: آموزش مدل‌ها...")
    baseline = BaselineModels(
        data_manager.X_train, data_manager.y_train,
        data_manager.X_val, data_manager.y_val,
        data_manager.X_test, data_manager.y_test
    )
    
    baseline.define_models()
    results = baseline.train_and_evaluate()
    
    # 4. ذخیره
    print("\n💾 مرحله ۱-۴: ذخیره نتایج...")
    baseline.save_results('results/stage1/baseline_results.csv')
    baseline.plot_comparison('plots/stage1/baseline_comparison.jpg')
    baseline.generate_report('reports/stage1_report.txt')
    
    print("\n" + "="*70)
    print("✅ مرحله ۱ کامل شد")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_stage1()
