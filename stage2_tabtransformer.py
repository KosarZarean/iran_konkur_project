#!/usr/bin/env python3
"""
مرحله ۲: TabTransformer
اجرای جداگانه برای دریافت خروجی و گزارش
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
    """
    اجرای مرحله ۲ و ذخیره نتایج
    """
    print("\n" + "="*70)
    print("🎯 مرحله ۲: TabTransformer")
    print("="*70)
    print(f"📅 زمان شروع: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # ایجاد پوشه‌های خروجی
    os.makedirs('results/stage2', exist_ok=True)
    os.makedirs('plots/stage2', exist_ok=True)
    os.makedirs('models/stage2', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # ۱. بارگذاری داده
    print("\n📊 مرحله ۲-۱: بارگذاری داده‌ها...")
    data_manager = ExamDataManager()
    df = data_manager.load_and_prepare_data(data_path, 'regression')
    
    # ۲. آماده‌سازی برای TabTransformer
    print("\n🔄 مرحله ۲-۲: آماده‌سازی داده برای TabTransformer...")
    X_cat, X_cont, y = data_manager.prepare_for_tabtransformer()
    
    print(f"   X_cat shape: {X_cat.shape}")
    print(f"   X_cont shape: {X_cont.shape}")
    print(f"   categories: {data_manager.categories}")
    
    # ۳. تقسیم داده
    print("\n✂️ مرحله ۲-۳: تقسیم داده‌ها...")
    n = len(y)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n*0.7)]
    val_idx = indices[int(n*0.7):int(n*0.85)]
    test_idx = indices[int(n*0.85):]
    
    print(f"   آموزش: {len(train_idx)} نمونه ({len(train_idx)/n*100:.1f}%)")
    print(f"   اعتبارسنجی: {len(val_idx)} نمونه ({len(val_idx)/n*100:.1f}%)")
    print(f"   آزمایش: {len(test_idx)} نمونه ({len(test_idx)/n*100:.1f}%)")
    
    # ۴. ساخت مدل
    print("\n🏗️ مرحله ۲-۴: ساخت مدل TabTransformer...")
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
    print(f"   تعداد کل پارامترها: {total_params:,}")
    
    # ۵. ایجاد Trainer
    print("\n🤖 مرحله ۲-۵: ایجاد Trainer...")
    trainer = ExamTrainer(
        model=model, 
        model_type='tabtransformer', 
        model_name='tabtransformer_stage2',
        save_dir='models/stage2'
    )
    
    # ۶. ایجاد DataLoader - ✅ اینجا مشکل قبلی را حل کردیم
    print("\n📦 مرحله ۲-۶: ایجاد DataLoader...")
    trainer.create_dataloaders(
        # داده‌های دسته‌ای و عددی برای آموزش
        X_cat_train=X_cat[train_idx],
        X_cont_train=X_cont[train_idx],
        y_train=y[train_idx],
        
        # داده‌های دسته‌ای و عددی برای اعتبارسنجی
        X_cat_val=X_cat[val_idx],
        X_cont_val=X_cont[val_idx],
        y_val=y[val_idx],
        
        # اندازه batch
        batch_size=64
    )
    
    # ۷. آموزش مدل
    print("\n🚀 مرحله ۲-۷: آموزش مدل...")
    trainer.train(
        epochs=50, 
        lr=0.001, 
        task_type='regression', 
        patience=10,
        verbose=True
    )
    
    # ۸. رسم نمودار آموزش
    print("\n📈 مرحله ۲-۸: رسم نمودار آموزش...")
    trainer.plot_history('plots/stage2/training_history.jpg')
    
    # ۹. ارزیابی مدل
    print("\n📊 مرحله ۲-۹: ارزیابی مدل...")
    results = trainer.evaluate(
        X_cat_test=X_cat[test_idx],
        X_cont_test=X_cont[test_idx],
        y_test=y[test_idx]
    )
    
    # ۱۰. ذخیره مدل
    print("\n💾 مرحله ۲-۱۰: ذخیره مدل...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_categorical': X_cat.shape[1],
            'num_continuous': X_cont.shape[1],
            'categories': data_manager.categories,
            'embedding_dim': 32,
            'num_heads': 4,
            'num_layers': 3,
            'mlp_hidden_dims': [128, 64],
            'dropout': 0.2
        },
        'results': results,
        'history': trainer.history
    }, 'models/stage2/tabtransformer_model.pt')
    
    # ۱۱. ذخیره نتایج
    print("\n💾 مرحله ۲-۱۱: ذخیره نتایج...")
    results_df = pd.DataFrame([{
        'Model': 'TabTransformer',
        'RMSE': results['rmse'],
        'R2': results['r2'],
        'MAE': results.get('mae', 0),
        'Parameters': total_params
    }])
    results_df.to_csv('results/stage2/tabtransformer_results.csv', index=False, encoding='utf-8-sig')
    
    # ۱۲. ایجاد گزارش
    print("\n📝 مرحله ۲-۱۲: ایجاد گزارش...")
    report = generate_report(results, trainer, data_manager, total_params)
    
    with open('reports/stage2_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*70)
    print("✅ مرحله ۲ با موفقیت کامل شد!")
    print("="*70)
    print(report)
    
    return results, trainer, report


def generate_report(results, trainer, data_manager, total_params):
    """
    ایجاد گزارش مرحله ۲
    """
    report = []
    report.append("="*70)
    report.append("📊 گزارش مرحله ۲: پیاده‌سازی TabTransformer")
    report.append("="*70)
    report.append(f"تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # اطلاعات معماری
    report.append("🏗️ معماری مدل:")
    report.append(f"  - تعداد ویژگی‌های دسته‌ای: {data_manager.X_cat.shape[1] if hasattr(data_manager, 'X_cat') else 0}")
    report.append(f"  - تعداد ویژگی‌های عددی: {data_manager.X_cont.shape[1] if hasattr(data_manager, 'X_cont') else 0}")
    report.append(f"  - تعداد کلاس‌های دسته‌ای: {data_manager.categories}")
    report.append(f"  - ابعاد Embedding: 32")
    report.append(f"  - تعداد لایه‌های Transformer: 3")
    report.append(f"  - تعداد Headهای Attention: 4")
    report.append(f"  - تعداد کل پارامترها: {total_params:,}")
    report.append("")
    
    # نتایج آموزش
    report.append("📈 تاریخچه آموزش:")
    report.append(f"  - بهترین Loss آموزش: {min(trainer.history['train_loss']):.4f}")
    report.append(f"  - بهترین Loss اعتبارسنجی: {min(trainer.history['val_loss']):.4f}")
    report.append(f"  - بهترین RMSE آموزش: {min(trainer.history['train_rmse']):.2f}")
    report.append(f"  - بهترین RMSE اعتبارسنجی: {min(trainer.history['val_rmse']):.2f}")
    report.append("")
    
    # نتایج نهایی
    report.append("🎯 نتایج نهایی:")
    report.append(f"  - RMSE روی داده تست: {results['rmse']:.2f}")
    report.append(f"  - MAE روی داده تست: {results.get('mae', 0):.2f}")
    report.append(f"  - R² روی داده تست: {results['r2']:.4f}")
    report.append("")
    
    report.append("="*70)
    report.append("✅ پایان گزارش مرحله ۲")
    report.append("="*70)
    
    return "\n".join(report)


if __name__ == "__main__":
    run_stage2()
