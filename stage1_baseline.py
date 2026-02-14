#!/usr/bin/env python3
"""
مرحله ۱: مدل‌های پایه
بر اساس کدهای شما (MLP, Random Forest, Gradient Boosting)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append('src')
from exam_data_manager import ExamDataManager


class BaselineModels:
    """
    کلاس مدل‌های پایه - منطبق با کد شما
    شامل: MLP, Random Forest, Gradient Boosting
    """
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.models = {}
        self.results = []
        self.prediction_plots = {}
    
    def define_models(self):
        """تعریف مدل‌ها - دقیقاً مطابق کد شما"""
        print("\n📋 تعریف مدل‌های پایه...")
        
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        self.models = {
            'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        print(f"✅ {len(self.models)} مدل تعریف شد")
        return self.models
    
    def train_and_evaluate(self):
        """آموزش و ارزیابی - دقیقاً مطابق کد شما"""
        print("\n" + "="*60)
        print("🚀 شروع آموزش مدل‌های پایه")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n📈 آموزش {name}...")
            
            # آموزش
            model.fit(self.X_train, self.y_train)
            
            # پیش‌بینی
            y_pred = model.predict(self.X_test)
            
            # محاسبه معیارها
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            
            print(f"  ✅ {name} RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
            # ذخیره نتایج
            self.results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae
            })
            
            # ذخیره پیش‌بینی برای رسم
            self.prediction_plots[name] = y_pred
        
        # ایجاد DataFrame نتایج
        results_df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("✅ آموزش همه مدل‌ها کامل شد")
        print("="*60)
        
        return results_df
    
    def plot_predictions(self, save_dir='plots/stage1'):
        """رسم نمودارهای پیش‌بینی - دقیقاً مطابق کد شما"""
        os.makedirs(save_dir, exist_ok=True)
        
        # رسم scatter plot برای هر مدل
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
        
        for i, (name, y_pred) in enumerate(self.prediction_plots.items()):
            sns.scatterplot(x=self.y_test, y=y_pred, ax=axes[i], alpha=0.5)
            axes[i].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--')
            axes[i].set_title(f'{name}: Actual vs Predicted')
            axes[i].set_xlabel('Actual Rank')
            axes[i].set_ylabel('Predicted Rank')
        
        plt.suptitle('Actual vs Predicted Ranks for Baseline Models')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/predictions_comparison.jpg', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rmse_comparison(self, save_dir='plots/stage1'):
        """رسم نمودار مقایسه RMSE - دقیقاً مطابق کد شما"""
        os.makedirs(save_dir, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        
        plt.figure(figsize=(6, 4))
        sns.barplot(x='Model', y='RMSE', data=df, palette='Oranges', edgecolor='black')
        plt.title('Baseline Regression Models - RMSE Comparison')
        plt.ylabel('RMSE')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/rmse_comparison.jpg', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_mae_comparison(self, save_dir='plots/stage1'):
        """رسم نمودار مقایسه MAE - دقیقاً مطابق کد شما"""
        os.makedirs(save_dir, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        
        plt.figure(figsize=(6, 4))
        sns.barplot(x='Model', y='MAE', data=df, palette='Blues', edgecolor='black')
        plt.title('Baseline Regression Models - MAE Comparison')
        plt.ylabel('MAE')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/mae_comparison.jpg', dpi=300, bbox_inches='tight')
        plt.show()


def run_stage1(data_path='data/iran_exam.csv'):
    """اجرای مرحله ۱"""
    print("\n" + "="*70)
    print("🎯 مرحله ۱: مدل‌های پایه")
    print("="*70)
    
    # ایجاد پوشه‌ها
    os.makedirs('results/stage1', exist_ok=True)
    os.makedirs('plots/stage1', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 1. بارگذاری و پیش‌پردازش داده
    print("\n📊 مرحله ۱-۱: بارگذاری و پیش‌پردازش داده‌ها...")
    data_manager = ExamDataManager()
    data_manager.load_and_prepare_data(data_path, 'regression')
    data_manager.prepare_for_traditional_models()
    data_manager.create_train_val_test_split()
    
    # 2. آموزش مدل‌ها
    print("\n🤖 مرحله ۱-۲: آموزش مدل‌های پایه...")
    baseline = BaselineModels(
        data_manager.X_train, data_manager.y_train,
        data_manager.X_test, data_manager.y_test
    )
    
    baseline.define_models()
    results = baseline.train_and_evaluate()
    
    # 3. رسم نمودارها
    print("\n📈 مرحله ۱-۳: رسم نمودارها...")
    baseline.plot_predictions()
    baseline.plot_rmse_comparison()
    baseline.plot_mae_comparison()
    
    # 4. ذخیره نتایج
    print("\n💾 مرحله ۱-۴: ذخیره نتایج...")
    results.to_csv('results/stage1/baseline_results.csv', index=False, encoding='utf-8-sig')
    
    # 5. گزارش
    print("\n📝 مرحله ۱-۵: ایجاد گزارش...")
    report = generate_report(results)
    
    with open('reports/stage1_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*70)
    print("✅ مرحله ۱ با موفقیت کامل شد!")
    print("="*70)
    print(report)
    
    return results


def generate_report(results):
    """ایجاد گزارش مرحله ۱"""
    report = []
    report.append("="*70)
    report.append("📊 گزارش مرحله ۱: مدل‌های پایه")
    report.append("="*70)
    report.append(f"تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # نتایج
    report.append("📊 نتایج مدل‌ها:")
    for _, row in results.iterrows():
        report.append(f"  {row['Model']}: RMSE={row['RMSE']:.2f}, MAE={row['MAE']:.2f}")
    
    report.append("")
    
    # بهترین مدل
    best_idx = results['RMSE'].idxmin()
    best = results.iloc[best_idx]
    report.append(f"🏆 بهترین مدل: {best['Model']}")
    report.append(f"   RMSE: {best['RMSE']:.2f}")
    report.append(f"   MAE: {best['MAE']:.2f}")
    
    report.append("")
    report.append("="*70)
    report.append("✅ پایان گزارش مرحله ۱")
    report.append("="*70)
    
    return "\n".join(report)


if __name__ == "__main__":
    run_stage1()
