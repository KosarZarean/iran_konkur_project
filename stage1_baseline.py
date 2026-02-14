"""
مدل‌های پایه برای داده‌های کنکور ایران
شامل: رگرسیون خطی، درخت تصمیم، جنگل تصادفی، XGBoost، LightGBM، MLP و ...
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# ❌ خط CatBoost حذف شد
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """
    کلاس آموزش و ارزیابی مدل‌های پایه
    شامل ۱۴ مدل مختلف رگرسیون
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, task_type='regression'):
        """
        مقداردهی اولیه با داده‌های تقسیم شده
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.task_type = task_type
        
        self.models = {}
        self.results = []
        self.predictions = {}
        self.training_times = {}
        
        print(f"📊 BaselineModels ایجاد شد")
        print(f"   آموزش: {X_train.shape[0]} نمونه")
        print(f"   اعتبارسنجی: {X_val.shape[0]} نمونه")
        print(f"   آزمایش: {X_test.shape[0]} نمونه")
    
    def define_models(self):
        """
        تعریف ۱۴ مدل پایه مختلف (بدون CatBoost)
        """
        print("\n📋 تعریف مدل‌های پایه...")
        
        self.models = {
            # 1. مدل‌های خطی
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            
            # 2. درخت‌ها
            'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, 
                                                  n_jobs=-1, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                                          max_depth=5, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            
            # 3. مدل‌های مبتنی بر نزدیک‌ترین همسایه
            'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1),
            
            # 4. ماشین بردار پشتیبان
            'SVR (linear)': SVR(kernel='linear', C=1.0),
            'SVR (rbf)': SVR(kernel='rbf', C=100, gamma=0.1),
            
            # 5. مدل‌های پیشرفته
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
            'LightGBM': LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                     num_leaves=31, random_state=42, verbose=-1),
            # ❌ CatBoost حذف شد
            
            # 6. شبکه عصبی
            'MLP': MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                               solver='adam', max_iter=500, random_state=42)
        }
        
        print(f"✅ {len(self.models)} مدل تعریف شد")
        
        # نمایش لیست مدل‌ها
        for i, (name, _) in enumerate(self.models.items(), 1):
            print(f"   {i:2d}. {name}")
    
    def train_and_evaluate(self, verbose=True):
        """آموزش و ارزیابی همه مدل‌ها"""
        print("\n" + "="*80)
        print("🚀 شروع آموزش مدل‌های پایه")
        print("="*80)
        
        for name, model in self.models.items():
            if verbose:
                print(f"\n📈 آموزش {name}...")
            
            start_time = time.time()
            
            try:
                # آموزش مدل
                model.fit(self.X_train, self.y_train)
                
                # زمان آموزش
                training_time = time.time() - start_time
                self.training_times[name] = training_time
                
                # پیش‌بینی روی مجموعه‌های مختلف
                y_pred_train = model.predict(self.X_train)
                y_pred_val = model.predict(self.X_val)
                y_pred_test = model.predict(self.X_test)
                
                # محاسبه معیارها
                result = {
                    'Model': name,
                    'Train RMSE': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                    'Val RMSE': np.sqrt(mean_squared_error(self.y_val, y_pred_val)),
                    'Test RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                    'Train MAE': mean_absolute_error(self.y_train, y_pred_train),
                    'Val MAE': mean_absolute_error(self.y_val, y_pred_val),
                    'Test MAE': mean_absolute_error(self.y_test, y_pred_test),
                    'Train R2': r2_score(self.y_train, y_pred_train),
                    'Val R2': r2_score(self.y_val, y_pred_val),
                    'Test R2': r2_score(self.y_test, y_pred_test),
                    'Time (s)': training_time
                }
                
                self.results.append(result)
                self.predictions[name] = y_pred_test
                
                if verbose:
                    print(f"   ✅ Test RMSE: {result['Test RMSE']:.2f}, R²: {result['Test R2']:.4f}, زمان: {training_time:.2f}s")
                
            except Exception as e:
                if verbose:
                    print(f"   ❌ خطا در آموزش {name}: {e}")
        
        # ایجاد DataFrame نتایج
        results_df = pd.DataFrame(self.results)
        
        # مرتب‌سازی بر اساس RMSE
        if not results_df.empty:
            results_df = results_df.sort_values('Test RMSE')
        
        print("\n" + "="*80)
        print("✅ آموزش همه مدل‌ها کامل شد")
        print("="*80)
        
        return results_df
    
    def get_best_model(self, metric='Test RMSE'):
        """دریافت بهترین مدل بر اساس معیار مشخص"""
        if not self.results:
            return None, None
        
        df = pd.DataFrame(self.results)
        
        if metric == 'Test R2':
            best_idx = df[metric].argmax()
            best_value = df[metric].max()
        else:
            best_idx = df[metric].argmin()
            best_value = df[metric].min()
        
        best_model = df.iloc[best_idx]['Model']
        
        return best_model, best_value
    
    def plot_comparison(self, metric='Test RMSE', save_path='plots/baseline_comparison.jpg'):
        """رسم نمودار مقایسه مدل‌ها"""
        if not self.results:
            print("❌ هیچ نتیجه‌ای برای رسم وجود ندارد")
            return
        
        df = pd.DataFrame(self.results)
        df = df.sort_values(metric)
        
        # انتخاب ۱۰ مدل برتر
        top_models = df.head(10)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. RMSE مقایسه
        axes[0, 0].barh(top_models['Model'], top_models['Test RMSE'], 
                       color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('RMSE (lower is better)')
        axes[0, 0].set_title('Top 10 Models - RMSE Comparison')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # 2. R² مقایسه
        axes[0, 1].barh(top_models['Model'], top_models['Test R2'], 
                       color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('R² (higher is better)')
        axes[0, 1].set_title('Top 10 Models - R² Comparison')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. زمان آموزش
        axes[1, 0].barh(top_models['Model'], top_models['Time (s)'], 
                       color='salmon', edgecolor='black')
        axes[1, 0].set_xlabel('Training Time (seconds)')
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Train vs Test RMSE
        x = np.arange(len(top_models))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, top_models['Train RMSE'], width, 
                      label='Train RMSE', color='skyblue', edgecolor='black')
        axes[1, 1].bar(x + width/2, top_models['Test RMSE'], width,
                      label='Test RMSE', color='lightcoral', edgecolor='black')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Train vs Test RMSE')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(top_models['Model'], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Baseline Models Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 نمودار در {save_path} ذخیره شد")
    
    def generate_report(self, save_path='reports/baseline_report.txt'):
        """ایجاد گزارش کامل از نتایج"""
        if not self.results:
            print("❌ هیچ نتیجه‌ای برای گزارش وجود ندارد")
            return
        
        df = pd.DataFrame(self.results)
        best_model, best_rmse = self.get_best_model('Test RMSE')
        best_r2_model, best_r2 = self.get_best_model('Test R2')
        
        report = []
        report.append("="*80)
        report.append("📊 گزارش کامل مدل‌های پایه")
        report.append("="*80)
        report.append(f"تاریخ: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # اطلاعات داده
        report.append("📋 اطلاعات داده:")
        report.append(f"   تعداد نمونه‌های آموزش: {len(self.X_train)}")
        report.append(f"   تعداد نمونه‌های اعتبارسنجی: {len(self.X_val)}")
        report.append(f"   تعداد نمونه‌های آزمایش: {len(self.X_test)}")
        report.append(f"   تعداد ویژگی‌ها: {self.X_train.shape[1]}")
        report.append("")
        
        # بهترین مدل‌ها
        report.append("🏆 بهترین مدل‌ها:")
        report.append(f"   بهترین مدل بر اساس RMSE: {best_model} (RMSE={best_rmse:.2f})")
        report.append(f"   بهترین مدل بر اساس R²: {best_r2_model} (R²={best_r2:.4f})")
        report.append("")
        
        # ۱۰ مدل برتر
        report.append("📊 ۱۰ مدل برتر (بر اساس RMSE):")
        report.append("-" * 80)
        top10 = df.nsmallest(10, 'Test RMSE')[['Model', 'Test RMSE', 'Test R2', 'Test MAE', 'Time (s)']]
        
        for _, row in top10.iterrows():
            report.append(f"   {row['Model']:25s} | RMSE={row['Test RMSE']:8.2f} | R²={row['Test R2']:.4f} | MAE={row['Test MAE']:7.2f} | زمان={row['Time (s)']:.2f}s")
        
        report.append("")
        report.append("="*80)
        report.append("✅ پایان گزارش")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # ذخیره گزارش
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📝 گزارش در {save_path} ذخیره شد")
        
        return report_text
    
    def save_results(self, path='results/baseline_results.csv'):
        """ذخیره نتایج در فایل CSV"""
        if not self.results:
            print("❌ هیچ نتیجه‌ای برای ذخیره وجود ندارد")
            return
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('Test RMSE')
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        
        print(f"💾 نتایج در {path} ذخیره شد")
        
        return df
