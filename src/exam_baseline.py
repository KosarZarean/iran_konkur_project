"""
مدل‌های پایه برای داده‌های کنکور ایران
شامل ۳ مدل اصلی: MLP، Random Forest و Gradient Boosting
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """
    کلاس آموزش و ارزیابی مدل‌های پایه
    شامل ۳ مدل: MLP، Random Forest، Gradient Boosting
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, task_type='regression'):
        """
        مقداردهی اولیه با داده‌های تقسیم شده
        
        پارامترها:
        -----------
        X_train : array
            ویژگی‌های آموزش
        y_train : array
            برچسب‌های آموزش
        X_val : array
            ویژگی‌های اعتبارسنجی
        y_val : array
            برچسب‌های اعتبارسنجی
        X_test : array
            ویژگی‌های آزمایش
        y_test : array
            برچسب‌های آزمایش
        task_type : str
            نوع وظیفه (فعلاً فقط regression)
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
        تعریف ۳ مدل پایه
        """
        print("\n📋 تعریف مدل‌های پایه...")
        
        self.models = {
            'MLP': MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
        }
        
        print(f"✅ {len(self.models)} مدل تعریف شد:")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def train_and_evaluate(self, verbose=True):
        """
        آموزش و ارزیابی همه مدل‌ها
        
        پارامترها:
        -----------
        verbose : bool
            نمایش جزئیات آموزش
        
        Returns:
        --------
        pd.DataFrame
            نتایج همه مدل‌ها
        """
        print("\n" + "="*60)
        print("🚀 شروع آموزش مدل‌های پایه")
        print("="*60)
        
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
                    print(f"   ❌ خطا: {e}")
        
        # ایجاد DataFrame نتایج
        results_df = pd.DataFrame(self.results)
        
        # مرتب‌سازی بر اساس RMSE
        results_df = results_df.sort_values('Test RMSE')
        
        print("\n" + "="*60)
        print("✅ آموزش همه مدل‌ها کامل شد")
        print("="*60)
        
        return results_df
    
    def get_best_model(self, metric='Test RMSE'):
        """
        دریافت بهترین مدل بر اساس معیار مشخص
        
        پارامترها:
        -----------
        metric : str
            معیار ارزیابی ('Test RMSE', 'Test R2')
        
        Returns:
        --------
        tuple
            (نام بهترین مدل, بهترین مقدار)
        """
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
    
    def plot_comparison(self, save_path='plots/baseline_comparison.jpg'):
        """
        رسم نمودار مقایسه مدل‌ها
        
        پارامترها:
        -----------
        save_path : str
            مسیر ذخیره نمودار
        """
        if not self.results:
            print("❌ هیچ نتیجه‌ای برای رسم وجود ندارد")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        # 1. RMSE Comparison
        axes[0, 0].bar(df['Model'], df['Test RMSE'], color=colors, edgecolor='black')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('Test RMSE Comparison (lower is better)')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. MAE Comparison
        axes[0, 1].bar(df['Model'], df['Test MAE'], color=colors, edgecolor='black')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Test MAE Comparison (lower is better)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. R² Comparison
        axes[0, 2].bar(df['Model'], df['Test R2'], color=colors, edgecolor='black')
        axes[0, 2].set_xlabel('Model')
        axes[0, 2].set_ylabel('R²')
        axes[0, 2].set_title('Test R² Comparison (higher is better)')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Training Time
        axes[1, 0].bar(df['Model'], df['Time (s)'], color=colors, edgecolor='black')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. Train vs Test RMSE
        x = np.arange(len(df))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, df['Train RMSE'], width, label='Train', color='skyblue', edgecolor='black')
        axes[1, 1].bar(x + width/2, df['Test RMSE'], width, label='Test', color='salmon', edgecolor='black')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Train vs Test RMSE')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(df['Model'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Model Performance Summary
        axes[1, 2].axis('off')
        
        # اضافه کردن متن خلاصه
        best_model, best_rmse = self.get_best_model('Test RMSE')
        best_r2_model, best_r2 = self.get_best_model('Test R2')
        
        summary_text = f"📊 خلاصه نتایج:\n\n"
        summary_text += f"بهترین RMSE: {best_model}\n"
        summary_text += f"   RMSE = {best_rmse:.2f}\n\n"
        summary_text += f"بهترین R²: {best_r2_model}\n"
        summary_text += f"   R² = {best_r2:.4f}\n\n"
        
        for _, row in df.iterrows():
            summary_text += f"{row['Model']}:\n"
            summary_text += f"   RMSE={row['Test RMSE']:.2f}, R²={row['Test R2']:.3f}\n"
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Baseline Models Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 نمودار در {save_path} ذخیره شد")
    
    def plot_predictions(self, model_name, save_path=None):
        """
        رسم نمودار پیش‌بینی برای یک مدل خاص
        
        پارامترها:
        -----------
        model_name : str
            نام مدل
        save_path : str
            مسیر ذخیره نمودار
        """
        if model_name not in self.predictions:
            print(f"❌ مدل {model_name} یافت نشد")
            return
        
        y_pred = self.predictions[model_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(self.y_test, y_pred, alpha=0.5, s=10)
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = self.y_test - y_pred
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name}: Residual Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, save_path='reports/baseline_report.txt'):
        """
        ایجاد گزارش کامل از نتایج
        
        پارامترها:
        -----------
        save_path : str
            مسیر ذخیره گزارش
        """
        if not self.results:
            print("❌ هیچ نتیجه‌ای برای گزارش وجود ندارد")
            return
        
        df = pd.DataFrame(self.results)
        best_model, best_rmse = self.get_best_model('Test RMSE')
        best_r2_model, best_r2 = self.get_best_model('Test R2')
        
        report = []
        report.append("="*70)
        report.append("📊 گزارش کامل مدل‌های پایه")
        report.append("="*70)
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
        
        # جدول نتایج
        report.append("📊 نتایج کامل:")
        report.append("-" * 80)
        for _, row in df.iterrows():
            report.append(f"   {row['Model']:15s} | RMSE={row['Test RMSE']:8.2f} | R²={row['Test R2']:.4f} | MAE={row['Test MAE']:7.2f} | زمان={row['Time (s)']:.2f}s")
        
        report.append("")
        report.append("="*70)
        report.append("✅ پایان گزارش")
        report.append("="*70)
        
        report_text = "\n".join(report)
        
        # ذخیره گزارش
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📝 گزارش در {save_path} ذخیره شد")
        
        return report_text
    
    def save_results(self, path='results/baseline_results.csv'):
        """
        ذخیره نتایج در فایل CSV
        
        پارامترها:
        -----------
        path : str
            مسیر ذخیره
        """
        if not self.results:
            print("❌ هیچ نتیجه‌ای برای ذخیره وجود ندارد")
            return
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('Test RMSE')
        
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        
        print(f"💾 نتایج در {path} ذخیره شد")
        
        return df


# ============================================
# تابع کمکی برای اجرای سریع
# ============================================

def run_baseline_quick(X_train, y_train, X_test, y_test):
    """
    اجرای سریع مدل‌های پایه
    
    پارامترها:
    -----------
    X_train, y_train : array
        داده آموزش
    X_test, y_test : array
        داده آزمایش
    
    Returns:
    --------
    tuple
        (نتایج, بهترین مدل, بهترین امتیاز)
    """
    baseline = BaselineModels(X_train, y_train, X_test, y_test, X_test, y_test)
    baseline.define_models()
    results = baseline.train_and_evaluate(verbose=False)
    best_model, best_score = baseline.get_best_model()
    
    return results, best_model, best_score


if __name__ == "__main__":
    # تست سریع
    print("🧪 تست کلاس BaselineModels")
    
    # ایجاد داده نمونه
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    # تقسیم داده
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # اجرا
    baseline = BaselineModels(X_train, y_train, X_val, y_val, X_test, y_test)
    baseline.define_models()
    results = baseline.train_and_evaluate()
    baseline.plot_comparison()
    baseline.generate_report()
