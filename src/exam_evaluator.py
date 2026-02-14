"""
ارزیابی مدل‌ها برای داده‌های کنکور ایران
شامل: معیارهای ارزیابی، گزارش‌گیری، تحلیل خطا و ...
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score)

from exam_utils import ExamUtils, MetricsCalculator, ExperimentLogger, Timer
from exam_visualization import ExamVisualizer


class ModelEvaluator:
    """
    کلاس ارزیابی مدل‌ها
    """
    
    def __init__(self, save_dir='evaluation'):
        """
        مقداردهی اولیه
        
        پارامترها:
        -----------
        save_dir : str
            پوشه ذخیره نتایج
        """
        self.save_dir = ExamUtils.ensure_dir(save_dir)
        self.results_dir = ExamUtils.ensure_dir(os.path.join(save_dir, 'results'))
        self.plots_dir = ExamUtils.ensure_dir(os.path.join(save_dir, 'plots'))
        
        self.visualizer = ExamVisualizer(self.plots_dir)
        self.logger = ExperimentLogger(os.path.join(save_dir, 'logs'))
        self.results = {}
        
        print(f"📊 ModelEvaluator ایجاد شد: save_dir={save_dir}")
    
    def evaluate_regression(self, y_true, y_pred, model_name='model', 
                            y_train=None, y_pred_train=None):
        """
        ارزیابی مدل رگرسیون
        
        پارامترها:
        -----------
        y_true : array
            مقادیر واقعی
        y_pred : array
            مقادیر پیش‌بینی
        model_name : str
            نام مدل
        y_train : array
            مقادیر واقعی آموزش
        y_pred_train : array
            مقادیر پیش‌بینی آموزش
        
        Returns:
        --------
        dict
            نتایج ارزیابی
        """
        print(f"\n📊 ارزیابی مدل رگرسیون: {model_name}")
        print("="*60)
        
        # محاسبه معیارها
        metrics = MetricsCalculator.regression_metrics(y_true, y_pred)
        
        # معیارهای آموزش (اختیاری)
        if y_train is not None and y_pred_train is not None:
            train_metrics = MetricsCalculator.regression_metrics(y_train, y_pred_train)
            for key, value in train_metrics.items():
                metrics[f'Train {key}'] = value
        
        # نمایش نتایج
        print("\n📈 نتایج ارزیابی:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        # رسم نمودارها
        self.visualizer.plot_predictions(y_true, y_pred, 
                                        title=f'{model_name} - Predictions',
                                        filename=f'{model_name}_predictions.jpg')
        
        self.visualizer.plot_residuals(y_true, y_pred,
                                      title=f'{model_name} - Residuals',
                                      filename=f'{model_name}_residuals.jpg')
        
        # ذخیره نتایج
        self.results[model_name] = {
            'type': 'regression',
            'metrics': metrics,
            'predictions': {
                'y_true': y_true.tolist() if hasattr(y_true, 'tolist') else y_true,
                'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
            }
        }
        
        # ثبت در لاگ
        self.logger.log_results(metrics, f"{model_name} - Regression")
        
        return metrics
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None,
                               model_name='model', class_names=None,
                               y_train=None, y_pred_train=None):
        """
        ارزیابی مدل طبقه‌بندی
        
        پارامترها:
        -----------
        y_true : array
            برچسب‌های واقعی
        y_pred : array
            برچسب‌های پیش‌بینی
        y_pred_proba : array
            احتمالات پیش‌بینی
        model_name : str
            نام مدل
        class_names : list
            نام کلاس‌ها
        y_train : array
            برچسب‌های واقعی آموزش
        y_pred_train : array
            برچسب‌های پیش‌بینی آموزش
        
        Returns:
        --------
        dict
            نتایج ارزیابی
        """
        print(f"\n📊 ارزیابی مدل طبقه‌بندی: {model_name}")
        print("="*60)
        
        # محاسبه معیارها
        metrics = MetricsCalculator.classification_metrics(y_true, y_pred, y_pred_proba)
        
        # معیارهای آموزش
        if y_train is not None and y_pred_train is not None:
            train_metrics = MetricsCalculator.classification_metrics(y_train, y_pred_train)
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f'Train {key}'] = value
        
        # حذف ماتریس درهم‌ریختگی از metrics برای نمایش
        cm = metrics.pop('Confusion Matrix', None)
        
        # نمایش نتایج
        print("\n📈 نتایج ارزیابی:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        # رسم ماتریس درهم‌ریختگی
        if cm is not None:
            self.visualizer.plot_confusion_matrix(
                y_true, y_pred, class_names=class_names,
                title=f'{model_name} - Confusion Matrix',
                filename=f'{model_name}_confusion_matrix.jpg'
            )
        
        # رسم منحنی ROC
        if y_pred_proba is not None:
            n_classes = len(np.unique(y_true))
            self.visualizer.plot_roc_curve(
                y_true, y_pred_proba, n_classes=n_classes,
                class_names=class_names,
                title=f'{model_name} - ROC Curve',
                filename=f'{model_name}_roc_curve.jpg'
            )
        
        # ذخیره نتایج
        self.results[model_name] = {
            'type': 'classification',
            'metrics': metrics,
            'confusion_matrix': cm.tolist() if cm is not None else None,
            'predictions': {
                'y_true': y_true.tolist() if hasattr(y_true, 'tolist') else y_true,
                'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
            }
        }
        
        if y_pred_proba is not None:
            self.results[model_name]['predictions']['y_pred_proba'] = y_pred_proba.tolist()
        
        # ثبت در لاگ
        self.logger.log_results(metrics, f"{model_name} - Classification")
        
        return metrics
    
    def compare_models(self, results_dict=None, metric='RMSE'):
        """
        مقایسه چند مدل
        
        پارامترها:
        -----------
        results_dict : dict
            دیکشنری نتایج
        metric : str
            معیار مقایسه
        
        Returns:
        --------
        pd.DataFrame
            جدول مقایسه
        """
        if results_dict is None:
            results_dict = self.results
        
        comparison = []
        
        for model_name, result in results_dict.items():
            row = {'Model': model_name}
            
            if result['type'] == 'regression':
                metrics = result['metrics']
                row['RMSE'] = metrics.get('RMSE', 0)
                row['MAE'] = metrics.get('MAE', 0)
                row['R2'] = metrics.get('R2', 0)
                row['MAPE'] = metrics.get('MAPE', 0)
            else:
                metrics = result['metrics']
                row['Accuracy'] = metrics.get('Accuracy', 0)
                row['F1 (macro)'] = metrics.get('F1 (macro)', 0)
                row['ROC-AUC'] = metrics.get('ROC-AUC', metrics.get('ROC-AUC (ovr)', 0))
            
            comparison.append(row)
        
        df_comparison = pd.DataFrame(comparison)
        
        # مرتب‌سازی
        if result['type'] == 'regression':
            df_comparison = df_comparison.sort_values('RMSE')
        else:
            df_comparison = df_comparison.sort_values('Accuracy', ascending=False)
        
        # رسم نمودار مقایسه
        self.visualizer.plot_model_comparison(
            df_comparison, metric=metric,
            title='Model Comparison',
            filename='model_comparison.jpg'
        )
        
        # ذخیره
        comparison_path = os.path.join(self.results_dir, 'model_comparison.csv')
        df_comparison.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"💾 جدول مقایسه در {comparison_path} ذخیره شد")
        
        return df_comparison
    
    def error_analysis(self, y_true, y_pred, model_name='model', 
                      bins=10, save=True):
        """
        تحلیل خطاها
        
        پارامترها:
        -----------
        y_true : array
            مقادیر واقعی
        y_pred : array
            مقادیر پیش‌بینی
        model_name : str
            نام مدل
        bins : int
            تعداد دسته‌ها
        save : bool
            ذخیره نتایج
        
        Returns:
        --------
        dict
            تحلیل خطاها
        """
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        # آمار خطاها
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mean_abs_error': np.mean(abs_errors),
            'median_abs_error': np.median(abs_errors),
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors),
            'q25_error': np.percentile(abs_errors, 25),
            'q75_error': np.percentile(abs_errors, 75)
        }
        
        # تحلیل بر اساس محدوده مقادیر واقعی
        bins_labels = pd.qcut(y_true, q=bins, labels=False, duplicates='drop')
        error_by_bin = {}
        
        for i in range(len(np.unique(bins_labels))):
            mask = bins_labels == i
            if np.sum(mask) > 0:
                bin_true = y_true[mask]
                bin_pred = y_pred[mask]
                bin_errors = errors[mask]
                
                error_by_bin[f'Bin_{i+1}'] = {
                    'range': f"{bin_true.min():.0f}-{bin_true.max():.0f}",
                    'count': np.sum(mask),
                    'mean_error': np.mean(bin_errors),
                    'std_error': np.std(bin_errors),
                    'mae': np.mean(np.abs(bin_errors))
                }
        
        # رسم تحلیل خطا
        if save:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. توزیع خطاها
            axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(x=0, color='r', linestyle='--', lw=2)
            axes[0, 0].set_xlabel('Error')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Error Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. خطا بر اساس مقدار واقعی
            axes[0, 1].scatter(y_true, errors, alpha=0.5, s=10)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[0, 1].set_xlabel('Actual Values')
            axes[0, 1].set_ylabel('Error')
            axes[0, 1].set_title('Error vs Actual')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. خطا بر اساس پیش‌بینی
            axes[1, 0].scatter(y_pred, errors, alpha=0.5, s=10)
            axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[1, 0].set_xlabel('Predicted Values')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].set_title('Error vs Predicted')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. خطای مطلق بر اساس دسته
            bins_list = list(error_by_bin.keys())
            mae_values = [error_by_bin[b]['mae'] for b in bins_list]
            
            axes[1, 1].bar(range(len(bins_list)), mae_values, color='skyblue', edgecolor='black')
            axes[1, 1].set_xlabel('Value Range')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].set_title('MAE by Value Range')
            axes[1, 1].set_xticks(range(len(bins_list)))
            axes[1, 1].set_xticklabels([error_by_bin[b]['range'] for b in bins_list], 
                                       rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle(f'Error Analysis - {model_name}', fontsize=14, y=1.02)
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, f'{model_name}_error_analysis.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        return {
            'error_stats': error_stats,
            'error_by_bin': error_by_bin
        }
    
    def cross_validation_report(self, cv_scores, model_name='model'):
        """
        گزارش اعتبارسنجی متقاطع
        
        پارامترها:
        -----------
        cv_scores : list
            لیست امتیازها
        model_name : str
            نام مدل
        
        Returns:
        --------
        dict
            گزارش
        """
        report = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores),
            'scores': cv_scores
        }
        
        # رسم نمودار
        plt.figure(figsize=(10, 6))
        
        plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'o-', color='steelblue', linewidth=2)
        plt.axhline(y=report['mean_score'], color='r', linestyle='--', 
                   label=f"Mean: {report['mean_score']:.4f}")
        plt.fill_between(range(1, len(cv_scores) + 1),
                        report['mean_score'] - report['std_score'],
                        report['mean_score'] + report['std_score'],
                        alpha=0.2, color='gray', label=f"Std: {report['std_score']:.4f}")
        
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title(f'Cross-Validation Scores - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f'{model_name}_cv_scores.jpg')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return report
    
    def generate_full_report(self, filename='evaluation_report.txt'):
        """
        ایجاد گزارش کامل
        
        پارامترها:
        -----------
        filename : str
            نام فایل گزارش
        """
        report_path = os.path.join(self.results_dir, filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("📊 گزارش کامل ارزیابی مدل‌ها\n")
            f.write("="*80 + "\n\n")
            f.write(f"تاریخ: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"تعداد مدل‌های ارزیابی شده: {len(self.results)}\n\n")
            
            for model_name, result in self.results.items():
                f.write("-"*60 + "\n")
                f.write(f"📌 مدل: {model_name}\n")
                f.write(f"نوع: {result['type']}\n")
                f.write("\nمعیارها:\n")
                
                for key, value in result['metrics'].items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                
                f.write("\n")
            
            f.write("="*80 + "\n")
        
        print(f"📝 گزارش کامل در {report_path} ذخیره شد")
    
    def save_results(self, filename='evaluation_results.json'):
        """
        ذخیره نتایج
        
        پارامترها:
        -----------
        filename : str
            نام فایل
        """
        filepath = os.path.join(self.results_dir, filename)
        ExamUtils.save_json(self.results, filepath)
        print(f"💾 نتایج در {filepath} ذخیره شد")


class EnsembleEvaluator:
    """
    کلاس ارزیابی مدل‌های ensemble
    """
    
    def __init__(self, evaluator):
        """
        مقداردهی اولیه
        
        پارامترها:
        -----------
        evaluator : ModelEvaluator
            شیء ارزیاب
        """
        self.evaluator = evaluator
    
    def evaluate_voting_ensemble(self, predictions_dict, y_true, weights=None):
        """
        ارزیابی ensemble با voting
        
        پارامترها:
        -----------
        predictions_dict : dict
            دیکشنری پیش‌بینی‌ها
        y_true : array
            مقادیر واقعی
        weights : list
            وزن‌ها
        
        Returns:
        --------
        dict
            نتایج
        """
        # میانگین گیری از پیش‌بینی‌ها
        all_preds = np.array(list(predictions_dict.values()))
        
        if weights is not None:
            weights = np.array(weights) / np.sum(weights)
            ensemble_pred = np.average(all_preds, axis=0, weights=weights)
        else:
            ensemble_pred = np.mean(all_preds, axis=0)
        
        # ارزیابی
        metrics = self.evaluator.evaluate_regression(
            y_true, ensemble_pred, model_name='Voting Ensemble'
        )
        
        return metrics
    
    def evaluate_stacking_ensemble(self, base_models, meta_model, 
                                   X_train, y_train, X_test, y_test):
        """
        ارزیابی stacking ensemble
        
        پارامترها:
        -----------
        base_models : list
            لیست مدل‌های پایه
        meta_model : any
            مدل فرا
        X_train, y_train : array
            داده آموزش
        X_test, y_test : array
            داده آزمایش
        
        Returns:
        --------
        dict
            نتایج
        """
        from sklearn.ensemble import StackingRegressor
        
        # ایجاد stacking ensemble
        estimators = [(f'model_{i}', model) for i, model in enumerate(base_models)]
        stack = StackingRegressor(estimators=estimators, final_estimator=meta_model)
        
        # آموزش
        stack.fit(X_train, y_train)
        
        # پیش‌بینی
        y_pred = stack.predict(X_test)
        
        # ارزیابی
        metrics = self.evaluator.evaluate_regression(
            y_test, y_pred, model_name='Stacking Ensemble'
        )
        
        return metrics, stack


# ============================================
# تست
# ============================================

def test_evaluator():
    """تست کلاس ارزیاب"""
    print("🧪 تست ModelEvaluator")
    print("="*60)
    
    # داده نمونه
    np.random.seed(42)
    y_true = np.random.randn(1000)
    y_pred1 = y_true + np.random.randn(1000) * 0.1
    y_pred2 = y_true + np.random.randn(1000) * 0.2
    y_pred3 = y_true + np.random.randn(1000) * 0.15
    
    # ایجاد evaluator
    evaluator = ModelEvaluator('test_eval')
    
    # ارزیابی مدل‌ها
    evaluator.evaluate_regression(y_true, y_pred1, model_name='Model_1')
    evaluator.evaluate_regression(y_true, y_pred2, model_name='Model_2')
    evaluator.evaluate_regression(y_true, y_pred3, model_name='Model_3')
    
    # مقایسه
    comparison = evaluator.compare_models()
    print("\n📊 مقایسه مدل‌ها:")
    print(comparison)
    
    # تحلیل خطا
    evaluator.error_analysis(y_true, y_pred1, model_name='Model_1')
    
    # ذخیره نتایج
    evaluator.save_results()
    evaluator.generate_full_report()
    
    print("\n✅ همه تست‌ها با موفقیت انجام شد")


if __name__ == "__main__":
    test_evaluator()
