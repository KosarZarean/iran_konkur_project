"""
توابع مصورسازی برای داده‌ها و نتایج پروژه کنکور
شامل: نمودارهای توزیع، مقایسه، منحنی ROC، ماتریس درهم‌ریختگی و ...
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# تنظیم استایل
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)


class ExamVisualizer:
    """
    کلاس جامع مصورسازی برای پروژه کنکور
    """
    
    def __init__(self, save_dir='plots'):
        """
        مقداردهی اولیه
        
        پارامترها:
        -----------
        save_dir : str
            پوشه ذخیره نمودارها
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # رنگ‌های ثابت برای نمودارها
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
                       '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        
        print(f"📊 ExamVisualizer ایجاد شد: save_dir={save_dir}")
    
    # ============================================
    # نمودارهای توزیع داده
    # ============================================
    
    def plot_distribution(self, data, column, title=None, bins=50, 
                          kde=True, figsize=(12, 5), save=True):
        """
        رسم توزیع یک ستون عددی
        
        پارامترها:
        -----------
        data : pd.DataFrame
            داده‌ها
        column : str
            نام ستون
        title : str
            عنوان نمودار
        bins : int
            تعداد bins
        kde : bool
            نمایش منحنی KDE
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # هیستوگرام
        sns.histplot(data[column].dropna(), bins=bins, kde=kde, ax=axes[0], 
                    color='skyblue', edgecolor='black')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of {column}')
        axes[0].grid(True, alpha=0.3)
        
        # Boxplot
        sns.boxplot(y=data[column], ax=axes[1], color='lightcoral')
        axes[1].set_ylabel(column)
        axes[1].set_title(f'Boxplot of {column}')
        axes[1].grid(True, alpha=0.3)
        
        if title:
            plt.suptitle(title, fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save:
            filename = f'distribution_{column}.jpg'
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_categorical(self, data, column, top_n=20, figsize=(12, 6), 
                         horizontal=False, save=True):
        """
        رسم توزیع یک ستون دسته‌ای
        
        پارامترها:
        -----------
        data : pd.DataFrame
            داده‌ها
        column : str
            نام ستون
        top_n : int
            تعداد دسته‌های برتر
        figsize : tuple
            اندازه شکل
        horizontal : bool
            نمودار افقی
        save : bool
            ذخیره نمودار
        """
        # محاسبه فراوانی
        value_counts = data[column].value_counts()
        
        if len(value_counts) > top_n:
            value_counts = value_counts.head(top_n)
            title = f'Top {top_n} Categories in {column}'
        else:
            title = f'Distribution of {column}'
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # نمودار میله‌ای
        if horizontal:
            axes[0].barh(range(len(value_counts)), value_counts.values, 
                        color=self.colors, edgecolor='black')
            axes[0].set_yticks(range(len(value_counts)))
            axes[0].set_yticklabels(value_counts.index)
            axes[0].set_xlabel('Count')
        else:
            axes[0].bar(range(len(value_counts)), value_counts.values, 
                       color=self.colors, edgecolor='black')
            axes[0].set_xticks(range(len(value_counts)))
            axes[0].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[0].set_ylabel('Count')
        
        axes[0].set_title(title)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # نمودار پای
        axes[1].pie(value_counts.values, labels=value_counts.index, 
                   autopct='%1.1f%%', colors=self.colors[:len(value_counts)])
        axes[1].set_title(f'Pie Chart of {column}')
        
        plt.suptitle(f'Categorical Analysis - {column}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save:
            filename = f'categorical_{column}.jpg'
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return value_counts
    
    def plot_correlation_matrix(self, data, columns=None, figsize=(12, 10), 
                                annot=True, cmap='coolwarm', save=True):
        """
        رسم ماتریس همبستگی
        
        پارامترها:
        -----------
        data : pd.DataFrame
            داده‌ها
        columns : list
            لیست ستون‌ها
        figsize : tuple
            اندازه شکل
        annot : bool
            نمایش مقادیر
        cmap : str
            نقشه رنگ
        save : bool
            ذخیره نمودار
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) < 2:
            print("⚠️ حداقل به 2 ستون عددی نیاز است")
            return
        
        # محاسبه همبستگی
        corr_matrix = data[columns].corr()
        
        # ایجاد mask برای مثلث بالا
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=figsize)
        
        # رسم heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f',
                   cmap=cmap, center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix of Numerical Features', fontsize=14, pad=20)
        plt.tight_layout()
        
        if save:
            filename = 'correlation_matrix.jpg'
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return corr_matrix
    
    def plot_missing_values(self, data, figsize=(12, 6), save=True):
        """
        رسم مقادیر گمشده
        
        پارامترها:
        -----------
        data : pd.DataFrame
            داده‌ها
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        """
        missing = data.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) == 0:
            print("✅ هیچ مقدار گمشده‌ای وجود ندارد")
            return
        
        missing_percent = (missing / len(data)) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # نمودار تعداد
        axes[0].barh(range(len(missing)), missing.values, 
                    color='salmon', edgecolor='black')
        axes[0].set_yticks(range(len(missing)))
        axes[0].set_yticklabels(missing.index)
        axes[0].set_xlabel('Number of Missing Values')
        axes[0].set_title('Missing Values Count')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # نمودار درصد
        axes[1].barh(range(len(missing_percent)), missing_percent.values, 
                    color='skyblue', edgecolor='black')
        axes[1].set_yticks(range(len(missing_percent)))
        axes[1].set_yticklabels(missing_percent.index)
        axes[1].set_xlabel('Percentage of Missing Values (%)')
        axes[1].set_title('Missing Values Percentage')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Missing Values Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save:
            filename = 'missing_values.jpg'
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    # ============================================
    # نمودارهای ارزیابی مدل
    # ============================================
    
    def plot_predictions(self, y_true, y_pred, title='Predictions vs Actual', 
                         figsize=(10, 8), save=True, filename='predictions.jpg'):
        """
        رسم نمودار پیش‌بینی‌ها
        
        پارامترها:
        -----------
        y_true : array
            مقادیر واقعی
        y_pred : array
            مقادیر پیش‌بینی
        title : str
            عنوان نمودار
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        plt.figure(figsize=figsize)
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
        
        # خط ایده‌آل
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
        
        # خط رگرسیون
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        plt.plot([min_val, max_val], p([min_val, max_val]), 'b-', lw=2, label='Regression Line')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # اضافه کردن معیارها
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}'
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, title='Residual Analysis', 
                       figsize=(15, 5), save=True, filename='residuals.jpg'):
        """
        رسم تحلیل باقیمانده‌ها
        
        پارامترها:
        -----------
        y_true : array
            مقادیر واقعی
        y_pred : array
            مقادیر پیش‌بینی
        title : str
            عنوان نمودار
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color='steelblue')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Histogram of residuals
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return residuals
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                              title='Confusion Matrix', figsize=(10, 8), 
                              normalize=False, save=True, filename='confusion_matrix.jpg'):
        """
        رسم ماتریس درهم‌ریختگی
        
        پارامترها:
        -----------
        y_true : array
            برچسب‌های واقعی
        y_pred : array
            برچسب‌های پیش‌بینی
        class_names : list
            نام کلاس‌ها
        title : str
            عنوان نمودار
        figsize : tuple
            اندازه شکل
        normalize : bool
            نرمال‌سازی
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = title + ' (Normalized)'
        else:
            fmt = 'd'
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        plt.figure(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, n_classes=2, class_names=None,
                       title='ROC Curve', figsize=(10, 8), save=True, 
                       filename='roc_curve.jpg'):
        """
        رسم منحنی ROC
        
        پارامترها:
        -----------
        y_true : array
            برچسب‌های واقعی
        y_pred_proba : array
            احتمالات پیش‌بینی
        n_classes : int
            تعداد کلاس‌ها
        class_names : list
            نام کلاس‌ها
        title : str
            عنوان نمودار
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        plt.figure(figsize=figsize)
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random Classifier')
            
        else:
            # Multiclass classification
            from sklearn.preprocessing import label_binarize
            
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    # ============================================
    # نمودارهای مقایسه مدل
    # ============================================
    
    def plot_model_comparison(self, results_df, metric='RMSE', 
                              title='Model Comparison', figsize=(12, 6),
                              sort=True, save=True, filename='model_comparison.jpg'):
        """
        رسم مقایسه مدل‌ها
        
        پارامترها:
        -----------
        results_df : pd.DataFrame
            نتایج مدل‌ها (باید ستون‌های Model و metric داشته باشد)
        metric : str
            نام معیار
        title : str
            عنوان نمودار
        figsize : tuple
            اندازه شکل
        sort : bool
            مرتب‌سازی
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        df = results_df.copy()
        
        if sort:
            ascending = False if metric in ['R2', 'Accuracy', 'F1'] else True
            df = df.sort_values(metric, ascending=ascending)
        
        plt.figure(figsize=figsize)
        
        # انتخاب رنگ بر اساس مقدار
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(df)))
        
        bars = plt.bar(df['Model'], df[metric], color=colors, edgecolor='black', alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.title(f'{title} - {metric}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # اضافه کردن مقادیر روی میله‌ها
        for bar, value in zip(bars, df[metric]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}' if value < 10 else f'{value:.1f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_training_history(self, history, metrics=['loss', 'rmse', 'mae', 'r2'],
                              figsize=(15, 10), save=True, filename='training_history.jpg'):
        """
        رسم تاریخچه آموزش
        
        پارامترها:
        -----------
        history : dict
            تاریخچه آموزش
        metrics : list
            لیست معیارها
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        for idx, metric in enumerate(metrics):
            if idx < len(axes):
                train_metric = history.get(f'train_{metric}', [])
                val_metric = history.get(f'val_{metric}', [])
                
                if train_metric:
                    axes[idx].plot(epochs, train_metric, 'b-', label=f'Train {metric.upper()}', linewidth=2)
                
                if val_metric:
                    axes[idx].plot(epochs, val_metric, 'r-', label=f'Val {metric.upper()}', linewidth=2)
                
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric.upper())
                axes[idx].set_title(f'Training and Validation {metric.upper()}')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        # حذف زیرنمودارهای اضافی
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Training History', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_feature_importance(self, importance_dict, title='Feature Importance',
                                top_n=20, figsize=(10, 8), save=True, 
                                filename='feature_importance.jpg'):
        """
        رسم اهمیت ویژگی‌ها
        
        پارامترها:
        -----------
        importance_dict : dict
            دیکشنری اهمیت ویژگی‌ها
        title : str
            عنوان نمودار
        top_n : int
            تعداد ویژگی‌های برتر
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        # مرتب‌سازی
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_items) > top_n:
            sorted_items = sorted_items[:top_n]
            plot_title = f'{title} (Top {top_n})'
        else:
            plot_title = title
        
        features = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
        plt.figure(figsize=figsize)
        
        # نمودار افقی
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        plt.barh(range(len(features)), scores, color=colors, edgecolor='black')
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(plot_title)
        plt.grid(True, alpha=0.3, axis='x')
        
        # اضافه کردن مقادیر
        for i, (feature, score) in enumerate(zip(features, scores)):
            plt.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    # ============================================
    # نمودارهای پیشرفته
    # ============================================
    
    def plot_tsne(self, X, y, title='t-SNE Visualization', perplexity=30,
                  figsize=(12, 10), save=True, filename='tsne.jpg'):
        """
        رسم نمودار t-SNE
        
        پارامترها:
        -----------
        X : array
            ویژگی‌ها
        y : array
            برچسب‌ها
        title : str
            عنوان نمودار
        perplexity : int
            پارامتر perplexity
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        print("🔄 در حال محاسبه t-SNE...")
        
        # کاهش ابعاد
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=figsize)
        
        # رسم
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
                            cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Target')
        
        plt.title(f'{title} (perplexity={perplexity})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return X_tsne
    
    def plot_parallel_coordinates(self, data, class_column, columns=None,
                                  title='Parallel Coordinates', figsize=(15, 8),
                                  save=True, filename='parallel_coordinates.jpg'):
        """
        رسم نمودار مختصات موازی
        
        پارامترها:
        -----------
        data : pd.DataFrame
            داده‌ها
        class_column : str
            ستون کلاس
        columns : list
            لیست ستون‌ها
        title : str
            عنوان نمودار
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        from pandas.plotting import parallel_coordinates
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        plt.figure(figsize=figsize)
        
        # نرمال‌سازی داده‌ها برای مقایسه بهتر
        data_normalized = data[columns].copy()
        for col in columns:
            data_normalized[col] = (data_normalized[col] - data_normalized[col].mean()) / data_normalized[col].std()
        
        data_normalized[class_column] = data[class_column]
        
        parallel_coordinates(data_normalized, class_column, color=self.colors, alpha=0.5)
        
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Normalized Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_learning_curve(self, train_sizes, train_scores, val_scores,
                           title='Learning Curve', figsize=(10, 6),
                           save=True, filename='learning_curve.jpg'):
        """
        رسم منحنی یادگیری
        
        پارامترها:
        -----------
        train_sizes : array
            اندازه‌های آموزش
        train_scores : array
            امتیازهای آموزش
        val_scores : array
            امتیازهای اعتبارسنجی
        title : str
            عنوان نمودار
        figsize : tuple
            اندازه شکل
        save : bool
            ذخیره نمودار
        filename : str
            نام فایل
        """
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
        
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 نمودار در {save_path} ذخیره شد")
        
        plt.show()


# ============================================
# تابع کمکی برای ایجاد گزارش تصویری
# ============================================

def create_visual_report(visualizer, data, predictions_dict, results_df, 
                         y_true=None, save_dir='visual_report'):
    """
    ایجاد گزارش تصویری کامل
    
    پارامترها:
    -----------
    visualizer : ExamVisualizer
        شیء مصورساز
    data : pd.DataFrame
        داده‌ها
    predictions_dict : dict
        دیکشنری پیش‌بینی‌ها
    results_df : pd.DataFrame
        نتایج مدل‌ها
    y_true : array
        مقادیر واقعی (اختیاری)
    save_dir : str
        پوشه ذخیره
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. توزیع داده‌ها
    for col in data.select_dtypes(include=[np.number]).columns[:3]:
        visualizer.plot_distribution(data, col, save=True)
    
    # 2. مقایسه مدل‌ها
    visualizer.plot_model_comparison(results_df, save=True)
    
    # 3. پیش‌بینی‌ها
    if y_true is not None:
        for name, y_pred in predictions_dict.items():
            visualizer.plot_predictions(y_true, y_pred, 
                                       title=f'{name} - Predictions',
                                       filename=f'predictions_{name}.jpg')
            visualizer.plot_residuals(y_true, y_pred,
                                     title=f'{name} - Residuals',
                                     filename=f'residuals_{name}.jpg')
    
    print(f"✅ گزارش تصویری در {save_dir} ایجاد شد")


# ============================================
# تست
# ============================================

def test_visualizer():
    """تست کلاس مصورساز"""
    print("🧪 تست ExamVisualizer")
    print("="*60)
    
    # داده نمونه
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000) * 2 + 1,
        'feature3': np.random.randn(1000) * 0.5,
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000)
    })
    
    y_true = np.random.randn(1000)
    y_pred = y_true + np.random.randn(1000) * 0.1
    
    # ایجاد visualizer
    visualizer = ExamVisualizer('test_plots')
    
    # تست نمودارها
    visualizer.plot_distribution(data, 'feature1')
    visualizer.plot_categorical(data, 'category')
    visualizer.plot_predictions(y_true, y_pred)
    visualizer.plot_residuals(y_true, y_pred)
    
    print("\n✅ همه تست‌ها با موفقیت انجام شد")
    print(f"📊 نمودارها در {visualizer.save_dir} ذخیره شدند")


if __name__ == "__main__":
    test_visualizer()
