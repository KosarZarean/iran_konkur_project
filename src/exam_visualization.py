"""
توابع مصورسازی برای داده‌ها و نتایج پروژه کنکور
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_squared_error, r2_score


class ExamVisualizer:
    """
    کلاس مصورسازی برای پروژه کنکور
    """
    
    def __init__(self, save_dir='plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    def plot_predictions(self, y_true, y_pred, title='Predictions vs Actual', 
                         save_name='predictions.jpg'):
        """رسم نمودار پیش‌بینی‌ها"""
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{title}\nRMSE={rmse:.2f}, R²={r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 نمودار در {save_path} ذخیره شد")
    
    def plot_residuals(self, y_true, y_pred, title='Residual Analysis', 
                       save_name='residuals.jpg'):
        """رسم تحلیل باقیمانده‌ها"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=10, color='steelblue')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 نمودار در {save_path} ذخیره شد")
        
        return residuals
    
    def plot_model_comparison(self, results_df, title='Model Comparison',
                              save_name='model_comparison.jpg'):
        """رسم مقایسه مدل‌ها"""
        plt.figure(figsize=(12, 6))
        
        x = range(len(results_df))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # RMSE
        color = 'tab:blue'
        ax1.set_xlabel('Model')
        ax1.set_ylabel('RMSE (lower is better)', color=color)
        bars1 = ax1.bar(x, results_df['RMSE'], width, label='RMSE', 
                        color='skyblue', edgecolor='black')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # R²
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('R² (higher is better)', color=color)
        bars2 = ax2.bar([i + width for i in x], results_df['R2'], width, 
                        label='R²', color='lightcoral', edgecolor='black')
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax1.set_xticks([i + width/2 for i in x])
        ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        
        # اضافه کردن مقادیر
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(title)
        fig.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 نمودار در {save_path} ذخیره شد")
    
    def plot_training_history(self, history, save_name='training_history.jpg'):
        """رسم تاریخچه آموزش"""
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RMSE
        axes[1].plot(epochs, history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
        axes[1].plot(epochs, history['val_rmse'], 'r-', label='Val RMSE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Training and Validation RMSE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 نمودار در {save_path} ذخیره شد")
    
    def plot_comparison_bar(self, df, x_col, y_col, title, xlabel, ylabel,
                           color='skyblue', save_name='bar_chart.jpg'):
        """رسم نمودار میله‌ای ساده"""
        plt.figure(figsize=(10, 6))
        
        plt.bar(df[x_col], df[y_col], color=color, edgecolor='black')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        for i, val in enumerate(df[y_col]):
            plt.text(i, val + 0.01, f'{val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 نمودار در {save_path} ذخیره شد")
