"""
مصورسازی نتایج
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ExamVisualizer:
    """کلاس مصورسازی"""
    
    def __init__(self, save_dir='plots'):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_predictions(self, y_true, y_pred, title='Predictions', save_name='predictions.jpg'):
        """رسم پیش‌بینی‌ها"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'{self.save_dir}/{save_name}', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, title='Residuals', save_name='residuals.jpg'):
        """رسم باقیمانده‌ها"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(y_pred, residuals, alpha=0.5, s=10)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}', dpi=300)
        plt.show()
    
    def plot_comparison(self, models, scores, title='Comparison', save_name='comparison.jpg'):
        """رسم مقایسه"""
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = plt.bar(models, scores, color=colors, edgecolor='black')
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}', dpi=300)
        plt.show()
