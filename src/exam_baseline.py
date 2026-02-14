"""
مدل‌های پایه برای داده‌های کنکور ایران
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class BaselineModels:
    """
    کلاس آموزش و ارزیابی مدل‌های پایه
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, task_type='regression'):
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
    
    def define_models(self):
        """تعریف مدل‌های پایه"""
        print(f"\n📋 تعریف مدل‌های پایه...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
            'LightGBM': LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1),
            'MLP': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
        }
        
        print(f"✅ {len(self.models)} مدل تعریف شد")
    
    def train_and_evaluate(self):
        """آموزش و ارزیابی همه مدل‌ها"""
        print("\n🚀 شروع آموزش مدل‌های پایه...")
        
        for name, model in self.models.items():
            print(f"  📈 آموزش {name}...", end='')
            
            start_time = time.time()
            
            try:
                model.fit(self.X_train, self.y_train)
                
                y_pred_train = model.predict(self.X_train)
                y_pred_val = model.predict(self.X_val)
                y_pred_test = model.predict(self.X_test)
                
                training_time = time.time() - start_time
                
                result = {
                    'Model': name,
                    'Train RMSE': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                    'Val RMSE': np.sqrt(mean_squared_error(self.y_val, y_pred_val)),
                    'Test RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                    'Train R2': r2_score(self.y_train, y_pred_train),
                    'Val R2': r2_score(self.y_val, y_pred_val),
                    'Test R2': r2_score(self.y_test, y_pred_test),
                    'Time (s)': training_time
                }
                
                self.results.append(result)
                self.predictions[name] = y_pred_test
                
                print(f" ✅ Test RMSE: {result['Test RMSE']:.2f}")
                
            except Exception as e:
                print(f" ❌ خطا: {e}")
        
        return pd.DataFrame(self.results)
    
    def plot_comparison(self, save_path='baseline_comparison.jpg'):
        """رسم نمودار مقایسه"""
        df_results = pd.DataFrame(self.results)
        
        plt.figure(figsize=(12, 6))
        
        # ۵ مدل برتر
        top_models = df_results.nsmallest(10, 'Test RMSE')
        
        plt.barh(top_models['Model'], top_models['Test RMSE'], color='skyblue', edgecolor='black')
        plt.xlabel('RMSE (lower is better)')
        plt.title('Top 10 Models Comparison')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_results
