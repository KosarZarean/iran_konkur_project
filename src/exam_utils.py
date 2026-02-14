"""
توابع کمکی برای پروژه کنکور ایران
شامل: ذخیره و بارگذاری، مدیریت فایل‌ها، محاسبات آماری و ...
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
import hashlib
import random


class ExamUtils:
    """
    کلاس توابع کمکی عمومی
    """
    
    @staticmethod
    def set_seed(seed=42):
        """
        تنظیم seed برای تکرارپذیری
        
        پارامترها:
        -----------
        seed : int
            مقدار seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"🎲 Seed تنظیم شد: {seed}")
    
    @staticmethod
    def ensure_dir(directory):
        """
        ایجاد پوشه در صورت عدم وجود
        
        پارامترها:
        -----------
        directory : str
            مسیر پوشه
        
        Returns:
        --------
        str
            مسیر پوشه
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        return directory
    
    @staticmethod
    def get_timestamp():
        """
        دریافت زمان فعلی به صورت رشته
        
        Returns:
        --------
        str
            زمان فعلی
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def save_json(data, filepath):
        """
        ذخیره داده در فایل JSON
        
        پارامترها:
        -----------
        data : dict
            داده
        filepath : str
            مسیر فایل
        """
        ExamUtils.ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON ذخیره شد: {filepath}")
    
    @staticmethod
    def load_json(filepath):
        """
        بارگذاری داده از فایل JSON
        
        پارامترها:
        -----------
        filepath : str
            مسیر فایل
        
        Returns:
        --------
        dict
            داده
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📂 JSON بارگذاری شد: {filepath}")
        return data
    
    @staticmethod
    def save_pickle(data, filepath):
        """
        ذخیره داده با pickle
        
        پارامترها:
        -----------
        data : any
            داده
        filepath : str
            مسیر فایل
        """
        ExamUtils.ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Pickle ذخیره شد: {filepath}")
    
    @staticmethod
    def load_pickle(filepath):
        """
        بارگذاری داده با pickle
        
        پارامترها:
        -----------
        filepath : str
            مسیر فایل
        
        Returns:
        --------
        any
            داده
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"📂 Pickle بارگذاری شد: {filepath}")
        return data
    
    @staticmethod
    def save_model(model, filepath, model_type='sklearn'):
        """
        ذخیره مدل
        
        پارامترها:
        -----------
        model : any
            مدل
        filepath : str
            مسیر فایل
        model_type : str
            نوع مدل ('sklearn', 'torch', 'joblib')
        """
        ExamUtils.ensure_dir(os.path.dirname(filepath))
        
        if model_type == 'torch':
            torch.save(model.state_dict(), filepath)
        elif model_type == 'joblib':
            joblib.dump(model, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"💾 مدل ذخیره شد: {filepath}")
    
    @staticmethod
    def load_model(filepath, model_type='sklearn', model_class=None):
        """
        بارگذاری مدل
        
        پارامترها:
        -----------
        filepath : str
            مسیر فایل
        model_type : str
            نوع مدل
        model_class : class
            کلاس مدل برای torch
        
        Returns:
        --------
        any
            مدل
        """
        if model_type == 'torch':
            if model_class is None:
                raise ValueError("برای بارگذاری مدل torch باید model_class مشخص شود")
            model = model_class()
            model.load_state_dict(torch.load(filepath))
        elif model_type == 'joblib':
            model = joblib.load(filepath)
        else:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        
        print(f"📂 مدل بارگذاری شد: {filepath}")
        return model
    
    @staticmethod
    def get_file_hash(filepath, algorithm='md5'):
        """
        محاسبه هش فایل
        
        پارامترها:
        -----------
        filepath : str
            مسیر فایل
        algorithm : str
            الگوریتم هش
        
        Returns:
        --------
        str
            هش فایل
        """
        hash_func = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    @staticmethod
    def get_file_size(filepath):
        """
        دریافت حجم فایل
        
        پارامترها:
        -----------
        filepath : str
            مسیر فایل
        
        Returns:
        --------
        str
            حجم فایل
        """
        size = os.path.getsize(filepath)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"


class MetricsCalculator:
    """
    کلاس محاسبه معیارهای ارزیابی
    """
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """
        محاسبه معیارهای رگرسیون
        
        پارامترها:
        -----------
        y_true : array
            مقادیر واقعی
        y_pred : array
            مقادیر پیش‌بینی
        
        Returns:
        --------
        dict
            معیارها
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # Explained variance
        explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Explained Variance': explained_var
        }
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None):
        """
        محاسبه معیارهای طبقه‌بندی
        
        پارامترها:
        -----------
        y_true : array
            برچسب‌های واقعی
        y_pred : array
            برچسب‌های پیش‌بینی
        y_pred_proba : array
            احتمالات پیش‌بینی
        
        Returns:
        --------
        dict
            معیارها
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                   f1_score, confusion_matrix, roc_auc_score)
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision (macro)': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall (macro)': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'F1 (macro)': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'Precision (weighted)': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall (weighted)': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 (weighted)': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['Confusion Matrix'] = cm
        
        # ROC-AUC
        if y_pred_proba is not None:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['ROC-AUC (ovr)'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                metrics['ROC-AUC (ovo)'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')
        
        return metrics


class ExperimentLogger:
    """
    کلاس ثبت گزارش آزمایش‌ها
    """
    
    def __init__(self, log_dir='logs'):
        """
        مقداردهی اولیه
        
        پارامترها:
        -----------
        log_dir : str
            پوشه لاگ
        """
        self.log_dir = ExamUtils.ensure_dir(log_dir)
        self.log_file = os.path.join(log_dir, f'experiment_{ExamUtils.get_timestamp()}.log')
        self.results = []
        
        print(f"📝 ExperimentLogger ایجاد شد: {self.log_file}")
    
    def log(self, message, level='INFO'):
        """
        ثبت پیام در لاگ
        
        پارامترها:
        -----------
        message : str
            پیام
        level : str
            سطح لاگ
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def log_config(self, config):
        """
        ثبت پیکربندی
        
        پارامترها:
        -----------
        config : dict
            پیکربندی
        """
        self.log("="*60)
        self.log("پیکربندی آزمایش:")
        for key, value in config.items():
            self.log(f"  {key}: {value}")
        self.log("="*60)
    
    def log_results(self, results, stage):
        """
        ثبت نتایج
        
        پارامترها:
        -----------
        results : dict
            نتایج
        stage : str
            مرحله
        """
        self.log(f"\n📊 نتایج مرحله {stage}:")
        for key, value in results.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")
        
        # ذخیره نتایج
        self.results.append({
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
    
    def save_results(self, filename=None):
        """
        ذخیره نتایج
        
        پارامترها:
        -----------
        filename : str
            نام فایل
        """
        if filename is None:
            filename = f'results_{ExamUtils.get_timestamp()}.json'
        
        filepath = os.path.join(self.log_dir, filename)
        ExamUtils.save_json(self.results, filepath)
        self.log(f"✅ نتایج در {filepath} ذخیره شد")
    
    def get_summary(self):
        """
        دریافت خلاصه نتایج
        
        Returns:
        --------
        pd.DataFrame
            خلاصه نتایج
        """
        summary = []
        for exp in self.results:
            row = {'stage': exp['stage'], 'timestamp': exp['timestamp']}
            row.update(exp['results'])
            summary.append(row)
        
        return pd.DataFrame(summary)


class ConfigManager:
    """
    کلاس مدیریت پیکربندی
    """
    
    def __init__(self, config_path='config.json'):
        """
        مقداردهی اولیه
        
        پارامترها:
        -----------
        config_path : str
            مسیر فایل پیکربندی
        """
        self.config_path = config_path
        self.config = self.load_or_create_config()
    
    def default_config(self):
        """
        پیکربندی پیش‌فرض
        
        Returns:
        --------
        dict
            پیکربندی
        """
        return {
            'project': {
                'name': 'iran_konkur_project',
                'version': '1.0.0',
                'description': 'مدلسازی داده‌های کنکور ایران'
            },
            'data': {
                'path': 'data/iran_exam.csv',
                'task_type': 'regression',
                'test_size': 0.2,
                'val_size': 0.15,
                'random_state': 42
            },
            'baseline_models': {
                'enabled': True,
                'models': ['Linear', 'Ridge', 'Lasso', 'RF', 'XGB', 'LGBM', 'MLP']
            },
            'tabtransformer': {
                'enabled': True,
                'embedding_dim': 32,
                'num_heads': 4,
                'num_layers': 3,
                'mlp_hidden': [128, 64],
                'dropout': 0.2
            },
            'numerical_embeddings': {
                'enabled': True,
                'methods': ['ple', 'periodic', 'bucket']
            },
            'training': {
                'batch_size': 64,
                'epochs': 100,
                'learning_rate': 0.001,
                'patience': 15
            }
        }
    
    def load_or_create_config(self):
        """
        بارگذاری یا ایجاد پیکربندی
        
        Returns:
        --------
        dict
            پیکربندی
        """
        if os.path.exists(self.config_path):
            config = ExamUtils.load_json(self.config_path)
            print(f"📂 پیکربندی از {self.config_path} بارگذاری شد")
        else:
            config = self.default_config()
            ExamUtils.save_json(config, self.config_path)
            print(f"✅ پیکربندی پیش‌فرض در {self.config_path} ایجاد شد")
        
        return config
    
    def get(self, key, default=None):
        """
        دریافت مقدار از پیکربندی
        
        پارامترها:
        -----------
        key : str
            کلید (با نقطه جدا می‌شود)
        default : any
            مقدار پیش‌فرض
        
        Returns:
        --------
        any
            مقدار
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """
        تنظیم مقدار در پیکربندی
        
        پارامترها:
        -----------
        key : str
            کلید
        value : any
            مقدار
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        ExamUtils.save_json(self.config, self.config_path)
    
    def update(self, new_config):
        """
        به‌روزرسانی پیکربندی
        
        پارامترها:
        -----------
        new_config : dict
            پیکربندی جدید
        """
        self.config.update(new_config)
        ExamUtils.save_json(self.config, self.config_path)


class Timer:
    """
    کلاس اندازه‌گیری زمان
    """
    
    def __init__(self, name='Timer'):
        """
        مقداردهی اولیه
        
        پارامترها:
        -----------
        name : str
            نام تایمر
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
        print(f"⏱️ {self.name}: {self.elapsed:.2f} ثانیه")
    
    def start(self):
        """شروع计时"""
        self.start_time = time.time()
    
    def stop(self):
        """پایان计时"""
        self.end_time = time.time()
    
    @property
    def elapsed(self):
        """زمان سپری شده"""
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def reset(self):
        """بازنشانی"""
        self.start_time = None
        self.end_time = None


# ============================================
# توابع آماری کمکی
# ============================================

def calculate_statistics(data):
    """
    محاسبه آمار توصیفی
    
    پارامترها:
    -----------
    data : array
        داده
    
    Returns:
    --------
    dict
        آمار
    """
    return {
        'count': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'q25': np.percentile(data, 25),
        'median': np.median(data),
        'q75': np.percentile(data, 75),
        'max': np.max(data),
        'skewness': pd.Series(data).skew(),
        'kurtosis': pd.Series(data).kurtosis()
    }


def normalize_data(X, method='standard'):
    """
    نرمال‌سازی داده
    
    پارامترها:
    -----------
    X : array
        داده
    method : str
        روش نرمال‌سازی ('standard', 'minmax', 'robust')
    
    Returns:
    --------
    array
        داده نرمال‌سازی شده
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"روش نامعتبر: {method}")
    
    return scaler.fit_transform(X), scaler


def train_val_test_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    تقسیم داده به سه مجموعه
    
    پارامترها:
    -----------
    X : array
        ویژگی‌ها
    y : array
        برچسب‌ها
    train_size : float
        نسبت آموزش
    val_size : float
        نسبت اعتبارسنجی
    test_size : float
        نسبت آزمایش
    random_state : int
        seed
    
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # تقسیم اول: آموزش و موقت
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state
    )
    
    # تقسیم دوم: اعتبارسنجی و آزمایش
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(val_size+test_size), random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_section(title, char='='):
    """
    چاپ عنوان با خط
    
    پارامترها:
    -----------
    title : str
        عنوان
    char : str
        کاراکتر خط
    """
    print(f"\n{char*60}")
    print(f"{title}")
    print(f"{char*60}")


# ============================================
# تست
# ============================================

def test_utils():
    """تست توابع کمکی"""
    print("🧪 تست توابع کمکی")
    print("="*60)
    
    # تست seed
    ExamUtils.set_seed(42)
    
    # تست ایجاد پوشه
    ExamUtils.ensure_dir('test_dir')
    
    # تست زمان
    with Timer("تست"):
        import time
        time.sleep(1)
    
    # تست logger
    logger = ExperimentLogger('test_logs')
    logger.log("این یک پیام تست است")
    logger.log_config({'test': True, 'value': 123})
    logger.log_results({'rmse': 10.5, 'r2': 0.85}, 'test')
    
    # تست config
    config_manager = ConfigManager('test_config.json')
    print(f"config.data.path: {config_manager.get('data.path')}")
    
    # تست metrics
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    
    metrics = MetricsCalculator.regression_metrics(y_true, y_pred)
    print(f"\n📊 معیارهای رگرسیون:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ همه تست‌ها با موفقیت انجام شد")


if __name__ == "__main__":
    test_utils()
