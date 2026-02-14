"""
مدیریت داده‌های کنکور ایران
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class ExamDataManager:
    """
    کلاس مدیریت داده‌های کنکور
    """
    
    def __init__(self, data_dir='data', recording_file='analysis.txt', plots_folder='plots'):
        self.data_dir = data_dir
        self.recording_file = recording_file
        self.plots_folder = plots_folder
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plots_folder, exist_ok=True)
        
        self.df = None
        self.X = None
        self.y = None
        self.num_classes = None
        self.target_col = None
        self.task_type = None
        
        # For TabTransformer
        self.categories = None
        self.continuous_features = 0
        self.X_cat = None
        self.X_cont = None
        
        # For splits
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        # Preprocessing objects
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        print(f"📁 مدیر داده‌ها ایجاد شد")
    
    def load_and_prepare_data(self, data_path=None, task_type='regression'):
        """بارگذاری و آماده‌سازی داده‌ها"""
        print("\n🎓 در حال بارگذاری داده‌های کنکور...")
        
        self.task_type = task_type
        
        if data_path is None:
            data_path = self._find_exam_data()
        
        self._load_exam_data(data_path)
        self._clean_data()
        self._define_task()
        
        print(f"✅ داده‌ها بارگذاری شدند: {len(self.df)} نمونه")
        return self.df
    
    def _find_exam_data(self):
        """پیدا کردن فایل داده"""
        possible_paths = [
            'iran_exam.csv',
            'data/iran_exam.csv',
            '/content/iran_exam.csv',
            '../data/iran_exam.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 یافتن داده‌ها در: {path}")
                return path
        
        raise FileNotFoundError("❌ فایل داده‌های کنکور یافت نشد")
    
    def _load_exam_data(self, data_path):
        """بارگذاری فایل CSV"""
        try:
            self.df = pd.read_csv(data_path)
            print(f"📊 داده‌ها بارگذاری شدند. شکل: {self.df.shape}")
        except Exception as e:
            print(f"❌ خطا در بارگذاری داده‌ها: {e}")
            raise
    
    def _clean_data(self):
        """پاکسازی داده‌ها"""
        print("\n🧹 در حال پاکسازی داده‌ها...")
        
        # مدیریت مقادیر گمشده
        missing_before = self.df.isnull().sum().sum()
        if missing_before > 0:
            print(f"  🔍 مقادیر گمشده: {missing_before}")
            
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'نامشخص'
                    self.df[col].fillna(mode_val, inplace=True)
        
        # استانداردسازی منطقه
        if 'منطقه' in self.df.columns:
            region_map = {
                'منطقه1': 'منطقه1', 'منطقهيک': 'منطقه1', 'منطقهیک': 'منطقه1',
                'منطقه2': 'منطقه2', 'منطقهدو': 'منطقه2',
                'منطقه3': 'منطقه3', 'منطقهسه': 'منطقه3'
            }
            self.df['منطقه'] = self.df['منطقه'].apply(
                lambda x: region_map.get(str(x).strip(), str(x).strip())
            )
        
        print("✅ پاکسازی داده‌ها انجام شد")
    
    def _define_task(self):
        """تعریف وظیفه یادگیری"""
        print(f"\n🎯 تعریف وظیفه: {self.task_type}")
        
        if self.task_type == 'classification':
            if 'رتبه کشوری' in self.df.columns:
                threshold = self.df['رتبه کشوری'].quantile(0.2)
                self.df['target'] = (self.df['رتبه کشوری'] <= threshold).astype(int)
                self.target_col = 'target'
                print(f"  طبقه‌بندی: آستانه رتبه ≤ {threshold:.0f}")
        else:
            if 'رتبه کشوری' in self.df.columns:
                self.target_col = 'رتبه کشوری'
                print(f"  رگرسیون: پیش‌بینی رتبه کشوری")
    
    def prepare_for_traditional_models(self):
        """آماده‌سازی برای مدل‌های سنتی"""
        print("\n🔄 آماده‌سازی داده برای مدل‌های سنتی...")
        
        feature_cols = [col for col in self.df.columns if col != self.target_col]
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col in feature_cols]
        
        X_list = []
        numerical_cols = [col for col in feature_cols if col not in categorical_cols]
        
        if numerical_cols:
            X_numerical = self.df[numerical_cols].values
            X_numerical = self.scaler.fit_transform(X_numerical)
            X_list.append(X_numerical)
        
        if categorical_cols:
            X_categorical_list = []
            for col in categorical_cols:
                le = LabelEncoder()
                encoded = le.fit_transform(self.df[col].astype(str))
                X_categorical_list.append(encoded.reshape(-1, 1))
                self.label_encoders[col] = le
            
            X_categorical = np.hstack(X_categorical_list)
            X_list.append(X_categorical)
        
        if len(X_list) > 1:
            self.X = np.hstack(X_list)
        else:
            self.X = X_list[0]
        
        self.y = self.df[self.target_col].values
        print(f"✅ داده‌ها آماده شدند: X shape: {self.X.shape}")
        return self.X, self.y
    
    def prepare_for_tabtransformer(self):
        """آماده‌سازی برای TabTransformer"""
        print("\n🔄 آماده‌سازی داده برای TabTransformer...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numerical_cols = [col for col in self.df.columns 
                         if col not in categorical_cols and col != self.target_col]
        
        if len(categorical_cols) > 0:
            X_cat_list = []
            self.categories = []
            
            for col in categorical_cols:
                le = LabelEncoder()
                encoded = le.fit_transform(self.df[col].astype(str))
                X_cat_list.append(encoded.reshape(-1, 1))
                self.categories.append(len(le.classes_))
                self.label_encoders[col] = le
            
            self.X_cat = np.hstack(X_cat_list).astype(np.int64)
        else:
            self.X_cat = np.zeros((len(self.df), 0), dtype=np.int64)
            self.categories = []
        
        if len(numerical_cols) > 0:
            X_cont = self.df[numerical_cols].values.astype(np.float32)
            self.X_cont = self.scaler.fit_transform(X_cont)
            self.continuous_features = len(numerical_cols)
        else:
            self.X_cont = np.zeros((len(self.df), 0), dtype=np.float32)
            self.continuous_features = 0
        
        self.y = self.df[self.target_col].values
        
        print(f"✅ داده‌ها برای TabTransformer آماده شدند")
        return self.X_cat, self.X_cont, self.y
    
    def create_train_val_test_split(self, train_size=0.7, val_size=0.15, test_size=0.15):
        """تقسیم داده‌ها"""
        print(f"\n✂️ ایجاد تقسیم‌بندی داده‌ها...")
        
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            self.X, self.y, np.arange(len(self.X)),
            test_size=(val_size + test_size),
            random_state=42,
            stratify=self.y if self.task_type == 'classification' else None
        )
        
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp,
            test_size=test_size/(val_size+test_size),
            random_state=42,
            stratify=y_temp if self.task_type == 'classification' else None
        )
        
        self.train_indices = idx_train
        self.val_indices = idx_val
        self.test_indices = idx_test
        
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
