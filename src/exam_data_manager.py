"""
مدیریت داده‌های کنکور ایران
این فایل شامل کلاس‌های مدیریت، پیش‌پردازش و آماده‌سازی داده‌ها است
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class ExamDataManager:
    """
    کلاس اصلی مدیریت داده‌های کنکور
    مسئولیت: بارگذاری، پاکسازی، پیش‌پردازش و تقسیم داده‌ها
    """
    
    def __init__(self, data_dir='data', recording_file='analysis.txt', plots_folder='plots'):
        """
        مقداردهی اولیه مدیر داده
        
        پارامترها:
        -----------
        data_dir : str
            پوشه حاوی داده‌ها
        recording_file : str
            فایل برای ثبت تحلیل‌ها
        plots_folder : str
            پوشه برای ذخیره نمودارها
        """
        self.data_dir = data_dir
        self.recording_file = recording_file
        self.plots_folder = plots_folder
        
        # ایجاد پوشه‌ها
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plots_folder, exist_ok=True)
        
        # ویژگی‌های داده
        self.df = None
        self.X = None
        self.y = None
        self.num_classes = None
        self.target_col = None
        self.task_type = None
        self.feature_names = None
        
        # برای TabTransformer
        self.categories = None
        self.continuous_features = 0
        self.X_cat = None
        self.X_cont = None
        
        # برای تقسیم داده
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # ابزارهای پیش‌پردازش
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        
        print(f"📁 مدیر داده‌ها ایجاد شد: data_dir={data_dir}")
    
    def load_and_prepare_data(self, data_path=None, task_type='regression'):
        """
        بارگذاری و آماده‌سازی داده‌ها
        
        پارامترها:
        -----------
        data_path : str
            مسیر فایل داده
        task_type : str
            نوع وظیفه: 'regression' یا 'classification'
        
        Returns:
        --------
        pd.DataFrame
            داده‌های بارگذاری شده
        """
        print("\n" + "="*60)
        print("🎓 شروع بارگذاری و آماده‌سازی داده‌ها")
        print("="*60)
        
        self.task_type = task_type
        
        # پیدا کردن فایل داده
        if data_path is None:
            data_path = self._find_exam_data()
        
        # بارگذاری داده
        self._load_exam_data(data_path)
        
        # پاکسازی داده
        self._clean_data()
        
        # تعریف وظیفه
        self._define_task()
        
        # شناسایی ستون‌ها
        self._identify_columns()
        
        print("\n" + "="*60)
        print(f"✅ داده‌ها با موفقیت بارگذاری شدند")
        print(f"   تعداد نمونه‌ها: {len(self.df):,}")
        print(f"   تعداد ویژگی‌ها: {len(self.df.columns)}")
        print("="*60)
        
        return self.df
    
    def _find_exam_data(self):
        """پیدا کردن فایل داده در مسیرهای مختلف"""
        possible_paths = [
            'iran_exam.csv',
            'data/iran_exam.csv',
            '/content/iran_exam.csv',
            '../data/iran_exam.csv',
            './iran_exam.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 فایل داده در مسیر یافت شد: {path}")
                return path
        
        # اگر فایل پیدا نشد، یک نمونه داده بساز
        print("⚠️ فایل داده پیدا نشد. یک نمونه داده ساخته می‌شود.")
        return self._create_sample_data()
    
    def _create_sample_data(self, n_samples=1000):
        """ایجاد داده نمونه برای تست"""
        print("📊 در حال ایجاد داده نمونه...")
        
        np.random.seed(42)
        data = {
            'شهر': np.random.choice(['تهران', 'مشهد', 'اصفهان', 'تبریز', 'شیراز'], n_samples),
            'رتبه کشوری': np.random.randint(1, 200000, n_samples),
            'رتبه در منطقه': np.random.randint(1, 50000, n_samples),
            'منطقه': np.random.choice(['منطقه1', 'منطقه2', 'منطقه3'], n_samples),
            'میانگین تراز کانون': np.random.uniform(4000, 8000, n_samples),
            'تعداد آزمون': np.random.randint(1, 30, n_samples),
            'رشته قبولی': np.random.choice(['پزشکی', 'مهندسی', 'حقوق'], n_samples),
            'دانشگاه قبولی': np.random.choice(['تهران', 'شریف', 'امیرکبیر'], n_samples),
            'سال': np.random.choice([1398, 1399, 1400, 1401], n_samples)
        }
        
        df = pd.DataFrame(data)
        sample_path = 'data/sample_iran_exam.csv'
        os.makedirs('data', exist_ok=True)
        df.to_csv(sample_path, index=False)
        print(f"✅ داده نمونه در {sample_path} ذخیره شد")
        
        return sample_path
    
    def _load_exam_data(self, data_path):
        """بارگذاری فایل CSV"""
        try:
            self.df = pd.read_csv(data_path)
            print(f"\n📊 داده‌ها با موفقیت بارگذاری شدند")
            print(f"   شکل داده‌ها: {self.df.shape}")
            print(f"   ستون‌ها: {list(self.df.columns)}")
            
        except Exception as e:
            print(f"❌ خطا در بارگذاری داده‌ها: {e}")
            raise
    
    def _identify_columns(self):
        """شناسایی ستون‌های عددی و دسته‌ای"""
        self.num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if 'رتبه کشوری' in self.num_cols:
            self.num_cols.remove('رتبه کشوری')
        
        print(f"\n📋 شناسایی ستون‌ها:")
        print(f"   ویژگی‌های عددی: {len(self.num_cols)} - {self.num_cols[:5]}")
        print(f"   ویژگی‌های دسته‌ای: {len(self.cat_cols)} - {self.cat_cols[:5]}")
    
    def _clean_data(self):
        """پاکسازی داده‌ها"""
        print("\n🧹 در حال پاکسازی داده‌ها...")
        
        # 1. مدیریت مقادیر گمشده
        self._handle_missing_values()
        
        # 2. استانداردسازی مقادیر دسته‌ای
        self._standardize_categorical()
        
        # 3. حذف داده‌های پرت (اختیاری)
        # self._remove_outliers()
        
        print("✅ پاکسازی داده‌ها انجام شد")
    
    def _handle_missing_values(self):
        """مدیریت مقادیر گمشده"""
        missing_before = self.df.isnull().sum().sum()
        
        if missing_before > 0:
            print(f"  🔍 مقادیر گمشده یافت شد: {missing_before}")
            
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    if col in self.num_cols:
                        # برای ستون‌های عددی از میانه استفاده کن
                        median_val = self.df[col].median()
                        self.df[col].fillna(median_val, inplace=True)
                        print(f"    {col}: پر شده با میانه ({median_val:.2f})")
                    else:
                        # برای ستون‌های دسته‌ای از مد استفاده کن
                        mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'نامشخص'
                        self.df[col].fillna(mode_val, inplace=True)
                        print(f"    {col}: پر شده با مد ({mode_val})")
        
        missing_after = self.df.isnull().sum().sum()
        print(f"  ✅ مقادیر گمشده باقی‌مانده: {missing_after}")
    
    def _standardize_categorical(self):
        """استانداردسازی مقادیر دسته‌ای"""
        print("  🏷️ استانداردسازی مقادیر دسته‌ای...")
        
        # استانداردسازی منطقه
        if 'منطقه' in self.df.columns:
            region_mapping = {
                'منطقه1': 'منطقه1', 'منطقهيک': 'منطقه1', 'منطقهیک': 'منطقه1', 'منطقه 1': 'منطقه1',
                'منطقه2': 'منطقه2', 'منطقهدو': 'منطقه2', 'منطقه 2': 'منطقه2',
                'منطقه3': 'منطقه3', 'منطقهسه': 'منطقه3', 'منطقه 3': 'منطقه3'
            }
            
            before = self.df['منطقه'].nunique()
            self.df['منطقه'] = self.df['منطقه'].apply(
                lambda x: region_mapping.get(str(x).strip(), str(x).strip())
            )
            after = self.df['منطقه'].nunique()
            print(f"    منطقه: {before} → {after} مقدار یکتا")
        
        # استانداردسازی نام شهرها
        if 'شهر' in self.df.columns:
            self.df['شهر'] = self.df['شهر'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            print(f"    شهر: {self.df['شهر'].nunique()} شهر یکتا")
    
    def _remove_outliers(self, threshold=3):
        """حذف داده‌های پرت"""
        print("  ⚠️ حذف داده‌های پرت...")
        
        for col in self.num_cols:
            if col != 'رتبه کشوری':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                before = len(self.df)
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                after = len(self.df)
                
                if before > after:
                    print(f"    {col}: {before - after} نمونه پرت حذف شد")
    
    def _define_task(self):
        """تعریف وظیفه یادگیری"""
        print(f"\n🎯 تعریف وظیفه: {self.task_type}")
        
        if self.task_type == 'classification':
            # طبقه‌بندی - 20% برتر
            if 'رتبه کشوری' in self.df.columns:
                threshold = self.df['رتبه کشوری'].quantile(0.2)
                self.df['target'] = (self.df['رتبه کشوری'] <= threshold).astype(int)
                self.target_col = 'target'
                
                class_counts = self.df['target'].value_counts()
                print(f"   آستانه: رتبه ≤ {threshold:.0f}")
                print(f"   توزیع کلاس‌ها: کلاس 0={class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(self.df)*100:.1f}%), کلاس 1={class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(self.df)*100:.1f}%)")
        
        elif self.task_type == 'regression':
            # رگرسیون - پیش‌بینی رتبه
            if 'رتبه کشوری' in self.df.columns:
                self.target_col = 'رتبه کشوری'
                print(f"   هدف: پیش‌بینی رتبه کشوری")
                print(f"   محدوده رتبه: {self.df['رتبه کشوری'].min():.0f} - {self.df['رتبه کشوری'].max():.0f}")
        
        else:
            raise ValueError(f"نوع وظیفه نامعتبر: {self.task_type}")
    
    def prepare_for_traditional_models(self):
        """
        آماده‌سازی داده برای مدل‌های سنتی
        شامل: کدگذاری، نرمال‌سازی و ایجاد ماتریس ویژگی‌ها
        """
        print("\n🔄 آماده‌سازی داده برای مدل‌های سنتی...")
        
        # انتخاب ستون‌های ویژگی
        feature_cols = [col for col in self.df.columns if col != self.target_col]
        categorical_cols = [col for col in self.cat_cols if col in feature_cols]
        numerical_cols = [col for col in self.num_cols if col in feature_cols]
        
        X_list = []
        
        # پردازش ویژگی‌های عددی
        if numerical_cols:
            print(f"  📊 ویژگی‌های عددی ({len(numerical_cols)}): {numerical_cols[:5]}")
            X_num = self.df[numerical_cols].values
            X_num = self.scaler.fit_transform(X_num)
            X_list.append(X_num)
        
        # پردازش ویژگی‌های دسته‌ای
        if categorical_cols:
            print(f"  🏷️ ویژگی‌های دسته‌ای ({len(categorical_cols)}): {categorical_cols[:5]}")
            X_cat_list = []
            
            for col in categorical_cols:
                le = LabelEncoder()
                encoded = le.fit_transform(self.df[col].astype(str))
                X_cat_list.append(encoded.reshape(-1, 1))
                self.label_encoders[col] = le
            
            X_cat = np.hstack(X_cat_list)
            X_list.append(X_cat)
        
        # ترکیب ویژگی‌ها
        if len(X_list) > 1:
            self.X = np.hstack(X_list)
        else:
            self.X = X_list[0]
        
        self.y = self.df[self.target_col].values
        self.feature_names = feature_cols
        
        print(f"  ✅ داده‌ها آماده شدند:")
        print(f"     X shape: {self.X.shape}")
        print(f"     y shape: {self.y.shape}")
        
        return self.X, self.y
    
    def prepare_for_tabtransformer(self):
        """
        آماده‌سازی داده برای TabTransformer
        جداسازی ویژگی‌های دسته‌ای و عددی
        """
        print("\n🔄 آماده‌سازی داده برای TabTransformer...")
        
        categorical_cols = [col for col in self.cat_cols if col != self.target_col]
        numerical_cols = [col for col in self.num_cols if col != self.target_col]
        
        # ویژگی‌های دسته‌ای
        if categorical_cols:
            X_cat_list = []
            self.categories = []
            
            for col in categorical_cols:
                le = LabelEncoder()
                encoded = le.fit_transform(self.df[col].astype(str))
                X_cat_list.append(encoded.reshape(-1, 1))
                self.categories.append(len(le.classes_))
                self.label_encoders[col] = le
            
            self.X_cat = np.hstack(X_cat_list).astype(np.int64)
            print(f"  🏷️ ویژگی‌های دسته‌ای: {self.X_cat.shape}, categories: {self.categories}")
        else:
            self.X_cat = np.zeros((len(self.df), 0), dtype=np.int64)
            self.categories = []
            print(f"  🏷️ هیچ ویژگی دسته‌ای یافت نشد")
        
        # ویژگی‌های عددی
        if numerical_cols:
            X_cont = self.df[numerical_cols].values.astype(np.float32)
            self.X_cont = self.scaler.fit_transform(X_cont)
            self.continuous_features = len(numerical_cols)
            print(f"  📊 ویژگی‌های عددی: {self.X_cont.shape}")
        else:
            self.X_cont = np.zeros((len(self.df), 0), dtype=np.float32)
            self.continuous_features = 0
            print(f"  📊 هیچ ویژگی عددی یافت نشد")
        
        self.y = self.df[self.target_col].values
        
        return self.X_cat, self.X_cont, self.y
    
    def create_train_val_test_split(self, train_size=0.7, val_size=0.15, test_size=0.15):
        """
        تقسیم داده به سه مجموعه آموزش، اعتبارسنجی و آزمایش
        
        پارامترها:
        -----------
        train_size : float
            نسبت داده آموزش
        val_size : float
            نسبت داده اعتبارسنجی
        test_size : float
            نسبت داده آزمایش
        """
        print(f"\n✂️ تقسیم داده‌ها: train={train_size}, val={val_size}, test={test_size}")
        
        # تقسیم اول: آموزش و موقت
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            self.X, self.y, np.arange(len(self.X)),
            test_size=(val_size + test_size),
            random_state=42,
            stratify=self.y if self.task_type == 'classification' else None
        )
        
        # تقسیم دوم: اعتبارسنجی و آزمایش
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp,
            test_size=test_size/(val_size+test_size),
            random_state=42,
            stratify=y_temp if self.task_type == 'classification' else None
        )
        
        # ذخیره اندیس‌ها
        self.train_indices = idx_train
        self.val_indices = idx_val
        self.test_indices = idx_test
        
        # ذخیره داده‌ها
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        print(f"  ✅ آموزش: {len(X_train)} نمونه ({len(X_train)/len(self.X)*100:.1f}%)")
        print(f"  ✅ اعتبارسنجی: {len(X_val)} نمونه ({len(X_val)/len(self.X)*100:.1f}%)")
        print(f"  ✅ آزمایش: {len(X_test)} نمونه ({len(X_test)/len(self.X)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def exploratory_data_analysis(self):
        """
        تحلیل اکتشافی داده‌ها و رسم نمودارهای اولیه
        """
        print("\n🔍 شروع تحلیل اکتشافی داده‌ها...")
        
        # ایجاد پوشه برای نمودارها
        eda_dir = os.path.join(self.plots_folder, 'eda')
        os.makedirs(eda_dir, exist_ok=True)
        
        # 1. آمار توصیفی
        self._basic_statistics()
        
        # 2. تحلیل متغیر هدف
        self._plot_target_distribution(eda_dir)
        
        # 3. تحلیل ویژگی‌های عددی
        self._plot_numerical_features(eda_dir)
        
        # 4. تحلیل ویژگی‌های دسته‌ای
        self._plot_categorical_features(eda_dir)
        
        # 5. ماتریس همبستگی
        self._plot_correlation_matrix(eda_dir)
        
        print(f"✅ تحلیل اکتشافی کامل شد. نمودارها در {eda_dir} ذخیره شدند")
    
    def _basic_statistics(self):
        """آمار توصیفی پایه"""
        print("\n📊 آمار توصیفی:")
        print(f"   تعداد کل نمونه‌ها: {len(self.df):,}")
        print(f"   تعداد ویژگی‌ها: {len(self.df.columns)}")
        print(f"   ویژگی‌های عددی: {len(self.num_cols)}")
        print(f"   ویژگی‌های دسته‌ای: {len(self.cat_cols)}")
        print(f"   مقادیر گمشده: {self.df.isnull().sum().sum()}")
    
    def _plot_target_distribution(self, save_dir):
        """رسم توزیع متغیر هدف"""
        plt.figure(figsize=(12, 5))
        
        if self.task_type == 'classification':
            class_counts = self.df[self.target_col].value_counts()
            
            plt.subplot(1, 2, 1)
            plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'lightcoral'])
            plt.title('توزیع کلاس‌های هدف')
            plt.xlabel('کلاس')
            plt.ylabel('تعداد')
            plt.xticks([0, 1], ['کلاس 0', 'کلاس 1'])
            
            plt.subplot(1, 2, 2)
            plt.pie(class_counts.values, labels=['کلاس 0', 'کلاس 1'], 
                   autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
            plt.title('نسبت کلاس‌ها')
            
        else:
            plt.subplot(1, 2, 1)
            plt.hist(self.df[self.target_col], bins=50, edgecolor='black', alpha=0.7)
            plt.title('توزیع رتبه کشوری')
            plt.xlabel('رتبه')
            plt.ylabel('تعداد')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(self.df[self.target_col])
            plt.title('Boxplot رتبه کشوری')
            plt.ylabel('رتبه')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'target_distribution.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_numerical_features(self, save_dir):
        """رسم توزیع ویژگی‌های عددی"""
        for col in self.num_cols[:6]:  # حداکثر ۶ ویژگی
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(self.df[col], bins=30, edgecolor='black', alpha=0.7)
            plt.title(f'توزیع {col}')
            plt.xlabel(col)
            plt.ylabel('تعداد')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(self.df[col])
            plt.title(f'Boxplot {col}')
            plt.ylabel(col)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'dist_{col}.jpg'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_categorical_features(self, save_dir):
        """رسم توزیع ویژگی‌های دسته‌ای"""
        for col in self.cat_cols[:3]:  # حداکثر ۳ ویژگی
            plt.figure(figsize=(12, 6))
            
            # ۱۵ دسته برتر
            value_counts = self.df[col].value_counts().head(15)
            
            plt.bar(range(len(value_counts)), value_counts.values, color='skyblue', edgecolor='black')
            plt.title(f'توزیع {col} (۱۵ دسته برتر)')
            plt.xlabel(col)
            plt.ylabel('تعداد')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'cat_{col}.jpg'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_correlation_matrix(self, save_dir):
        """رسم ماتریس همبستگی"""
        if len(self.num_cols) >= 2:
            plt.figure(figsize=(12, 10))
            
            corr_matrix = self.df[self.num_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap='coolwarm', center=0, square=True,
                       linewidths=1, cbar_kws={"shrink": 0.8})
            
            plt.title('ماتریس همبستگی ویژگی‌های عددی')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'correlation_matrix.jpg'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def get_data_summary(self):
        """گرفتن خلاصه اطلاعات داده"""
        summary = {
            'total_samples': len(self.df),
            'total_features': len(self.df.columns),
            'numeric_features': len(self.num_cols),
            'categorical_features': len(self.cat_cols),
            'missing_values': self.df.isnull().sum().sum(),
            'task_type': self.task_type,
            'target_column': self.target_col
        }
        
        if self.task_type == 'classification':
            summary['class_distribution'] = self.df[self.target_col].value_counts().to_dict()
        else:
            summary['target_min'] = self.df[self.target_col].min()
            summary['target_max'] = self.df[self.target_col].max()
            summary['target_mean'] = self.df[self.target_col].mean()
        
        return summary


# کلاس کمکی برای تحلیل سریع
class ExamDataAnalyzer:
    """
    کلاس تحلیل سریع داده‌ها
    """
    
    def __init__(self, df):
        self.df = df
        self.num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    def quick_summary(self):
        """گزارش سریع"""
        print("\n📊 گزارش سریع داده‌ها:")
        print(f"   تعداد نمونه‌ها: {len(self.df):,}")
        print(f"   تعداد ویژگی‌ها: {len(self.df.columns)}")
        print(f"   ویژگی‌های عددی: {len(self.num_cols)}")
        print(f"   ویژگی‌های دسته‌ای: {len(self.cat_cols)}")
        print(f"   مقادیر گمشده: {self.df.isnull().sum().sum()}")
        
        if len(self.num_cols) > 0:
            print("\n   آمار ویژگی‌های عددی:")
            print(self.df[self.num_cols].describe().round(2))
    
    def check_missing(self):
        """بررسی مقادیر گمشده"""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) > 0:
            print("\n🔍 مقادیر گمشده:")
            for col, val in missing.items():
                print(f"   {col}: {val} ({val/len(self.df)*100:.1f}%)")
        else:
            print("\n✅ هیچ مقدار گمشده‌ای وجود ندارد")
        
        return missing
    
    def check_duplicates(self):
        """بررسی داده‌های تکراری"""
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"\n⚠️ {duplicates} داده تکراری وجود دارد")
        else:
            print("\n✅ هیچ داده تکراری وجود ندارد")
        
        return duplicates
