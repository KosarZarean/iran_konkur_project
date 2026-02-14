"""
توابع کمکی برای پروژه کنکور ایران
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
import random


class ExamUtils:
    """
    کلاس توابع کمکی عمومی
    """
    
    @staticmethod
    def set_seed(seed=42):
        """تنظیم seed برای تکرارپذیری"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"🎲 Seed تنظیم شد: {seed}")
    
    @staticmethod
    def ensure_dir(directory):
        """ایجاد پوشه در صورت عدم وجود"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        return directory
    
    @staticmethod
    def get_timestamp():
        """دریافت زمان فعلی به صورت رشته"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def save_json(data, filepath):
        """ذخیره داده در فایل JSON"""
        ExamUtils.ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON ذخیره شد: {filepath}")
    
    @staticmethod
    def load_json(filepath):
        """بارگذاری داده از فایل JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📂 JSON بارگذاری شد: {filepath}")
        return data
    
    @staticmethod
    def save_pickle(data, filepath):
        """ذخیره داده با pickle"""
        ExamUtils.ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Pickle ذخیره شد: {filepath}")
    
    @staticmethod
    def load_pickle(filepath):
        """بارگذاری داده با pickle"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"📂 Pickle بارگذاری شد: {filepath}")
        return data


class Timer:
    """
    کلاس اندازه‌گیری زمان
    """
    def __init__(self, name='Timer'):
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
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    @property
    def elapsed(self):
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


def print_section(title, char='='):
    """چاپ عنوان با خط"""
    print(f"\n{char*60}")
    print(f"{title}")
    print(f"{char*60}")


def calculate_statistics(data):
    """محاسبه آمار توصیفی"""
    return {
        'count': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'q25': np.percentile(data, 25),
        'median': np.median(data),
        'q75': np.percentile(data, 75),
        'max': np.max(data)
    }
