"""
آموزش مدل‌های PyTorch برای داده‌های کنکور ایران
شامل: توابع آموزش، اعتبارسنجی و ارزیابی
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from exam_models import ExamDataset, TabTransformerDataset, EarlyStopping, ModelUtils


class ExamTrainer:
    """
    کلاس آموزش مدل‌های PyTorch
    پشتیبانی از MLP، TabTransformer و Regressor
    """
    
    def __init__(self, model, model_type='mlp', device=None, 
                 model_name='model', save_dir='models'):
        """
        مقداردهی اولیه trainer
        
        پارامترها:
        -----------
        model : nn.Module
            مدل PyTorch
        model_type : str
            نوع مدل: 'mlp', 'tabtransformer', 'regressor'
        device : torch.device
            دستگاه اجرا
        model_name : str
            نام مدل برای ذخیره
        save_dir : str
            پوشه ذخیره مدل
        """
        self.model = model
        self.model_type = model_type
        self.model_name = model_name
        self.save_dir = save_dir
        
        # تعیین دستگاه
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # تاریخچه آموزش
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_mae': [],
            'val_mae': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # بهترین مدل
        self.best_model_state = None
        self.best_epoch = 0
        self.best_val_rmse = float('inf')
        
        # ایجاد پوشه ذخیره
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"✅ Trainer ایجاد شد:")
        print(f"   مدل: {model_type}")
        print(f"   دستگاه: {self.device}")
    
    def create_dataloaders(self, 
                          X_cat_train=None, X_cont_train=None, y_train=None,
                          X_cat_val=None, X_cont_val=None, y_val=None,
                          X_train=None, y_train_mlp=None, 
                          X_val=None, y_val_mlp=None,
                          batch_size=64, num_workers=2):
        """
        ایجاد DataLoader برای آموزش و اعتبارسنجی
        
        پارامترها:
        -----------
        X_cat_train, X_cont_train, y_train : array
            داده دسته‌ای و عددی برای TabTransformer (آموزش)
        X_cat_val, X_cont_val, y_val : array
            داده دسته‌ای و عددی برای TabTransformer (اعتبارسنجی)
        X_train, y_train_mlp : array
            داده برای MLP (آموزش)
        X_val, y_val_mlp : array
            داده برای MLP (اعتبارسنجی)
        batch_size : int
            اندازه batch
        num_workers : int
            تعداد workers برای DataLoader
        """
        print("\n📦 ایجاد DataLoader...")
        
        if self.model_type == 'tabtransformer':
            # بررسی وجود داده برای TabTransformer
            if X_cat_train is None or X_cont_train is None or y_train is None:
                raise ValueError("برای TabTransformer باید X_cat_train, X_cont_train و y_train مشخص شوند")
            
            # Dataset برای TabTransformer
            train_dataset = TabTransformerDataset(
                X_cat_train, X_cont_train, y_train
            )
            val_dataset = TabTransformerDataset(
                X_cat_val, X_cont_val, y_val
            )
            print(f"   📊 TabTransformer: categorical={X_cat_train.shape[1]}, continuous={X_cont_train.shape[1]}")
            
        else:
            # بررسی وجود داده برای MLP
            if X_train is None or y_train_mlp is None:
                raise ValueError("برای MLP باید X_train و y_train_mlp مشخص شوند")
            
            # Dataset برای مدل‌های معمولی
            train_dataset = ExamDataset(X_train, y_train_mlp)
            val_dataset = ExamDataset(X_val, y_val_mlp)
            print(f"   📊 MLP: features={X_train.shape[1]}")
        
        # DataLoader
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"   ✅ Train: {len(train_dataset)} نمونه ({len(self.train_loader)} batch)")
        print(f"   ✅ Val: {len(val_dataset)} نمونه ({len(self.val_loader)} batch)")
    
    def train_epoch(self):
        """یک دوره آموزش"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in self.train_loader:
            if self.model_type == 'tabtransformer':
                x_cat, x_cont, y = batch
                x_cat = x_cat.to(self.device)
                x_cont = x_cont.to(self.device)
                y = y.to(self.device)
                
                # پیش‌بینی
                self.optimizer.zero_grad()
                output = self.model(x_cat, x_cont)
                
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                # پیش‌بینی
                self.optimizer.zero_grad()
                output = self.model(x)
            
            # محاسبه loss
            loss = self.criterion(output, y)
            
            # پس‌رو و بهینه‌سازی
            loss.backward()
            
            # Gradient clipping (اختیاری)
            if hasattr(self, 'clip_grad_norm') and self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.optimizer.step()
            
            # ذخیره نتایج
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        
        # محاسبه معیارها
        avg_loss = total_loss / len(self.train_loader)
        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return avg_loss, rmse, mae, r2
    
    def validate_epoch(self):
        """یک دوره اعتبارسنجی"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if self.model_type == 'tabtransformer':
                    x_cat, x_cont, y = batch
                    x_cat = x_cat.to(self.device)
                    x_cont = x_cont.to(self.device)
                    y = y.to(self.device)
                    
                    output = self.model(x_cat, x_cont)
                    
                else:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    output = self.model(x)
                
                loss = self.criterion(output, y)
                
                total_loss += loss.item()
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # محاسبه معیارها
        avg_loss = total_loss / len(self.val_loader)
        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return avg_loss, rmse, mae, r2
    
    def train(self, epochs=100, lr=0.001, weight_decay=1e-5, task_type='regression',
             patience=15, min_delta=0.001, clip_grad_norm=None,
             scheduler=None, verbose=True):
        """
        آموزش کامل مدل
        
        پارامترها:
        -----------
        epochs : int
            تعداد دوره‌ها
        lr : float
            نرخ یادگیری
        weight_decay : float
            تنظیم L2
        task_type : str
            نوع وظیفه ('regression' یا 'classification')
        patience : int
            تعداد دوره‌های تحمل برای early stopping
        min_delta : float
            حداقل بهبود
        clip_grad_norm : float
            حداکثر نرم گرادیان
        scheduler : torch.optim.lr_scheduler
            برنامه‌ریز نرخ یادگیری
        verbose : bool
            نمایش جزئیات
        """
        print("\n" + "="*80)
        print("🚀 شروع آموزش مدل")
        print("="*80)
        
        # تنظیمات
        self.clip_grad_norm = clip_grad_norm
        self.criterion = nn.MSELoss()  # برای رگرسیون
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Early stopping
        from exam_models import EarlyStopping
        early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            verbose=verbose
        )
        
        # زمان شروع
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # آموزش
            train_loss, train_rmse, train_mae, train_r2 = self.train_epoch()
            
            # اعتبارسنجی
            val_loss, val_rmse, val_mae, val_r2 = self.validate_epoch()
            
            # زمان دوره
            epoch_time = time.time() - epoch_start
            
            # ذخیره در تاریخچه
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)
            self.history['epoch_time'].append(epoch_time)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # به‌روزرسانی scheduler
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # بررسی بهترین مدل
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
            
            # Early stopping
            early_stopping(-val_rmse, self.model, epoch)
            
            # نمایش پیشرفت
            if verbose and (epoch % 10 == 0 or epoch == 1 or early_stopping.early_stop):
                print(f"\n📊 Epoch {epoch}/{epochs}")
                print(f"   Train - Loss: {train_loss:.4f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}")
                print(f"   Val   - Loss: {val_loss:.4f}, RMSE: {val_rmse:.2f}, R²: {val_r2:.4f}")
                print(f"   زمان: {epoch_time:.2f}s, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if early_stopping.early_stop:
                print(f"\n🛑 Early stopping در epoch {epoch}")
                break
        
        # بارگذاری بهترین مدل
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ بهترین مدل از epoch {self.best_epoch} بارگذاری شد")
            print(f"   بهترین RMSE اعتبارسنجی: {self.best_val_rmse:.2f}")
        
        # زمان کل
        total_time = time.time() - start_time
        print(f"\n⏱️  زمان کل آموزش: {total_time:.2f} ثانیه ({total_time/60:.2f} دقیقه)")
        print("="*80)
    
    def evaluate(self, 
                X_cat_test=None, X_cont_test=None, y_test=None,
                X_test=None, y_test_mlp=None,
                batch_size=64):
        """
        ارزیابی مدل روی داده آزمایش
        
        پارامترها:
        -----------
        X_cat_test, X_cont_test, y_test : array
            داده دسته‌ای و عددی برای TabTransformer
        X_test, y_test_mlp : array
            داده برای MLP
        batch_size : int
            اندازه batch
        
        Returns:
        --------
        dict
            نتایج ارزیابی
        """
        print("\n🧪 ارزیابی مدل روی داده آزمایش...")
        
        self.model.eval()
        
        # ایجاد DataLoader
        if self.model_type == 'tabtransformer':
            if X_cat_test is None or X_cont_test is None or y_test is None:
                raise ValueError("برای TabTransformer باید X_cat_test, X_cont_test و y_test مشخص شوند")
            
            test_dataset = TabTransformerDataset(X_cat_test, X_cont_test, y_test)
        else:
            if X_test is None or y_test_mlp is None:
                raise ValueError("برای MLP باید X_test و y_test_mlp مشخص شوند")
            
            test_dataset = ExamDataset(X_test, y_test_mlp)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if self.model_type == 'tabtransformer':
                    x_cat, x_cont, y = batch
                    x_cat = x_cat.to(self.device)
                    x_cont = x_cont.to(self.device)
                    output = self.model(x_cat, x_cont)
                    
                else:
                    x, y = batch
                    x = x.to(self.device)
                    output = self.model(x)
                
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(y.numpy())
        
        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)
        
        # محاسبه معیارها
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        print(f"\n📊 نتایج ارزیابی:")
        print(f"   RMSE: {results['rmse']:.2f}")
        print(f"   MAE: {results['mae']:.2f}")
        print(f"   R²: {results['r2']:.4f}")
        
        return results
    
    def predict(self, 
               X_cat=None, X_cont=None,
               X=None,
               batch_size=64):
        """
        پیش‌بینی با مدل
        
        پارامترها:
        -----------
        X_cat, X_cont : array
            ویژگی‌های دسته‌ای و عددی برای TabTransformer
        X : array
            ویژگی‌ها برای MLP
        batch_size : int
            اندازه batch
        
        Returns:
        --------
        array
            پیش‌بینی‌ها
        """
        self.model.eval()
        
        if self.model_type == 'tabtransformer':
            if X_cat is None or X_cont is None:
                raise ValueError("برای TabTransformer باید X_cat و X_cont مشخص شوند")
            
            dataset = TabTransformerDataset(X_cat, X_cont, np.zeros(len(X_cat)))
        else:
            if X is None:
                raise ValueError("برای MLP باید X مشخص شود")
            
            dataset = ExamDataset(X, np.zeros(len(X)))
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                if self.model_type == 'tabtransformer':
                    x_cat, x_cont, _ = batch
                    x_cat = x_cat.to(self.device)
                    x_cont = x_cont.to(self.device)
                    output = self.model(x_cat, x_cont)
                    
                else:
                    x, _ = batch
                    x = x.to(self.device)
                    output = self.model(x)
                
                predictions.extend(output.cpu().numpy())
        
        return np.array(predictions)
    
    def plot_history(self, save_path='plots/training_history.jpg'):
        """
        رسم تاریخچه آموزش
        
        پارامترها:
        -----------
        save_path : str
            مسیر ذخیره نمودار
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE
        axes[0, 1].plot(epochs, self.history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_rmse'], 'r-', label='Val RMSE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Training and Validation RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MAE
        axes[0, 2].plot(epochs, self.history['train_mae'], 'b-', label='Train MAE', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_mae'], 'r-', label='Val MAE', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].set_title('Training and Validation MAE')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. R²
        axes[1, 0].plot(epochs, self.history['train_r2'], 'b-', label='Train R²', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_r2'], 'r-', label='Val R²', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('Training and Validation R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Learning Rate
        axes[1, 1].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Epoch Time
        axes[1, 2].plot(epochs, self.history['epoch_time'], 'm-', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].set_title('Epoch Training Time')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {self.model_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # ذخیره
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 نمودار در {save_path} ذخیره شد")
    
    def save_model(self, filename=None):
        """
        ذخیره مدل
        
        پارامترها:
        -----------
        filename : str
            نام فایل
        """
        if filename is None:
            filename = f'{self.model_name}_best.pt'
        
        save_path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_model_state': self.best_model_state,
            'best_val_rmse': self.best_val_rmse,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'model_type': self.model_type,
            'model_name': self.model_name
        }
        
        torch.save(checkpoint, save_path)
        print(f"💾 مدل در {save_path} ذخیره شد")
        
        return save_path
    
    def load_model(self, path):
        """
        بارگذاری مدل
        
        پارامترها:
        -----------
        path : str
            مسیر فایل مدل
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model_state = checkpoint.get('best_model_state')
        self.best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"📂 مدل از {path} بارگذاری شد")
        print(f"   بهترین RMSE: {self.best_val_rmse:.2f} (epoch {self.best_epoch})")


# ============================================
# تابع تست
# ============================================

def test_trainer():
    """تست trainer با داده نمونه"""
    print("🧪 تست ExamTrainer")
    print("="*60)
    
    from exam_models import ExamMLP
    
    # داده نمونه
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 10)
    y_val = np.random.randn(200)
    X_test = np.random.randn(200, 10)
    y_test = np.random.randn(200)
    
    # ایجاد مدل
    model = ExamMLP(input_dim=10, hidden_dims=[64, 32], output_dim=1)
    
    # ایجاد trainer
    trainer = ExamTrainer(model, model_type='mlp', model_name='test_model')
    
    # ایجاد dataloader
    trainer.create_dataloaders(
        X_train=X_train, 
        y_train_mlp=y_train,
        X_val=X_val, 
        y_val_mlp=y_val,
        batch_size=32
    )
    
    # آموزش
    trainer.train(epochs=10, verbose=True)
    
    # رسم تاریخچه
    trainer.plot_history('plots/test_history.jpg')
    
    # ارزیابی
    results = trainer.evaluate(X_test=X_test, y_test_mlp=y_test)
    
    print("\n✅ تست با موفقیت انجام شد")
    return results


if __name__ == "__main__":
    test_trainer()
