"""
آموزش مدل‌های PyTorch برای داده‌های کنکور ایران
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from exam_models import ExamDataset, TabTransformerDataset, EarlyStopping


class ExamTrainer:
    """
    کلاس آموزش مدل‌های PyTorch
    """
    
    def __init__(self, model, model_type='mlp', model_name='model', save_dir='models'):
        self.model = model
        self.model_type = model_type
        self.model_name = model_name
        self.save_dir = save_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_r2': [],
            'val_r2': [],
            'epoch_time': []
        }
        
        self.best_model_state = None
        self.best_epoch = 0
        self.best_val_rmse = float('inf')
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"✅ Trainer ایجاد شد: {model_type} روی {self.device}")
    
    def create_dataloaders(self, 
                          X_cat_train=None, X_cont_train=None, y_train=None,
                          X_cat_val=None, X_cont_val=None, y_val=None,
                          X_train=None, y_train_mlp=None, 
                          X_val=None, y_val_mlp=None,
                          batch_size=64, num_workers=2):
        """
        ایجاد DataLoader برای آموزش و اعتبارسنجی
        """
        print("\n📦 ایجاد DataLoader...")
        
        if self.model_type == 'tabtransformer':
            if X_cat_train is None or X_cont_train is None or y_train is None:
                raise ValueError("برای TabTransformer باید X_cat_train, X_cont_train و y_train مشخص شوند")
            
            train_dataset = TabTransformerDataset(X_cat_train, X_cont_train, y_train)
            val_dataset = TabTransformerDataset(X_cat_val, X_cont_val, y_val)
            print(f"   📊 TabTransformer: categorical={X_cat_train.shape[1]}, continuous={X_cont_train.shape[1]}")
            
        else:
            if X_train is None or y_train_mlp is None:
                raise ValueError("برای MLP باید X_train و y_train_mlp مشخص شوند")
            
            train_dataset = ExamDataset(X_train, y_train_mlp)
            val_dataset = ExamDataset(X_val, y_val_mlp)
            print(f"   📊 MLP: features={X_train.shape[1]}")
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
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
                
                self.optimizer.zero_grad()
                output = self.model(x_cat, x_cont)
                
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
            
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return avg_loss, rmse, r2
    
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
        
        avg_loss = total_loss / len(self.val_loader)
        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return avg_loss, rmse, r2
    
    def train(self, epochs=100, lr=0.001, task_type='regression', patience=10, verbose=True):
        """
        آموزش کامل مدل
        """
        print("\n" + "="*80)
        print("🚀 شروع آموزش مدل")
        print("="*80)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        early_stopping = EarlyStopping(patience=patience, verbose=verbose)
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # آموزش
            train_loss, train_rmse, train_r2 = self.train_epoch()
            
            # اعتبارسنجی
            val_loss, val_rmse, val_r2 = self.validate_epoch()
            
            # زمان دوره
            epoch_time = time.time() - epoch_start
            
            # ذخیره در تاریخچه
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)
            self.history['epoch_time'].append(epoch_time)
            
            # بررسی بهترین مدل
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
            
            # Early stopping
            early_stopping(-val_rmse, self.model, epoch)
            
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"\n📊 Epoch {epoch}/{epochs}")
                print(f"   Train - Loss: {train_loss:.4f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}")
                print(f"   Val   - Loss: {val_loss:.4f}, RMSE: {val_rmse:.2f}, R²: {val_r2:.4f}")
            
            if early_stopping.early_stop:
                print(f"\n🛑 Early stopping در epoch {epoch}")
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ بهترین مدل از epoch {self.best_epoch} با RMSE={self.best_val_rmse:.2f}")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  زمان کل: {total_time:.2f} ثانیه ({total_time/60:.2f} دقیقه)")
        print("="*80)
    
    def evaluate(self, 
                X_cat_test=None, X_cont_test=None, y_test=None,
                X_test=None, y_test_mlp=None,
                batch_size=64):
        """
        ارزیابی مدل روی داده آزمایش
        """
        print("\n🧪 ارزیابی مدل روی داده آزمایش...")
        
        self.model.eval()
        
        if self.model_type == 'tabtransformer':
            test_dataset = TabTransformerDataset(X_cat_test, X_cont_test, y_test)
        else:
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
    
    def plot_history(self, save_path='plots/training_history.jpg'):
        """رسم تاریخچه آموزش"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RMSE
        axes[1].plot(epochs, self.history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
        axes[1].plot(epochs, self.history['val_rmse'], 'r-', label='Val RMSE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Training and Validation RMSE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {self.model_name}', fontsize=14)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 نمودار در {save_path} ذخیره شد")
    
    def save_model(self, filename=None):
        """ذخیره مدل"""
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
