"""
آموزش مدل‌ها
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from exam_models import ExamDataset, TabTransformerDataset, EarlyStopping


class ExamTrainer:
    """کلاس آموزش مدل"""
    
    def __init__(self, model, model_type='mlp', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.model_type = model_type
        self.device = device
        self.model.to(device)
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': []}
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, 
                          X_cat_train=None, X_cont_train=None,
                          X_cat_val=None, X_cont_val=None, batch_size=64):
        """ایجاد DataLoader"""
        if self.model_type == 'tabtransformer':
            train_dataset = TabTransformerDataset(X_cat_train, X_cont_train, y_train)
            val_dataset = TabTransformerDataset(X_cat_val, X_cont_val, y_val)
        else:
            train_dataset = ExamDataset(X_train, y_train)
            val_dataset = ExamDataset(X_val, y_val)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    def train(self, epochs=100, lr=0.001, patience=10):
        """آموزش مدل"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience)
        
        print(f"\n🚀 شروع آموزش...")
        
        for epoch in range(epochs):
            # آموزش
            self.model.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            for batch in self.train_loader:
                if self.model_type == 'tabtransformer':
                    x_cat, x_cont, y = batch
                    x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(x_cat, x_cont)
                else:
                    X, y = batch
                    X, y = X.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(X)
                
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(output.detach().cpu().numpy())
                train_targets.extend(y.cpu().numpy())
            
            train_loss /= len(self.train_loader)
            
            # اعتبارسنجی
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    if self.model_type == 'tabtransformer':
                        x_cat, x_cont, y = batch
                        x_cat, x_cont = x_cat.to(self.device), x_cont.to(self.device)
                        output = self.model(x_cat, x_cont)
                    else:
                        X, y = batch
                        X = X.to(self.device)
                        output = self.model(X)
                    
                    loss = criterion(output, y.to(self.device))
                    val_loss += loss.item()
                    val_preds.extend(output.cpu().numpy())
                    val_targets.extend(y.numpy())
            
            val_loss /= len(self.val_loader)
            
            # محاسبه RMSE
            train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            
            # Early stopping
            early_stopping(-val_rmse, self.model)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train RMSE={train_rmse:.2f}, Val RMSE={val_rmse:.2f}")
            
            if early_stopping.early_stop:
                break
        
        print(f"✅ آموزش کامل شد")
    
    def evaluate(self, X_test, y_test, X_cat_test=None, X_cont_test=None):
        """ارزیابی روی داده تست"""
        self.model.eval()
        
        if self.model_type == 'tabtransformer':
            test_dataset = TabTransformerDataset(X_cat_test, X_cont_test, y_test)
        else:
            test_dataset = ExamDataset(X_test, y_test)
        
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if self.model_type == 'tabtransformer':
                    x_cat, x_cont, y = batch
                    x_cat, x_cont = x_cat.to(self.device), x_cont.to(self.device)
                    output = self.model(x_cat, x_cont)
                else:
                    X, y = batch
                    X = X.to(self.device)
                    output = self.model(X)
                
                predictions.extend(output.cpu().numpy())
                targets.extend(y.numpy())
        
        y_pred = np.array(predictions)
        y_true = np.array(targets)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n📊 نتایج روی داده تست:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        return {'y_true': y_true, 'y_pred': y_pred, 'rmse': rmse, 'r2': r2}
    
    def plot_history(self, save_path='training_history.jpg'):
        """رسم تاریخچه آموزش"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_rmse'], label='Train RMSE')
        plt.plot(self.history['val_rmse'], label='Val RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
