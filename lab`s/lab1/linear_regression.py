import numpy as np
from typing import Literal, Optional, Tuple
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==================== НОРМАЛИЗАЦИЯ ====================

def z_score_normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Вычисляем среднее
    mean = np.mean(X, axis=0)
    if not isinstance(mean, np.ndarray):
        mean = np.array([mean])
    mean = np.atleast_1d(mean)
    
    variance = np.mean((X - mean) ** 2, axis=0)
    std = np.sqrt(variance)
    if not isinstance(std, np.ndarray):
        std = np.array([std])
    std = np.atleast_1d(std)
    
    std = np.where(std == 0, 1.0, std)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def min_max_normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    if not isinstance(min_vals, np.ndarray):
        min_vals = np.array([min_vals])
    if not isinstance(max_vals, np.ndarray):
        max_vals = np.array([max_vals])
    min_vals = np.atleast_1d(min_vals)
    max_vals = np.atleast_1d(max_vals)
    
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1.0, range_vals)
    X_normalized = (X - min_vals) / range_vals
    return X_normalized, min_vals, max_vals


# ==================== МЕТРИКИ ====================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ==================== ЛИНЕЙНАЯ РЕГРЕССИЯ ====================

class LinearRegression:
    
    def __init__(self, method: Literal['analytical', 'gd', 'sgd'] = 'analytical',
                 learning_rate: float = 0.01, max_iter: int = 1000, 
                 tol: float = 1e-6, random_state: Optional[int] = None,
                 alpha: float = 0.0, regularization: Optional[Literal['L1', 'L2', 'L1L2', 'Lp']] = None,
                 p: float = 2.0):
        
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.alpha = alpha
        self.regularization = regularization
        self.p = p
        
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Добавляем столбец единиц для bias
        X_with_bias = np.column_stack([np.ones(n_samples), X])
        
        if self.method == 'analytical':
            self._fit_analytical(X_with_bias, y)
        elif self.method == 'gd':
            self._fit_gradient_descent(X_with_bias, y)
        elif self.method == 'sgd':
            self._fit_stochastic_gradient_descent(X_with_bias, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def _fit_analytical(self, X: np.ndarray, y: np.ndarray):
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or Inf values")
        
        # Проверка масштаба данных
        X_std = np.std(X, axis=0)
        if np.any(X_std > 100):
            print(f"Warning: Some features have very large std (max: {np.max(X_std):.2f})")
        
        # w = (X^T * X)^(-1) * X^T * y
        XTX = X.T @ X
        
        # Проверка числа обусловленности
        try:
            cond_num = np.linalg.cond(XTX)
            if cond_num > 1e12:
                print(f"Warning: High condition number ({cond_num:.2e}), using stronger regularization")
                default_reg = 1e-3
            else:
                default_reg = 1e-6
        except:
            default_reg = 1e-6
        
        # Регуляризация для аналитического метода
        if self.regularization == 'L2' or (self.regularization is None and self.alpha > 0):
            reg_strength = self.alpha if self.alpha > 0 else default_reg
            XTX += np.eye(XTX.shape[0]) * reg_strength
            XTX[0, 0] -= reg_strength
        elif self.regularization == 'L1' or self.regularization == 'L1L2' or self.regularization == 'Lp':
            XTX += np.eye(XTX.shape[0]) * default_reg
            XTX[0, 0] -= default_reg
        else:   
            XTX += np.eye(XTX.shape[0]) * default_reg
            XTX[0, 0] -= default_reg
        
        XTy = X.T @ y
        try:
            self.weights = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            print("Using pseudoinverse due to LinAlgError")
            self.weights = np.linalg.pinv(XTX) @ XTy
        
        if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
            raise ValueError("Computed weights contain NaN or Inf. Check your data.")
        
        # Проверка масштаба весов
        weights_abs = np.abs(self.weights)
        if np.any(weights_abs > 1e6):
            print(f"Warning: Some weights are very large (max abs: {np.max(weights_abs):.2e})")
        
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
    
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Обучение через градиентный спуск"""
        # Проверка данных
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or Inf values")
        
        n_samples = X.shape[0]
        # Инициализация весов
        self.weights = np.random.randn(X.shape[1]) * 0.01
        
        prev_loss = float('inf')
        
        for i in range(self.max_iter):
            y_pred = X @ self.weights
            
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                print(f"Warning: NaN/Inf in predictions at iteration {i}")
                break
            
            gradient = (2 / n_samples) * X.T @ (y_pred - y)
            
            # Добавляем регуляризацию к градиенту (кроме bias)
            if self.alpha > 0 and self.regularization:
                if self.regularization == 'L1':
                    gradient[1:] += (self.alpha / n_samples) * np.sign(self.weights[1:])
                elif self.regularization == 'L2':
                    gradient[1:] += (2 * self.alpha / n_samples) * self.weights[1:]
                elif self.regularization == 'L1L2':
                    gradient[1:] += (self.alpha / n_samples) * (np.sign(self.weights[1:]) + 2 * self.weights[1:])
                elif self.regularization == 'Lp':
                    gradient[1:] += (self.alpha / n_samples) * self.p * np.sign(self.weights[1:]) * np.abs(self.weights[1:]) ** (self.p - 1)
            elif self.alpha > 0:
                gradient[1:] += (2 * self.alpha / n_samples) * self.weights[1:]
            
            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                print(f"Warning: NaN/Inf in gradient at iteration {i}")
                break
            
            new_weights = self.weights - self.learning_rate * gradient
            
            if np.any(np.isnan(new_weights)) or np.any(np.isinf(new_weights)):
                print(f"Warning: NaN/Inf in weights at iteration {i}")
                break
            
            current_loss = mse(y, y_pred)
            if np.isnan(current_loss) or np.isinf(current_loss):
                print(f"Warning: NaN/Inf in loss at iteration {i}")
                break
            
            if np.linalg.norm(new_weights - self.weights) < self.tol:
                print(f"Converged at iteration {i}")
                break
            
            if current_loss > prev_loss * 10:
                print(f"Warning: Loss diverging at iteration {i}")
                break
            
            self.weights = new_weights
            prev_loss = current_loss
            
            if i % 100 == 0:
                loss = mse(y, y_pred)
                self.loss_history.append(loss)
                if i % 500 == 0:
                    print(f"Iteration {i}, Loss: {loss:.6f}")
        
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
    
    def _fit_stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Обучение через стохастический градиентный спуск"""
        # Проверка данных
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or Inf values")
        
        n_samples = X.shape[0]
        # Инициализация весов
        self.weights = np.random.randn(X.shape[1]) * 0.01
        
        prev_loss = float('inf')
        # Адаптивный learning rate для лучшей сходимости
        current_lr = self.learning_rate
        
        for i in range(self.max_iter):
            # Перемешиваем данные на каждой эпохе
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for j in range(n_samples):
                
                x_sample = X_shuffled[j:j+1]
                y_sample = y_shuffled[j:j+1]
                
                # Предсказание
                y_pred = x_sample @ self.weights
                
                # Проверка на NaN
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    print(f"Warning: NaN/Inf in predictions at epoch {i}, sample {j}")
                    break
                
                if isinstance(y_pred, np.ndarray):
                    error = y_pred[0] - y_sample[0] if isinstance(y_sample, np.ndarray) else y_pred[0] - y_sample
                else:
                    error = y_pred - (y_sample[0] if isinstance(y_sample, np.ndarray) else y_sample)
                    
                x_row = x_sample[0]
                gradient = 2 * x_row * error
                
                # Добавляем регуляризацию к градиенту (кроме bias)
                if self.alpha > 0 and self.regularization:
                    if self.regularization == 'L1':
                        gradient[1:] += self.alpha * np.sign(self.weights[1:])
                    elif self.regularization == 'L2':
                        gradient[1:] += 2 * self.alpha * self.weights[1:]
                    elif self.regularization == 'L1L2':
                        gradient[1:] += self.alpha * (np.sign(self.weights[1:]) + 2 * self.weights[1:])
                    elif self.regularization == 'Lp':
                        gradient[1:] += self.alpha * self.p * np.sign(self.weights[1:]) * np.abs(self.weights[1:]) ** (self.p - 1)
                elif self.alpha > 0:
                    gradient[1:] += 2 * self.alpha * self.weights[1:]
                
                # Проверка градиента
                if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                    print(f"Warning: NaN/Inf in gradient at epoch {i}, sample {j}")
                    break
                
                # Обновление весов с адаптивным learning rate
                new_weights = self.weights - current_lr * gradient
                
                # Проверка на NaN в новых весах
                if np.any(np.isnan(new_weights)) or np.any(np.isinf(new_weights)):
                    print(f"Warning: NaN/Inf in weights at epoch {i}, sample {j}")
                    break
                
                self.weights = new_weights
                if isinstance(y_pred, np.ndarray) and isinstance(y_sample, np.ndarray):
                    epoch_loss += float((y_pred[0] - y_sample[0]) ** 2)
                elif isinstance(y_pred, np.ndarray):
                    epoch_loss += float((y_pred[0] - y_sample) ** 2)
                elif isinstance(y_sample, np.ndarray):
                    epoch_loss += float((y_pred - y_sample[0]) ** 2)
                else:
                    epoch_loss += float((y_pred - y_sample) ** 2)
            
            # Сохранение истории потерь
            avg_loss = epoch_loss / n_samples
            current_loss = avg_loss[0] if isinstance(avg_loss, np.ndarray) else avg_loss
            
            if np.isnan(current_loss) or np.isinf(current_loss):
                print(f"Warning: NaN/Inf in loss at epoch {i}")
                break
            
            self.loss_history.append(current_loss)
            
            # Проверка сходимости
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                print(f"Converged at epoch {i}")
                break
            
            # Проверка на расходимость
            if current_loss > prev_loss * 10:
                print(f"Warning: Loss diverging at epoch {i}")
                break
            
            prev_loss = current_loss
            
            # Адаптивное уменьшение learning rate
            current_lr = self.learning_rate / (1 + 0.01 * i)
            
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {current_loss:.6f}, LR: {current_lr:.6f}")
        
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=np.float64)
        
        predictions = X @ self.weights + self.bias
        
        predictions = np.asarray(predictions, dtype=np.float64)
        
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        
        return predictions


# ==================== КРОСС-ВАЛИДАЦИЯ ====================

def k_fold_cv(model, X: np.ndarray, y: np.ndarray, k: int = 5, 
              metric: callable = mse, random_state: Optional[int] = None) -> Tuple[float, float]:
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k
    
    scores = []
    
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n_samples
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        score = metric(y_val, y_pred)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)


def leave_one_out_cv(model, X: np.ndarray, y: np.ndarray, 
                     metric: callable = mse) -> Tuple[float, float]:
    n_samples = X.shape[0]
    scores = []
    
    for i in range(n_samples):
        train_indices = np.concatenate([np.arange(i), np.arange(i+1, n_samples)])
        val_indices = np.array([i])
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        score = metric(y_val, y_pred)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

