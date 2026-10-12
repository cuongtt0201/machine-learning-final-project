"""
Module: models.py
Huấn luyện, tinh chỉnh siêu tham số và đánh giá mô hình học máy:
- Logistic Regression, Random Forest, LightGBM, XGBoost
- RandomizedSearchCV + StratifiedKFold
- Threshold Calibration (tìm ngưỡng tối ưu cho F1-score)
- Đánh giá đa chiều: ROC-AUC, PR-AUC, F1-score, Precision-Recall curve
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, num_steps: int = 100) -> tuple:
    """
    Tìm ngưỡng xác suất tối ưu nhằm cực đại hóa F1-score trên tập xác thực.
    """
    thresholds = np.linspace(0.05, 0.95, num_steps)
    best_thresh = 0.5
    best_f1 = -1.0
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
            
    return float(best_thresh), float(best_f1)


def evaluate_model(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Đánh giá mô hình toàn diện theo các độ đo phân loại nhị phân.
    """
    y_preds = (y_probs >= threshold).astype(int)
    
    return {
        'ROC-AUC': roc_auc_score(y_true, y_probs),
        'PR-AUC': average_precision_score(y_true, y_probs),
        'F1-Score': f1_score(y_true, y_preds, zero_division=0),
        'Precision': precision_score(y_true, y_preds, zero_division=0),
        'Recall': recall_score(y_true, y_preds, zero_division=0),
        'Threshold': threshold
    }


def get_model_search_spaces(scale_pos_weight: float = 12.0):
    """
    Định nghĩa không gian siêu tham số cho RandomizedSearchCV.
    """
    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs']
            }
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            {
                'n_estimators': [100, 150, 200],
                'max_depth': [6, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        ),
        'LightGBM': (
            lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1),
            {
                'n_estimators': [80, 120, 180],
                'learning_rate': [0.03, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'num_leaves': [15, 31, 63],
                'subsample': [0.7, 0.85, 1.0],
                'colsample_bytree': [0.7, 0.85, 1.0]
            }
        ),
        'XGBoost': (
            XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss'),
            {
                'n_estimators': [80, 120, 180],
                'learning_rate': [0.03, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.85, 1.0],
                'colsample_bytree': [0.7, 0.85, 1.0]
            }
        )
    }
    return models
