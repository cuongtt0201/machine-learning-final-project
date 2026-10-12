"""
Module: drift.py
Phát hiện & Xử lý Distribution Shift:
1. Tính PSI (Population Stability Index) & KS-Test (Kolmogorov-Smirnov)
2. Huấn luyện Drift Classifier (Adversarial Validation)
3. Xử lý shift bằng Importance Reweighting (Density Ratio Estimation)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Tính Population Stability Index (PSI) giữa tập expected (Train) và actual (Test).
    - PSI < 0.1: Không có shift (Ổn định)
    - 0.1 <= PSI <= 0.25: Shift nhẹ/vừa (Moderate shift)
    - PSI > 0.25: Shift mạnh (Significant shift)
    """
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)
    
    if len(bin_edges) <= 1:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def evaluate_feature_drift(df_train: pd.DataFrame, df_test: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Đánh giá mức độ distribution shift cho danh sách các biến số bằng PSI và KS-Test.
    """
    results = []
    for col in num_cols:
        train_vals = df_train[col].dropna().values
        test_vals = df_test[col].dropna().values
        
        psi = calculate_psi(train_vals, test_vals)
        ks_stat, p_val = stats.ks_2samp(train_vals, test_vals)
        
        if psi < 0.10:
            level = "Không đáng kể (< 0.10)"
        elif psi <= 0.25:
            level = "Nhẹ / Vừa (0.10 - 0.25)"
        else:
            level = "Mạnh (> 0.25)"
            
        results.append({
            'Feature': col,
            'Train Mean': train_vals.mean(),
            'Test Mean': test_vals.mean(),
            'Diff Mean': test_vals.mean() - train_vals.mean(),
            'PSI': psi,
            'KS Statistic': ks_stat,
            'p-value': p_val,
            'Mức độ Shift': level
        })
    return pd.DataFrame(results).sort_values(by='PSI', ascending=False).reset_index(drop=True)


def train_drift_classifier(X_train: pd.DataFrame, X_test: pd.DataFrame, n_splits: int = 5):
    """
    Huấn luyện Drift Classifier phân biệt dữ liệu Train (nhãn 0) vs Test (nhãn 1).
    Trả về:
    - roc_auc: Điểm ROC-AUC trung bình qua K-Fold CV
    - feature_importance: Bảng tầm quan trọng của các feature phân biệt Train vs Test
    - oof_probs: Xác suất dự đoán mẫu thuộc Test set (dùng cho Importance Reweighting)
    """
    X_tr_copy = X_train.copy()
    X_te_copy = X_test.copy()
    
    X_combined = pd.concat([X_tr_copy, X_te_copy], axis=0).reset_index(drop=True)
    y_drift = np.array([0] * len(X_tr_copy) + [1] * len(X_te_copy))
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X_combined))
    
    feature_importances = np.zeros(X_combined.shape[1])
    
    for tr_idx, val_idx in cv.split(X_combined, y_drift):
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1
        )
        clf.fit(X_combined.iloc[tr_idx], y_drift[tr_idx])
        oof_probs[val_idx] = clf.predict_proba(X_combined.iloc[val_idx])[:, 1]
        feature_importances += clf.feature_importances_ / n_splits
        
    roc_auc = stats_auc = cross_val_score(
        lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1),
        X_combined, y_drift, cv=cv, scoring='roc_auc'
    ).mean()
    
    fi_df = pd.DataFrame({
        'Feature': X_combined.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return roc_auc, fi_df, oof_probs


def compute_density_ratio_weights(oof_probs: np.ndarray, n_train: int, n_test: int, clip_percentile: float = 95.0) -> np.ndarray:
    """
    Tính trọng số Importance Reweighting cho các mẫu tập Train dựa trên mật độ phân phối:
    w(x) = [P(Test|x) / P(Train|x)] * [P(Train) / P(Test)]
    Clip trọng số tại phân vị cao để kiểm soát phương sai mô hình.
    """
    p_train = oof_probs[:n_train]
    p_train = np.clip(p_train, 1e-4, 0.99)
    
    prior_ratio = float(n_train) / float(n_test)
    raw_weights = (p_train / (1.0 - p_train)) * prior_ratio
    
    # Clip trọng số tránh outlier
    lower_bound = np.percentile(raw_weights, 1.0)
    upper_bound = np.percentile(raw_weights, clip_percentile)
    clipped_weights = np.clip(raw_weights, lower_bound, upper_bound)
    
    # Chuẩn hoá về mean = 1
    normalized_weights = clipped_weights / np.mean(clipped_weights)
    return normalized_weights
