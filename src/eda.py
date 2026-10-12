"""
Module: eda.py
Hàm trực quan hóa dữ liệu và xuất biểu đồ phục vụ báo cáo:
- So sánh phân phối Dây chuyền A (Train) vs Dây chuyền B (Test)
- Correlation heatmap
- Precision-Recall Curves, ROC Curves
- Tầm quan trọng của đặc trưng gây Drift
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Thiết lập phong cách hiển thị biểu đồ hiện đại, chuyên nghiệp
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_target_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, save_path: str = None):
    """Vẽ phân bố biến mục tiêu hong_hoc giữa 2 tập."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, (df, name) in zip(axes, [(train_df, 'Train (Dây chuyền A)'), (test_df, 'Test (Dây chuyền B)')]):
        counts = df['hong_hoc'].value_counts()
        pcts = df['hong_hoc'].value_counts(normalize=True) * 100
        bars = ax.bar(['Bình thường (0)', 'Hỏng hóc (1)'], counts.values, color=['#3b82f6', '#ef4444'], width=0.5)
        ax.set_title(f'{name}\nTỷ lệ hỏng: {pcts[1]:.2f}%', fontsize=11, fontweight='bold')
        ax.set_ylabel('Số lượng mẫu')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_drift_distributions(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, save_path: str = None):
    """So sánh trực quan phân phối các đặc trưng giữa Train và Test."""
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        ax = axes[i]
        sns.kdeplot(train_df[feat], ax=ax, label='Train (Dây chuyền A)', color='#2563eb', fill=True, alpha=0.3, linewidth=2)
        sns.kdeplot(test_df[feat], ax=ax, label='Test (Dây chuyền B)', color='#dc2626', fill=True, alpha=0.3, linewidth=2)
        ax.set_title(f'Phân phối: {feat}', fontsize=11, fontweight='bold')
        ax.set_xlabel(feat)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8, loc='best')
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, num_features: list, save_path: str = None):
    """Vẽ correlation heatmap giữa các biến số và biến mục tiêu."""
    plt.figure(figsize=(9, 7))
    cols = num_features + ['hong_hoc']
    corr = df[cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'shrink': 0.8}, mask=mask)
    plt.title('Ma trận tương quan (Correlation Matrix) với Nhãn Hỏng hóc', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_and_roc_curves(y_true: np.ndarray, model_probs: dict, save_path: str = None):
    """Vẽ đường cong Precision-Recall và ROC cho nhiều mô hình trên cùng 1 biểu đồ."""
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    colors = ['#2563eb', '#16a34a', '#d97706', '#9333ea', '#dc2626']
    
    # 1. PR Curve
    no_skill = np.sum(y_true) / len(y_true)
    ax_pr.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label=f'Baseline (No Skill: {no_skill:.3f})')
    
    for (name, probs), color in zip(model_probs.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, color=color, lw=2, label=f'{name} (PR-AUC = {pr_auc:.3f})')
        
    ax_pr.set_xlabel('Recall', fontsize=10, fontweight='bold')
    ax_pr.set_ylabel('Precision', fontsize=10, fontweight='bold')
    ax_pr.set_title('Precision-Recall Curve (Trọng tâm cho Imbalance Data)', fontsize=11, fontweight='bold')
    ax_pr.legend(loc='best', fontsize=9)
    ax_pr.set_ylim([-0.02, 1.05])
    
    # 2. ROC Curve
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (AUC = 0.500)')
    for (name, probs), color in zip(model_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{name} (ROC-AUC = {roc_auc:.3f})')
        
    ax_roc.set_xlabel('False Positive Rate (FPR)', fontsize=10, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=10, fontweight='bold')
    ax_roc.set_title('Receiver Operating Characteristic (ROC Curve)', fontsize=11, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=9)
    ax_roc.set_ylim([-0.02, 1.05])
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
