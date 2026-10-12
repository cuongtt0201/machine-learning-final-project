"""
Module: features.py
Trích chọn và tiền xử lý đặc trưng (Feature Engineering & Preprocessing)
Tuân thủ nguyên tắc: Fit scaler và encoder trên Train, transform trên Test (tránh rò rỉ dữ liệu).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các biến đặc trưng kỹ thuật cơ học:
    1. chenh_lech_nhiet_do: Chênh lệch nhiệt độ (T_process - T_air) đo khả năng tản nhiệt (Heat Dissipation).
    2. cong_suat_co: Công suất cơ học P = tau * omega = momen_xoan * (toc_do_quay * 2*pi/60).
    3. tich_mon_momen: Tích mòn x mômen (Tool Wear x Torque) phát hiện cơ chế quá tải ứng suất (Overstrain).
    4. ti_le_nhiet_do: Tỷ lệ nhiệt độ T_process / T_air.
    5. toc_do_goc: Tốc độ góc (rad/s) = toc_do_quay * 2*pi/60.
    """
    df_feat = df.copy()
    
    # 1. Cơ chế tản nhiệt (Heat Dissipation Failure)
    df_feat['chenh_lech_nhiet_do'] = df_feat['nhiet_do_quy_trinh'] - df_feat['nhiet_do_moi_truong']
    
    # 2. Cơ chế công suất cơ (Power Overload Failure)
    omega = df_feat['toc_do_quay'] * (2.0 * np.pi / 60.0)
    df_feat['toc_do_goc'] = omega
    df_feat['cong_suat_co'] = df_feat['momen_xoan'] * omega
    
    # 3. Cơ chế ứng suất mài mòn (Overstrain / Tool Wear Failure)
    df_feat['tich_mon_momen'] = df_feat['do_mon_dao'] * df_feat['momen_xoan']
    
    # 4. Tỷ lệ nhiệt độ tương đối
    df_feat['ti_le_nhiet_do'] = df_feat['nhiet_do_quy_trinh'] / df_feat['nhiet_do_moi_truong']
    
    return df_feat


class DataPipeline:
    """
    Pipeline tiền xử lý dữ liệu chuẩn cho bài toán Machine Learning.
    - Categorical Encoding (OneHotEncoder)
    - Feature Scaling (StandardScaler)
    - Đảm bảo fit CHỈ trên tập Train, sau đó transform cả Train và Test.
    """
    def __init__(self, cat_cols=None, num_cols=None):
        self.cat_cols = cat_cols or ['loai_san_pham', 'ca_lam_viec']
        self.num_cols = num_cols
        self.preprocessor = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame):
        X_eng = add_domain_features(X)
        if self.num_cols is None:
            self.num_cols = [c for c in X_eng.columns if c not in self.cat_cols and c != 'hong_hoc']
            
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), self.cat_cols)
            ]
        )
        self.preprocessor.fit(X_eng)
        
        # Lấy tên các cột sau khi biến đổi
        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + encoded_cat_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_eng = add_domain_features(X)
        transformed = self.preprocessor.transform(X_eng)
        return pd.DataFrame(transformed, columns=self.feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
