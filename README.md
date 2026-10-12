# Machine Learning Final Project: Dự đoán Hỏng hóc Thiết bị dưới Distribution Shift

## 📌 Bối cảnh dự án
Dự án bảo trì dự đoán (Predictive Maintenance) cho hệ thống máy phay CNC (milling machines) trong nhà máy cơ khí nhằm dự đoán nguy cơ hỏng hóc trong ca kế tiếp (`hong_hoc` $\in \{0, 1\}$).
- **Train (Dây chuyền A):** Nhà máy cũ, vận hành 2 năm ổn định, cảm biến chuẩn.
- **Test (Dây chuyền B):** Nhà máy mới, khí hậu nóng hơn, cảm biến hiệu chỉnh lại, chế độ tải khác $\rightarrow$ **Distribution Shift**.

---

## 📁 Cấu trúc thư mục dự án

```text
final-project/
├── data/
│   ├── raw/                      # Dữ liệu gốc từ Data_Final.zip (train.csv, test.csv)
│   └── processed/                # Dữ liệu sau feature engineering & transform
├── notebooks/
│   └── bai_tap_cuoi_khoa.ipynb   # Jupyter Notebook hoàn chỉnh, chạy hiển thị toàn bộ output
├── src/                          # Mã nguồn module tái sử dụng
│   ├── __init__.py
│   ├── eda.py                   # Hàm vẽ phân phối, thống kê mô tả, correlation heatmap
│   ├── features.py              # Feature engineering (chênh lệch nhiệt độ, công suất cơ, overstrain)
│   ├── drift.py                 # Tính PSI, KS-test, Drift Classifier & Density Ratio Reweighting
│   └── models.py                # Pipeline huấn luyện, RandomizedSearchCV, Threshold Calibration, Metrics
├── reports/
│   ├── figures/                 # Biểu đồ EDA, PR-curve, ROC-curve, Drift feature importances
│   └── bao_cao_tom_tat.md       # Báo cáo tóm tắt cấu trúc theo tiêu chí chấm điểm
└── requirements.txt
```

---

## 🎯 Phân tích & Yêu cầu theo Tiêu chí chấm điểm (10.0 điểm)

1. **Phần 1: EDA (1.0 điểm)**
   - Thống kê mô tả từng feature, tỷ lệ mất cân bằng nhãn (`hong_hoc`: ~7.4% ở Train, ~7.95% ở Test).
   - So sánh trực quan phân phối giữa Dây chuyền A và B; chỉ rõ dấu hiệu covariate shift.
   - Correlation heatmap và nhận diện các biến tương quan với nguy cơ hỏng.

2. **Phần 2: Tiền xử lý & Feature Engineering (1.5 điểm)**
   - Fit scaler/encoder trên Train, transform trên Test (tuyệt đối không fit trên Test).
   - Xử lý mất cân bằng nhãn (`class_weight='balanced'`, SMOTE).
   - Tạo các features kỹ thuật cơ học:
     - `chenh_lech_nhiet_do`: $T_{process} - T_{air}$ (đánh giá tản nhiệt).
     - `cong_suat_co`: $P = \tau \times \omega = \text{momen\_xoan} \times (\text{toc\_do\_quay} \times \frac{2\pi}{60})$ (công suất cơ học).
     - `ap_luc_mai_mon`: $\text{do\_mon\_dao} \times \text{momen\_xoan}$ (đánh giá hao mòn và ứng suất).
     - `ti_le_nhiet_do`: $T_{process} / T_{air}$.

3. **Phần 3: Phát hiện & Xử lý Distribution Shift (2.0 điểm - TRỌNG TÂM)**
   - **Định lượng Shift:**
     - PSI (Population Stability Index) phân loại: $< 0.1$ (ổn định), $0.1 - 0.25$ (shift nhẹ), $> 0.25$ (shift mạnh).
     - KS-Test (Kolmogorov-Smirnov test): thống kê khoảng cách $D$ và $p$-value.
   - **Drift Classifier (Adversarial Validation):**
     - Gán nhãn Train = 0, Test = 1. Huấn luyện phân loại, tính ROC-AUC và feature importance để tìm ra biến "thủ phạm".
   - **Xử lý Shift:**
     - Kỹ thuật 1: **Importance Reweighting (Density Ratio Estimation)**: Trọng số mẫu $w(x) = \frac{P_{test}(x)}{P_{train}(x)} \propto \frac{p(x)}{1 - p(x)}$ với $p(x)$ là xác suất từ Drift Classifier.
     - Kỹ thuật 2: **Threshold Calibration**: Tối ưu ngưỡng quyết định trên validation set nhằm tối đa hóa $F_1$-score.
     - So sánh hiệu năng mô hình trước và sau khi xử lý.

4. **Phần 4: Xây dựng mô hình & Đánh giá (2.0 điểm)**
   - Huấn luyện $\ge 3$ mô hình: Logistic Regression (baseline), Random Forest, LightGBM / XGBoost.
   - Hyperparameter Tuning với `RandomizedSearchCV` + `StratifiedKFold`.
   - Đánh giá đa chiều: ROC-AUC, PR-AUC, F1-score, Precision-Recall curve.
   - Bảng so sánh tổng hợp các mô hình.

5. **Phần 5: Báo cáo & Insight (0.5 điểm) + Kết quả thực thi (3.0 điểm)**
   - Đầy đủ nhận xét, giải thích vật lý/kỹ thuật, kiến nghị vận hành và bảo trì chủ động.
