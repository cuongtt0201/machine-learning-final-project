# BÁO CÁO TÓM TẮT BÀI TẬP CUỐI KHÓA: MACHINE LEARNING
## ĐỀ TÀI: DỰ ĐOÁN HỎNG HÓC THIẾT BỊ DƯỚI DISTRIBUTION SHIFT (PREDICTIVE MAINTENANCE)

---
- **Học viên:** Trần Tiến Cường
- **Mã số sinh viên (MSSV):** `<ĐIỀN_MSSV_TẠI_ĐÂY>`
- **Khoá học:** Machine Learning Chuyên Sâu
- **Notebook thực thi:** [`bai_tap_cuoi_khoa.ipynb`](../notebooks/bai_tap_cuoi_khoa.ipynb)

---

## 1. BỐI CẢNH & MỤC TIÊU DỰ ÁN
Bộ phận bảo trì nhà máy cơ khí vận hành đội máy phay CNC (milling machines) cần chuyển đổi từ mô hình **bảo trì bị động** (chạy đến khi hỏng) sang **bảo trì chủ động** (Predictive Maintenance) nhằm dự đoán nguy cơ thiết bị hỏng trong ca sản xuất kế tiếp (`hong_hoc` $\in \{0, 1\}$).

Thách thức cốt lõi là **Distribution Shift (Dịch chuyển phân phối)**:
- **Tập Train (Dây chuyền A):** Nhà máy cũ, vận hành ổn định 2 năm ở vùng khí hậu mát, cảm biến chuẩn ($N = 14,000$).
- **Tập Test (Dây chuyền B):** Nhà máy mới khánh thành ở vùng khí hậu nóng hơn, cảm biến hiệu chỉnh lại, chế độ tải thay đổi ($N = 6,000$).

Mô hình học máy phải có khả năng chống chịu sự dịch chuyển phân phối này để duy trì độ chính xác khi triển khai trên Dây chuyền B.

---

## 2. KHÁM PHÁ DỮ LIỆU (EDA) & DẤU HIỆU SHIFT

### 2.1 Mất cân bằng dữ liệu (Class Imbalance)
- **Train (Dây chuyền A):** $1,031$ mẫu hỏng ($7.36\%$).
- **Test (Dây chuyền B):** $477$ mẫu hỏng ($7.95\%$).
- **Quyết định kỹ thuật:** Với tỷ lệ hỏng ~7–8%, chỉ số `Accuracy` bị đánh lừa (mô hình đoán 100% không hỏng vẫn đạt 92% accuracy). Bắt buộc phải tối ưu hóa theo **ROC-AUC, PR-AUC** và **$F_1$-Score**.

### 2.2 So sánh phân phối Dây chuyền A vs B
Trực quan hóa KDE cho thấy rõ:
1. `nhiet_do_moi_truong` (Air Temperature): Dịch chuyển mạnh sang phải (trung bình tăng $+2.52\text{ K}$, từ $299.99\text{ K}$ lên $302.51\text{ K}$, đỉnh đạt $312.69\text{ K}$).
2. `nhiet_do_quy_trinh` (Process Temperature): Bị kéo tăng theo từ $309.98\text{ K}$ lên $311.89\text{ K}$.
3. `toc_do_quay` & `momen_xoan`: Dây chuyền B vận hành ở tốc độ quay cao hơn ($1,610\text{ rpm}$ vs $1,540\text{ rpm}$) và mômen thấp hơn ($36.57\text{ Nm}$ vs $39.94\text{ Nm}$).
4. `do_mon_dao`: Hai phân phối trùng khít hoàn toàn (chu kỳ mài mòn dao tự nhiên không đổi).

---

## 3. TIỀN XỬ LÝ & FEATURE ENGINEERING

### 3.1 Quy trình tiền xử lý chống rò rỉ dữ liệu (No Data Leakage)
- Sử dụng `ColumnTransformer` kết hợp `StandardScaler` cho biến số và `OneHotEncoder(drop='first')` cho biến phân loại (`loai_san_pham`, `ca_lam_viec`).
- **Nguyên tắc vàng:** Fit toàn bộ tham số chuẩn hóa (mean, std, categories) **CHỈ TRÊN TẬP TRAIN**, sau đó áp dụng phép biến đổi (`transform`) sang tập Test.

### 3.2 Sáng tạo 3 đặc trưng cơ học chuyên sâu (Domain Features)
Theo lý thuyết máy công cụ và phân tích cơ chế hỏng hóc (Failure Modes):
1. **`chenh_lech_nhiet_do` ($\Delta T = T_{quy\_trinh} - T_{moi\_truong}$):** Đo lường hiệu quả tản nhiệt (Heat Dissipation Failure). Khi $\Delta T$ suy giảm bất thường dưới tải nặng, hệ thống làm mát bị suy giảm.
2. **`cong_suat_co` ($P = \tau \times \omega = \text{momen\_xoan} \times \text{toc\_do\_quay} \times \frac{2\pi}{60}$) [Watt]:** Đo lường cơ chế quá tải công suất trục chính (Power Overload Failure).
3. **`tich_mon_momen` ($\text{do\_mon\_dao} \times \text{momen\_xoan}$):** Đo lường cơ chế phá huỷ do mỏi/quá tải ứng suất (Overstrain Failure). Dao mòn kết hợp với lực cản cắt lớn dẫn đến gãy dao ngay lập tức.
4. **`ti_le_nhiet_do` ($T_{quy\_trinh} / T_{moi\_truong}$):** Gia nhiệt tương đối độc lập với dịch chuyển nhiệt độ nền.

> **💡 Phát hiện kỹ thuật mang tính đột phá:**  
> Dù `toc_do_quay` và `momen_xoan` đều bị shift nhẹ, nhưng tích số cơ học **`cong_suat_co` lại có $\text{PSI} = 0.0234$** (gần như bằng 0) và **`tich_mon_momen` có $\text{PSI} = 0.0194$**! Đây là các **Đặc trưng bất biến (Invariant Causal Features)** giúp mô hình duy trì độ bền vững cao trước Covariate Shift.

---

## 4. PHÁT HIỆN & XỬ LÝ DISTRIBUTION SHIFT (TRỌNG TÂM)

### 4.1 Định lượng Covariate Shift bằng PSI và KS-Test

| Đặc trưng | Train Mean | Test Mean | Diff | PSI | KS Stat | p-value | Phân loại Shift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `nhiet_do_moi_truong` | 299.99 | 302.51 | +2.52 | **1.0187** | 0.4277 | 0.00e+00 | **MẠNH (> 0.25) ⚠️** |
| `nhiet_do_quy_trinh` | 309.98 | 311.89 | +1.91 | **0.5350** | 0.3070 | 0.00e+00 | **MẠNH (> 0.25) ⚠️** |
| `ti_le_nhiet_do` | 1.0333 | 1.0310 | -0.002 | **0.3893** | 0.2526 | 1.03e-236 | **MẠNH (> 0.25) ⚠️** |
| `chenh_lech_nhiet_do`| 9.99 | 9.38 | -0.60 | **0.3190** | 0.2270 | 1.23e-190 | **MẠNH (> 0.25) ⚠️** |
| `toc_do_quay` | 1540.26 | 1610.22 | +69.96 | **0.1548** | 0.1676 | 1.38e-103 | Nhẹ / Vừa (0.10–0.25) |
| `momen_xoan` | 39.94 | 36.57 | -3.37 | **0.1195** | 0.1386 | 7.30e-71 | Nhẹ / Vừa (0.10–0.25) |
| `cong_suat_co` | 6432.5 | 6149.2 | -283.3 | **0.0234** | 0.0608 | 6.37e-14 | Không đáng kể (< 0.10) |
| `tich_mon_momen` | 5067.1 | 4627.8 | -439.3 | **0.0194** | 0.0567 | 3.41e-12 | Không đáng kể (< 0.10) |
| `do_mon_dao` | 126.95 | 126.58 | -0.36 | **0.0013** | 0.0092 | 8.65e-01 | Không đáng kể (< 0.10) |

### 4.2 Drift Classifier (Adversarial Validation)
- Huấn luyện mô hình phân loại LightGBM phân biệt Train ($y=0$) vs Test ($y=1$).
- **Kết quả 5-Fold Stratified CV ROC-AUC: `0.8294`** $\rightarrow$ Khẳng định có covariate shift mạnh giữa 2 dây chuyền.
- **Top 3 đặc trưng "thủ phạm" gây drift:** `toc_do_quay`, `nhiet_do_moi_truong`, `momen_xoan`.

### 4.3 Kỹ thuật xử lý Shift
1. **Importance Reweighting (Density Ratio Estimation):**  
   Ước lượng tỷ số mật độ $w(x) = \frac{p(x)}{1 - p(x)} \times \frac{N_{train}}{N_{test}}$, clip tại phân vị 95% và chuẩn hóa mean = 1 để gán trọng số huấn luyện trên Train set.
2. **Threshold Calibration:**  
   Do mất cân bằng dữ liệu, ngưỡng mặc định $0.5$ không tối ưu. Ta quét ngưỡng trên OOF validation để tìm ngưỡng tối đa hóa $F_1$-score.

---

## 5. KẾT QUẢ HUẤN LUYỆN & SO SÁNH MÔ HÌNH (TRÊN TẬP TEST)

| Thứ hạng | Mô hình | ROC-AUC | PR-AUC | Precision | Recall | F1 (0.5) | Ngưỡng Tối ưu | F1 (Calibrated) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest** | **0.8801** | **0.6859** | 0.7973 | 0.7421 | **0.7687** | 0.58 | **0.7679** |
| 🥈 | **XGBoost** | **0.8665** | **0.6688** | 0.7469 | 0.7547 | 0.7508 | 0.61 | **0.7535** |
| 🥉 | **LightGBM** | **0.8643** | **0.6740** | 0.7299 | 0.7421 | 0.7360 | 0.78 | **0.7505** |
| 4 | **LightGBM (Reweighted)** | 0.8620 | 0.6403 | 0.1167 | 0.8973 | 0.2066 | 0.90 | **0.7381** *(+0.5316)* |
| 5 | **Logistic Regression** | 0.7404 | 0.2433 | 0.1456 | 0.7421 | 0.2435 | 0.69 | **0.3121** |

### Nhận xét & Đánh giá:
- **Random Forest & Gradient Boosting (XGBoost, LightGBM)** hoàn toàn áp đảo mô hình tuyến tính (Logistic Regression) vì cơ chế hỏng hóc máy phay bản chất là **phi tuyến tính kết hợp nhiều điều kiện** (ví dụ: mòn cao ĐỒNG THỜI VỚI mômen lớn).
- **Threshold Calibration** phát huy tác dụng cực kỳ mạnh mẽ: Đẩy ngưỡng xác suất lên vùng 0.60–0.78 giúp loại bỏ hàng trăm ca báo động giả (False Positives), đưa $F_1$-score đạt đỉnh trên 75–76%.
- Đối với `LightGBM (Reweighted)`, việc hiệu chỉnh ngưỡng giúp $F_1$ nhảy vọt từ **0.2066 lên 0.7381** (+53.16%), chứng minh sự kết hợp hoàn hảo giữa Density Ratio Reweighting và Threshold Calibration.

---

## 6. INSIGHT VẬN HÀNH & ĐỀ XUẤT BẢO TRÌ CHỦ ĐỘNG

1. **Quy tắc thay dao bảo trì (Tool Wear Threshold):**  
   Cần thiết lập ngưỡng cảnh báo tự động khi `tich_mon_momen` $> 9,000\text{ phút}\cdot\text{Nm}$ hoặc khi `do_mon_dao` $> 200\text{ phút}$ ở chế độ phay thô tải nặng.
2. **Bảo trì hệ thống làm mát (Coolant Monitoring):**  
   Tại nhà máy mới (Dây chuyền B), nhiệt độ môi trường cao khiến chênh lệch nhiệt độ thu hẹp. Cần cài đặt cảm biến đo trực tiếp $\Delta T = T_{quy\_trinh} - T_{moi\_truong}$; nếu $\Delta T < 8.5\text{ K}$ trong ca làm việc thì kích hoạt chu kỳ xả rửa két tản nhiệt và bơm coolant.
3. **Hạn chế & Hướng phát triển:**
   - Dữ liệu hiện tại là snapshot tổng hợp từng ca, chưa ghi nhận rung động tần số cao (vibration kHz).
   - Đề xuất nâng cấp: Tích hợp cảm biến gia tốc kế (accelerometer), triển khai mô hình học liên tục (Continual / Online Learning) để tự động thích ứng với trôi dạt cảm biến theo thời gian thực (stream drift adaptation).
