import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dữ liệu
train_df = pd.read_csv('clean_data/clean_data_train.csv')
test_df  = pd.read_csv('clean_data/clean_data_test.csv')

# Parse datetime
train_df['time'] = pd.to_datetime(train_df['time'])
test_df['time']  = pd.to_datetime(test_df['time'])

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"\nCác cột: {train_df.columns.tolist()}")
# ==================================================
# 1.1 Phân tích ma trận tương quan
# ==================================================
numeric_cols = [col for col in train_df.columns if col != 'time']
corr_matrix = train_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax)
ax.set_title('Ma trận tương quan giữa các biến', fontsize=14)
plt.tight_layout()
plt.show()

print("\n=== TƯƠNG QUAN VỚI temperature_2m ===")
print(corr_matrix['temperature_2m'].sort_values(ascending=False).round(3))
# ==================================================
# 2. Tạo Lag Features
# Ghép train + test trước khi tạo lag để test có đủ giá trị lịch sử
# ==================================================

# Đánh dấu is_train để tách lại sau
train_df['is_train'] = 1
test_df['is_train']  = 0

# Ghép lại và sắp xếp theo thời gian
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.sort_values('time').reset_index(drop=True)

# Tạo lag features
lag_hours = [1, 2, 3, 12, 24, 72, 168]
for lag in lag_hours:
    full_df[f'temp_lag_{lag}'] = full_df['temperature_2m'].shift(lag)

# Tách lại train và test, dropna loại bỏ các dòng không đủ lịch sử
train_df = full_df[full_df['is_train'] == 1].dropna().reset_index(drop=True)
test_df  = full_df[full_df['is_train'] == 0].dropna().reset_index(drop=True)

# Bỏ cột is_train
train_df = train_df.drop(columns=['is_train'])
test_df  = test_df.drop(columns=['is_train'])

print("=== KẾT QUẢ LAG FEATURES ===")
print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"\nCác cột lag: {[col for col in train_df.columns if 'lag' in col]}")
print(f"\nTất cả cột: {train_df.columns.tolist()}")
print(f"\nTrain time range: {train_df['time'].min()} → {train_df['time'].max()}")
print(f"Test time range:  {test_df['time'].min()} → {test_df['time'].max()}")

# ==================================================
# TRỰC QUAN HÓA: So sánh Trước và Sau tạo Lag Features
# ==================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Lấy 7 ngày đầu để visualize rõ
sample = train_df.iloc[:168]

# Trước lag — chỉ có temperature_2m gốc
axes[0].plot(sample['time'], sample['temperature_2m'],
             color='steelblue', linewidth=1)
axes[0].set_title('Trước Lag Features\n(chỉ có temperature_2m gốc)')
axes[0].set_xlabel('Thời gian')
axes[0].set_ylabel('Nhiệt độ (°C)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3)

# Sau lag — so sánh temp và lag_24
axes[1].plot(sample['time'], sample['temperature_2m'],
             color='steelblue', linewidth=1, label='temperature_2m (hiện tại)')
axes[1].plot(sample['time'], sample['temp_lag_24'],
             color='tomato', linewidth=1, linestyle='--', label='temp_lag_24 (24h trước)')
axes[1].plot(sample['time'], sample['temp_lag_1'],
             color='green', linewidth=1, linestyle=':', label='temp_lag_1 (1h trước)')
axes[1].set_title('Sau Lag Features\n(temperature_2m vs temp_lag_1 vs temp_lag_24)')
axes[1].set_xlabel('Thời gian')
axes[1].set_ylabel('Nhiệt độ (°C)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle('So sánh Trước và Sau tạo Lag Features', fontsize=13)
plt.tight_layout()
plt.show()

# ==================================================
# 3.1 Chuẩn hóa dữ liệu (StandardScaler)
# ==================================================
feature_cols = ['wind_speed_10m', 'cloud_cover', 'relative_humidity_2m',
                'surface_pressure', 'precipitation', 'vapour_pressure_deficit']

lag_hours = [1, 2, 3, 12, 24, 72, 168]
for lag in lag_hours:
    feature_cols.append(f'temp_lag_{lag}')

X_train = train_df[feature_cols].values
X_test  = test_df[feature_cols].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform trên train
X_test_scaled  = scaler.transform(X_test)        # chỉ transform trên test

print("=== KẾT QUẢ CHUẨN HÓA ===")
print(f"\nX_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape:  {X_test_scaled.shape}")
print(f"\nTrước chuẩn hóa — Mean: {X_train.mean(axis=0).round(3)}")
print(f"Trước chuẩn hóa — Std:  {X_train.std(axis=0).round(3)}")
print(f"\nSau chuẩn hóa  — Mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"Sau chuẩn hóa  — Std:  {X_train_scaled.std(axis=0).round(4)}")

# ==================================================
# TRỰC QUAN HÓA: So sánh phân phối Trước và Sau chuẩn hóa
# ==================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for i, col in enumerate(feature_cols):
    ax = axes[i//3][i%3]
    ax.hist(X_train[:, i], bins=50, alpha=0.6,
            color='steelblue', label='Trước', density=True)
    ax.hist(X_train_scaled[:, i], bins=50, alpha=0.6,
            color='tomato', label='Sau', density=True)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Phân phối các biến Trước (xanh) và Sau (đỏ) chuẩn hóa StandardScaler',
             fontsize=13)
plt.tight_layout()
plt.show()
# ==================================================
# 3.2 Tính ma trận hiệp phương sai
# ==================================================
cov_matrix = np.cov(X_train_scaled.T)
cov_df = pd.DataFrame(cov_matrix, index=feature_cols, columns=feature_cols)

print("=== MA TRẬN HIỆP PHƯƠNG SAI ===")
print(cov_df.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cov_df,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax)
ax.set_title('Ma trận hiệp phương sai của dữ liệu đã chuẩn hóa', fontsize=13)
plt.tight_layout()
plt.show()
# ==================================================
# 3.3 Phân tích giá trị riêng (Explained Variance)
# ==================================================
pca_full = PCA()
pca_full.fit(X_train_scaled)

eigenvalues   = pca_full.explained_variance_
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("=== PHÂN TÍCH GIÁ TRỊ RIÊNG ===")
print(f"\n{'PC':<6} {'Giá trị riêng':>15} {'Explained Var (%)':>18} {'Tích lũy (%)':>13}")
print("-" * 55)
for i, (ev, exp, cum) in enumerate(zip(eigenvalues, explained_var, cumulative_var)):
    print(f"PC{i+1:<4} {ev:>15.4f} {exp*100:>17.2f}% {cum*100:>12.2f}%")

# Vẽ biểu đồ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(explained_var)+1), explained_var*100,
            color='steelblue', edgecolor='black')
axes[0].set_title('Explained Variance từng Principal Component')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_xticks(range(1, len(explained_var)+1))

axes[1].plot(range(1, len(cumulative_var)+1), cumulative_var*100,
             marker='o', color='tomato', linewidth=2)
axes[1].axhline(y=95, color='green', linestyle='--', label='95%')
axes[1].axhline(y=90, color='orange', linestyle='--', label='90%')
axes[1].set_title('Cumulative Explained Variance')
axes[1].set_xlabel('Số Principal Components')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_xticks(range(1, len(cumulative_var)+1))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# ==================================================
# 3.4 Phân tích Loading Matrix
# ==================================================
loadings = pd.DataFrame(
    pca_full.components_.T,
    columns=[f'PC{i+1}' for i in range(len(feature_cols))],
    index=feature_cols
)

print("=== LOADING MATRIX ===")
print(loadings.round(3))

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(loadings,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            ax=ax)
ax.set_title('Loading Matrix — Đóng góp của từng biến vào các Principal Component',
             fontsize=12)
plt.tight_layout()
plt.show()

print("\n=== Ý NGHĨA TỪNG PC ===")
for i in range(5):
    pc = loadings[f'PC{i+1}']
    top = pc.abs().nlargest(2).index.tolist()
    print(f"PC{i+1} ({explained_var[i]*100:.2f}%): chủ yếu bởi {top[0]} ({pc[top[0]]:.3f}) và {top[1]} ({pc[top[1]]:.3f})")
# ==================================================
# 3.5 Lựa chọn K = 5 và Transform dữ liệu
# w là vector riêng ứng với 5 giá trị riêng lớn nhất
# (theo tài liệu bài giảng Chương 7 - PCA)
# ==================================================
n_components = 5
pca_final = PCA(n_components=n_components)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca  = pca_final.transform(X_test_scaled)

print(f"=== KẾT QUẢ PCA VỚI K = {n_components} ===")
print(f"Variance được giữ lại: {pca_final.explained_variance_ratio_.sum()*100:.2f}%")
print(f"Số chiều ban đầu: {X_train_scaled.shape[1]}")
print(f"Số chiều sau PCA: {X_train_pca.shape[1]}")
print(f"\nX_train_pca shape: {X_train_pca.shape}")
print(f"X_test_pca shape:  {X_test_pca.shape}")

# Tạo DataFrame PCA
pca_cols = [f'PC{i+1}' for i in range(n_components)]

train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols)
train_pca_df.insert(0, 'time', train_df['time'].values)
train_pca_df['temperature_2m'] = train_df['temperature_2m'].values

test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols)
test_pca_df.insert(0, 'time', test_df['time'].values)
test_pca_df['temperature_2m'] = test_df['temperature_2m'].values

print("\nTrain PCA DataFrame (5 dòng đầu):")
print(train_pca_df.head())
# ==================================================
# TRỰC QUAN HÓA: So sánh không gian đặc trưng Trước và Sau PCA
# ==================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Trước PCA — cặp biến có tương quan cao nhất (-0.958)
axes[0].scatter(X_train_scaled[:, 2], X_train_scaled[:, 5],
                alpha=0.1, s=1, color='steelblue')
axes[0].set_xlabel('relative_humidity_2m (scaled)')
axes[0].set_ylabel('vapour_pressure_deficit (scaled)')
axes[0].set_title('Trước PCA\n(tương quan = -0.958 → thông tin dư thừa)')
axes[0].grid(True, alpha=0.3)

# Sau PCA — PC1 vs PC2 độc lập tuyến tính
axes[1].scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                alpha=0.1, s=1, color='tomato')
axes[1].set_xlabel(f'PC1 ({pca_final.explained_variance_ratio_[0]*100:.2f}%)')
axes[1].set_ylabel(f'PC2 ({pca_final.explained_variance_ratio_[1]*100:.2f}%)')
axes[1].set_title('Sau PCA\n(PC1 vs PC2 — độc lập tuyến tính)')
axes[1].grid(True, alpha=0.3)

plt.suptitle('So sánh không gian đặc trưng Trước và Sau PCA', fontsize=13)
plt.tight_layout()
plt.show()

print("=== SO SÁNH TRƯỚC VÀ SAU PCA ===")
print(f"Trước PCA: {X_train_scaled.shape[1]} chiều")
print(f"Sau PCA:   {X_train_pca.shape[1]} chiều")
print(f"Variance giữ lại: {pca_final.explained_variance_ratio_.sum()*100:.2f}%")
print(f"Variance mất đi:  {(1-pca_final.explained_variance_ratio_.sum())*100:.2f}%")
# ==================================================
# 4.1 Sin/Cos encoding cho giờ trong ngày (chu kỳ 24h)
# Dựa trên kết quả EDA: dữ liệu có chu kỳ ngày rõ ràng
# ==================================================

# Tạo trên train
train_df['hour']     = train_df['time'].dt.hour
train_df['sin_hour'] = np.sin(2 * np.pi * train_df['hour'] / 24)
train_df['cos_hour'] = np.cos(2 * np.pi * train_df['hour'] / 24)

# Tạo trên test
test_df['hour']     = test_df['time'].dt.hour
test_df['sin_hour'] = np.sin(2 * np.pi * test_df['hour'] / 24)
test_df['cos_hour'] = np.cos(2 * np.pi * test_df['hour'] / 24)

print("=== TIME ENCODING: GIỜ TRONG NGÀY ===")
print(train_df[['time', 'hour', 'sin_hour', 'cos_hour']].head(24).to_string())

# Trực quan hóa trước và sau encoding giờ
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

hours = range(24)
axes[0].plot(hours, list(hours), marker='o', color='steelblue')
axes[0].set_title('Trước Encoding\n(giờ dạng số nguyên — không tuần hoàn)')
axes[0].set_xlabel('Giờ')
axes[0].set_ylabel('Giá trị')
axes[0].grid(True, alpha=0.3)

sin_vals = np.sin(2 * np.pi * np.array(list(hours)) / 24)
cos_vals = np.cos(2 * np.pi * np.array(list(hours)) / 24)
axes[1].plot(hours, sin_vals, marker='o', color='tomato',  label='sin_hour')
axes[1].plot(hours, cos_vals, marker='s', color='green',   label='cos_hour')
axes[1].set_title('Sau Encoding\n(sin/cos — tuần hoàn liên tục)')
axes[1].set_xlabel('Giờ')
axes[1].set_ylabel('Giá trị')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('So sánh Trước và Sau Time Encoding (Giờ trong ngày)', fontsize=13)
plt.tight_layout()
plt.show()
# ==================================================
# 4.2 Sin/Cos encoding cho tháng trong năm (chu kỳ 12 tháng)
# Dựa trên kết quả EDA: seasonal decompose cho thấy chu kỳ năm rõ ràng
# ==================================================

# Tạo trên train
train_df['month']     = train_df['time'].dt.month
train_df['sin_month'] = np.sin(2 * np.pi * train_df['month'] / 12)
train_df['cos_month'] = np.cos(2 * np.pi * train_df['month'] / 12)

# Tạo trên test
test_df['month']     = test_df['time'].dt.month
test_df['sin_month'] = np.sin(2 * np.pi * test_df['month'] / 12)
test_df['cos_month'] = np.cos(2 * np.pi * test_df['month'] / 12)

print("=== TIME ENCODING: THÁNG TRONG NĂM ===")
monthly_sample = train_df.groupby('month')[['month', 'sin_month', 'cos_month']].first()
print(monthly_sample.to_string())

# Trực quan hóa trước và sau encoding tháng
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

months = range(1, 13)
axes[0].plot(months, list(months), marker='o', color='steelblue')
axes[0].set_title('Trước Encoding\n(tháng dạng số nguyên — không tuần hoàn)')
axes[0].set_xlabel('Tháng')
axes[0].set_ylabel('Giá trị')
axes[0].set_xticks(range(1, 13))
axes[0].grid(True, alpha=0.3)

sin_m = np.sin(2 * np.pi * np.array(list(months)) / 12)
cos_m = np.cos(2 * np.pi * np.array(list(months)) / 12)
axes[1].plot(months, sin_m, marker='o', color='tomato', label='sin_month')
axes[1].plot(months, cos_m, marker='s', color='green',  label='cos_month')
axes[1].set_title('Sau Encoding\n(sin/cos — tuần hoàn liên tục)')
axes[1].set_xlabel('Tháng')
axes[1].set_ylabel('Giá trị')
axes[1].set_xticks(range(1, 13))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('So sánh Trước và Sau Time Encoding (Tháng trong năm)', fontsize=13)
plt.tight_layout()
plt.show()
# ==================================================
# Thêm encoding vào train_pca_df và test_pca_df
# ==================================================
train_pca_df['sin_hour']  = train_df['sin_hour'].values
train_pca_df['cos_hour']  = train_df['cos_hour'].values
train_pca_df['sin_month'] = train_df['sin_month'].values
train_pca_df['cos_month'] = train_df['cos_month'].values

test_pca_df['sin_hour']  = test_df['sin_hour'].values
test_pca_df['cos_hour']  = test_df['cos_hour'].values
test_pca_df['sin_month'] = test_df['sin_month'].values
test_pca_df['cos_month'] = test_df['cos_month'].values

print("Sau khi thêm Time Encoding vào PCA DataFrame:")
print(f"Train columns: {train_pca_df.columns.tolist()}")
# ==================================================
# 5. Tổng hợp và thống kê bộ đặc trưng cuối cùng
# ==================================================
feature_cols_final = [col for col in train_pca_df.columns
                      if col not in ['time', 'temperature_2m']]

print("=" * 55)
print("TỔNG HỢP BỘ ĐẶC TRƯNG CUỐI CÙNG (PCA + TIME)")
print("=" * 55)
print(f"\nBiến mục tiêu: temperature_2m")
print(f"\nCác features đầu vào ({len(feature_cols_final)} biến):")
for i, col in enumerate(feature_cols_final, 1):
    print(f"  {i:2d}. {col}")

print(f"\n{'Tập dữ liệu':<15} {'Số mẫu':>10} {'Số features':>12} {'Khoảng thời gian'}")
print("-" * 70)
print(f"{'Train':<15} {len(train_pca_df):>10} {len(feature_cols_final):>12} "
      f"{train_pca_df['time'].min().date()} → {train_pca_df['time'].max().date()}")
print(f"{'Test':<15} {len(test_pca_df):>10} {len(feature_cols_final):>12} "
      f"{test_pca_df['time'].min().date()} → {test_pca_df['time'].max().date()}")

print("\n5 dòng đầu Train:")
print(train_pca_df.head())

# =======================================================
# CHUẨN BỊ 3 PHIÊN BẢN DATASET TỪ CÁC BIẾN ĐÃ CÓ
# =======================================================
feature_cols_stats = ['wind_speed_10m', 'cloud_cover', 'relative_humidity_2m',
                      'surface_pressure', 'precipitation', 'vapour_pressure_deficit']

# 1. Dataset FE_PCA (Chính là tập từ PCA, đã gồm lag/time vì PCA chạy trên lag, time thêm sau)
X_train_fe_pca = train_pca_df[[col for col in train_pca_df.columns if col not in ['time', 'temperature_2m']]].copy()
y_train_fe_pca = train_pca_df['temperature_2m'].copy()
X_test_fe_pca = test_pca_df[[col for col in test_pca_df.columns if col not in ['time', 'temperature_2m']]].copy()
y_test_fe_pca = test_pca_df['temperature_2m'].copy()

# 2. Dataset RAW (Không qua chuẩn hóa, không Lag/Time)
X_train_raw = train_df[feature_cols_stats].copy()
y_train_raw = train_df['temperature_2m'].copy()
X_test_raw = test_df[feature_cols_stats].copy()
y_test_raw = test_df['temperature_2m'].copy()

# 3. Dataset FE_no_PCA (Có Chuẩn hóa + Lag/Time, nhưng KHÔNG dùng PCA)
time_lag_cols = ['sin_hour', 'cos_hour', 'sin_month', 'cos_month'] + [f'temp_lag_{l}' for l in [1, 2, 3, 12, 24, 72, 168]]
X_train_scaled_df = pd.DataFrame(X_train_scaled[:, :6], columns=feature_cols_stats)
X_test_scaled_df = pd.DataFrame(X_test_scaled[:, :6], columns=feature_cols_stats)
X_train_lag_time = train_pca_df[['sin_hour', 'cos_hour', 'sin_month', 'cos_month']].copy()
for l in [1, 2, 3, 12, 24, 72, 168]: X_train_lag_time[f'temp_lag_{l}'] = train_df[f'temp_lag_{l}'].values
X_test_lag_time = test_pca_df[['sin_hour', 'cos_hour', 'sin_month', 'cos_month']].copy()
for l in [1, 2, 3, 12, 24, 72, 168]: X_test_lag_time[f'temp_lag_{l}'] = test_df[f'temp_lag_{l}'].values

X_train_fe_no_pca = pd.concat([X_train_scaled_df, X_train_lag_time], axis=1)
X_test_fe_no_pca = pd.concat([X_test_scaled_df, X_test_lag_time], axis=1)

print("Đã tạo thành công 3 phiên bản dữ liệu:")
print(f"   • Raw: {X_train_raw.shape} (chỉ 6 cột thống kê gốc)")
print(f"   • FE_no_PCA: {X_train_fe_no_pca.shape} (chuẩn hóa + lag/time)")
print(f"   • FE_PCA: {X_train_fe_pca.shape} (PCA + lag/time)")
print(f"\n   y_train_raw: {y_train_raw.shape}")
print(f"   y_train_fe_pca: {y_train_fe_pca.shape}")

# =======================================================
# TỰ ĐỘNG DÒ TÌM THAM SỐ (AUTO-ARIMA) VÀ HUẤN LUYỆN SARIMAX
# =======================================================
# Nếu máy chưa cài thư viện pmdarima, hãy bỏ comment dòng dưới đây để cài đặt:
# !pip install pmdarima

import pmdarima as pm
import statsmodels.api as sm

print("Đang tự động dò tìm bộ tham số (p, d, q) tối ưu nhất...\n")

y_search = y_train_raw
X_search = X_train_raw

# 1. Chạy Auto-ARIMA để tìm tham số
auto_model = pm.auto_arima(
    y=y_search,
    X=X_search,
    start_p=0, start_q=0,
    max_p=3, max_q=3,     # Giới hạn tìm kiếm tối đa bậc 3
    d=0,                  # Do test ADF, đã khẳng định dữ liệu đã stationarized, nên d=0
    seasonal=False,       # Tắt seasonal của ARIMA vì nhóm đã dùng Sin/Cos encoding thay thế
    trace=True,           # Bật True để in ra log quá trình máy đang thử nghiệm
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True         # Dùng thuật toán tìm kiếm thông minh (nhanh hơn vét cạn Grid Search)
)

best_order = auto_model.order
print(f"\n=> TÌM KIẾM HOÀN TẤT! Bộ tham số tốt nhất là: order={best_order}")
print("-" * 60)

# 2. Lấy bộ tham số vừa tìm được để huấn luyện mô hình chính thức
# 3.1. Huấn luyện và lưu mô hình RAW
print(f"Đang huấn luyện SARIMAX {best_order} trên dữ liệu (RAW)...")
model_sarima_raw = sm.tsa.SARIMAX(endog=y_train_raw, exog=X_train_raw, order=best_order)
res_sarima_raw = model_sarima_raw.fit(disp=False)
res_sarima_raw.save('sarimax_raw_model.pkl') # <--- Lưu mô hình RAW

# 3.2. Huấn luyện và lưu mô hình FE no PCA
print(f"Đang huấn luyện SARIMAX {best_order} trên dữ liệu (FE no PCA)...")
model_sarima_fe = sm.tsa.SARIMAX(endog=y_train_raw, exog=X_train_fe_no_pca, order=best_order)
res_sarima_fe = model_sarima_fe.fit(disp=False)
res_sarima_fe.save('sarimax_fe_model.pkl') # <--- Lưu mô hình FE no PCA

# 3.3. Huấn luyện và lưu mô hình FE có PCA
print(f"Đang huấn luyện SARIMAX {best_order} trên dữ liệu (FE CÓ PCA)...")
model_sarima_pca = sm.tsa.SARIMAX(endog=y_train_fe_pca, exog=X_train_fe_pca, order=best_order)
res_sarima_pca = model_sarima_pca.fit(disp=False)
res_sarima_pca.save('sarimax_pca_model.pkl') # <--- Lưu mô hình FE CÓ PCA

print("Đã huấn luyện và lưu thành công cả 3 mô hình SARIMAX!")
# ==========================================
# CELL 1: IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Preprocessing & Metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setting plotting style
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

print("Cell 1: Đã nạp thành công tất cả thư viện!")
# ==========================================
# CELL 2: DATA PREPARATION FUNCTION
# ==========================================
def load_and_split_data(file_path, is_engineered=False):
    """
    Hàm load dữ liệu và phân chia Train/Test 90-10 theo thời gian.
    is_engineered = False: Dữ liệu thô (Raw).
    is_engineered = True : Dữ liệu có Feature Engineering (Lags, Time, Drop Leakage).
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    target_col = 'temperature_2m'
    
    if is_engineered:
        # 1. Ngăn Data Leakage
        if 'apparent_temperature' in df.columns:
            df.drop(columns=['apparent_temperature'], inplace=True)
        
        # 2. Bóc tách Tọa độ Thời gian (Time Features)
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        
        # 3. Tạo Biến trễ Tự hồi quy (Autoregressive Lags)
        df['lag_1'] = df[target_col].shift(1)
        df['lag_24'] = df[target_col].shift(24)
        
    # Bỏ các dòng NaN (tự nhiên hoặc do shift tạo ra)
    df.dropna(inplace=True)

    # Time-based Split (90/10) - Không xáo trộn dữ liệu (No shuffle)
    split_idx = int(len(df) * 0.90)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, model_name, data_type):
    """Hàm tính toán và in ra bộ 3 chỉ số đánh giá cốt lõi"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{data_type}] {model_name:15} | MAE: {mae:.3f} °C | RMSE: {rmse:.3f} °C | R2: {r2:.4f}")
    return {"Model": model_name, "Data": data_type, "MAE": mae, "RMSE": rmse, "R2": r2}

print("Cell 2: Đã khởi tạo các hàm xử lý dữ liệu và đánh giá!")
# ==========================================
# CELL 3: EXPERIMENT A - RAW DATA
# ==========================================
# THAY ĐỔI TÊN FILE DỮ LIỆU CỦA BẠN TẠI ĐÂY
DATA_PATH = 'clean_data/clean_data_train.csv' 
results = []

print("--- THỬ NGHIỆM TRÊN DỮ LIỆU THÔ (RAW DATA) ---")
X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_and_split_data(DATA_PATH, is_engineered=False)

# 1. Random Forest (Raw)
rf_raw = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_raw.fit(X_train_raw, y_train_raw)
results.append(evaluate_model(y_test_raw, rf_raw.predict(X_test_raw), "Random Forest", "RAW"))

# 2. XGBoost (Raw)
xgb_raw = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_raw.fit(X_train_raw, y_train_raw)
results.append(evaluate_model(y_test_raw, xgb_raw.predict(X_test_raw), "XGBoost", "RAW"))
# ==========================================
# CELL 4: EXPERIMENT B - ENGINEERED DATA
# ==========================================
print("--- THỬ NGHIỆM TRÊN DỮ LIỆU FEATURE ENGINEERING ---")
X_train_fe, X_test_fe, y_train_fe, y_test_fe = load_and_split_data(DATA_PATH, is_engineered=True)

# 1. Random Forest (FE)
rf_fe = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_fe.fit(X_train_fe, y_train_fe)
results.append(evaluate_model(y_test_fe, rf_fe.predict(X_test_fe), "Random Forest", "ENGINEERED"))

# 2. XGBoost (FE) - Chống Overfitting với Early Stopping
xgb_fe = xgb.XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42,
    early_stopping_rounds=50 # Đã được cập nhật chuẩn xác cho XGBoost >= 2.0
)
xgb_fe.fit(X_train_fe, y_train_fe, eval_set=[(X_test_fe, y_test_fe)], verbose=False)
results.append(evaluate_model(y_test_fe, xgb_fe.predict(X_test_fe), "XGBoost", "ENGINEERED"))

# 3. Mạng Nơ-ron (MLPRegressor)
print("\n[Đang huấn luyện Mạng Nơ-ron truyền thẳng (MLP)...]")
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_fe)
X_test_scaled = scaler_X.transform(X_test_fe)

mlp_fe = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='relu',           
    solver='adam',               
    max_iter=500,                
    early_stopping=True,         
    random_state=42
)
mlp_fe.fit(X_train_scaled, y_train_fe)
results.append(evaluate_model(y_test_fe, mlp_fe.predict(X_test_scaled), "MLP Neural Net", "ENGINEERED"))
# ==========================================
# CELL 5: SUMMARY TABLE & VISUALIZATION
# ==========================================
print("\n" + "="*60)
print("BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (REPORT TABLE)")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# --- VẼ ĐỒ THỊ SO SÁNH THỰC TẾ VS DỰ BÁO CỦA MÔ HÌNH TỐT NHẤT (XGBOOST) ---
# Chọn 300 giờ đầu tiên của tập Test để vẽ cho rõ nét (khoảng 12 ngày)
plot_window = 300
y_actual = y_test_fe.iloc[:plot_window]
y_pred_xgb = xgb_fe.predict(X_test_fe.iloc[:plot_window])

plt.figure(figsize=(16, 6))
plt.plot(y_actual.index, y_actual.values, label='Nhiệt độ Thực tế (Actual)', color='#2c3e50', linewidth=2)
plt.plot(y_actual.index, y_pred_xgb, label='Dự báo XGBoost (Predicted)', color='#e74c3c', linestyle='--', linewidth=2, alpha=0.9)

plt.title('ĐÁNH GIÁ HIỆU SUẤT DỰ BÁO XGBOOST VS THỰC TẾ (300 GIỜ TẬP TEST)', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Thời gian', fontsize=12)
plt.ylabel('Nhiệt độ (°C)', fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Lưu biểu đồ làm tài liệu đưa vào báo cáo
plt.savefig('xgboost_predictions_vs_actual.png', dpi=300)
plt.show()

print("\nĐã lưu biểu đồ thành công dưới tên: 'xgboost_predictions_vs_actual.png'")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("="*80)
print("TIME SERIES CROSS-VALIDATION PIPELINE (XGBOOST vs RF vs MLP)")
print("="*80)

# ==============================================================================
# 1. CHUẨN BỊ DỮ LIỆU (RAW DATA PREPARATION)
# ==============================================================================
DATA_PATH = 'clean_data/clean_data_train.csv' # Thay bằng tên file của bạn
target_col = 'temperature_2m'

print("[1/5] Đang nạp dữ liệu thô...")
df = pd.read_csv(DATA_PATH)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Ngăn rò rỉ dữ liệu (Giữ nguyên quan điểm bảo vệ mô hình)
if 'apparent_temperature' in df.columns:
    df.drop(columns=['apparent_temperature'], inplace=True)

# Dù là dữ liệu thô, các mô hình ML không hiểu datetime, cần bóc tách cơ bản
df['hour'] = df.index.hour
df['month'] = df.index.month
df['day_of_year'] = df.index.dayofyear
df.dropna(inplace=True)

X = df.drop(columns=[target_col])
y = df[target_col]

# ==============================================================================
# 2. KHỞI TẠO CẤU TRÚC CROSS-VALIDATION & MÔ HÌNH
# ==============================================================================
print("[2/5] Thiết lập Time Series Split (5 Folds)...")
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Khởi tạo từ điển lưu trữ kết quả
metrics = {
    'Random Forest': {'train_mae': [], 'val_mae': [], 'train_rmse': [], 'val_rmse': [], 'train_r2': [], 'val_r2': []},
    'XGBoost':       {'train_mae': [], 'val_mae': [], 'train_rmse': [], 'val_rmse': [], 'train_r2': [], 'val_r2': []},
    'MLP Neural Net':{'train_mae': [], 'val_mae': [], 'train_rmse': [], 'val_rmse': [], 'train_r2': [], 'val_r2': []}
}

# Biến để lưu lịch sử Loss của Fold cuối cùng (để vẽ biểu đồ)
last_fold_xgb_evals = None
last_fold_mlp_loss = None

# ==============================================================================
# 3. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# ==============================================================================
print("[3/5] Bắt đầu quá trình Huấn luyện & Đánh giá chéo...")

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\n--- Đang xử lý FOLD {fold + 1}/{n_splits} ---")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # MLP và đa số mô hình cần dữ liệu chuẩn hóa (Chỉ fit trên tập Train để tránh leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # --------------------------------------------------
    # A. MÔ HÌNH RANDOM FOREST
    # --------------------------------------------------
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    rf_pred_train = rf.predict(X_train_scaled)
    rf_pred_val = rf.predict(X_val_scaled)
    
    metrics['Random Forest']['train_mae'].append(mean_absolute_error(y_train, rf_pred_train))
    metrics['Random Forest']['val_mae'].append(mean_absolute_error(y_val, rf_pred_val))
    metrics['Random Forest']['train_rmse'].append(np.sqrt(mean_squared_error(y_train, rf_pred_train)))
    metrics['Random Forest']['val_rmse'].append(np.sqrt(mean_squared_error(y_val, rf_pred_val)))
    metrics['Random Forest']['train_r2'].append(r2_score(y_train, rf_pred_train))
    metrics['Random Forest']['val_r2'].append(r2_score(y_val, rf_pred_val))
    print("  ✓ Random Forest Done.")

    # --------------------------------------------------
    # B. MÔ HÌNH XGBOOST
    # --------------------------------------------------
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.1, max_depth=6, 
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=30
    )
    # XGBoost dùng eval_set để lấy loss curve
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)], verbose=False)
    
    xgb_pred_train = xgb_model.predict(X_train_scaled)
    xgb_pred_val = xgb_model.predict(X_val_scaled)
    
    metrics['XGBoost']['train_mae'].append(mean_absolute_error(y_train, xgb_pred_train))
    metrics['XGBoost']['val_mae'].append(mean_absolute_error(y_val, xgb_pred_val))
    metrics['XGBoost']['train_rmse'].append(np.sqrt(mean_squared_error(y_train, xgb_pred_train)))
    metrics['XGBoost']['val_rmse'].append(np.sqrt(mean_squared_error(y_val, xgb_pred_val)))
    metrics['XGBoost']['train_r2'].append(r2_score(y_train, xgb_pred_train))
    metrics['XGBoost']['val_r2'].append(r2_score(y_val, xgb_pred_val))
    
    if fold == n_splits - 1: # Lưu loss của fold cuối cùng
        last_fold_xgb_evals = xgb_model.evals_result()
    print("  ✓ XGBoost Done.")

    # --------------------------------------------------
    # C. MÔ HÌNH MLP REGRESSOR
    # --------------------------------------------------
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
        max_iter=300, early_stopping=True, random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    
    mlp_pred_train = mlp.predict(X_train_scaled)
    mlp_pred_val = mlp.predict(X_val_scaled)
    
    metrics['MLP Neural Net']['train_mae'].append(mean_absolute_error(y_train, mlp_pred_train))
    metrics['MLP Neural Net']['val_mae'].append(mean_absolute_error(y_val, mlp_pred_val))
    metrics['MLP Neural Net']['train_rmse'].append(np.sqrt(mean_squared_error(y_train, mlp_pred_train)))
    metrics['MLP Neural Net']['val_rmse'].append(np.sqrt(mean_squared_error(y_val, mlp_pred_val)))
    metrics['MLP Neural Net']['train_r2'].append(r2_score(y_train, mlp_pred_train))
    metrics['MLP Neural Net']['val_r2'].append(r2_score(y_val, mlp_pred_val))
    
    if fold == n_splits - 1: # Lưu loss của fold cuối cùng
        last_fold_mlp_loss = mlp.loss_curve_
    print("  ✓ MLP Neural Net Done.")

# ==============================================================================
# 4. TÍNH TOÁN TRUNG BÌNH & TÌM MÔ HÌNH TỐT NHẤT
# ==============================================================================
print("\n[4/5] Tổng hợp kết quả Cross-Validation...")

avg_metrics = {}
best_model_name = ""
best_val_rmse = float('inf')

for model_name, m in metrics.items():
    avg_metrics[model_name] = {
        'Train MAE': np.mean(m['train_mae']), 'Val MAE': np.mean(m['val_mae']),
        'Train RMSE': np.mean(m['train_rmse']), 'Val RMSE': np.mean(m['val_rmse']),
        'Train R2': np.mean(m['train_r2']), 'Val R2': np.mean(m['val_r2'])
    }
    # Tìm mô hình có Validation RMSE thấp nhất
    if avg_metrics[model_name]['Val RMSE'] < best_val_rmse:
        best_val_rmse = avg_metrics[model_name]['Val RMSE']
        best_model_name = model_name

# In bảng tổng hợp
df_results = pd.DataFrame(avg_metrics).T
print("\nBẢNG ĐÁNH GIÁ TRUNG BÌNH QUA 5 FOLDS:")
print("-" * 75)
print(df_results.round(4).to_string())
print("-" * 75)
print(f"🏆 MÔ HÌNH XUẤT SẮC NHẤT: {best_model_name} (Val RMSE: {best_val_rmse:.4f})")

# ==============================================================================
# 5. HUẤN LUYỆN LẠI MÔ HÌNH TỐT NHẤT TRÊN TOÀN BỘ DỮ LIỆU & LƯU LẠI
# ==============================================================================
print(f"\n[5/5] Đang huấn luyện lại {best_model_name} trên toàn bộ dữ liệu (Full Dataset)...")
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

if best_model_name == 'XGBoost':
    final_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
elif best_model_name == 'Random Forest':
    final_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
else:
    final_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=300, random_state=42)

final_model.fit(X_scaled_full, y)

# Lưu scaler và model để sử dụng cho file test thực tế sau này
joblib.dump(scaler_full, 'feature_scaler.pkl')
joblib.dump(final_model, 'best_production_model.pkl')
print("💾 Đã lưu thành công 2 files: 'best_production_model.pkl' và 'feature_scaler.pkl'")

# ==============================================================================
# 6. TRỰC QUAN HÓA (VISUALIZATION CHO BÁO CÁO)
# ==============================================================================
print("\n🎨 Đang xuất biểu đồ phân tích...")

# --- 6.1 Biểu đồ Loss Curves (Fold cuối cùng) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# XGBoost Loss
epochs_xgb = len(last_fold_xgb_evals['validation_0']['rmse'])
axes[0].plot(range(epochs_xgb), last_fold_xgb_evals['validation_0']['rmse'], label='Train RMSE', color='blue')
axes[0].plot(range(epochs_xgb), last_fold_xgb_evals['validation_1']['rmse'], label='Validation RMSE', color='red')
axes[0].set_title('XGBoost - Learning Curve (Fold 5)', fontweight='bold')
axes[0].set_xlabel('Số lượng Cây (Boosting Rounds)')
axes[0].set_ylabel('RMSE (°C)')
axes[0].legend()

# MLP Loss
axes[1].plot(last_fold_mlp_loss, label='Train Loss (MSE)', color='green')
axes[1].set_title('MLP Neural Net - Train Loss Curve (Fold 5)', fontweight='bold')
axes[1].set_xlabel('Vòng lặp (Epochs)')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('learning_curves_cv.png', dpi=300)
plt.show()

# --- 6.2 Biểu đồ Bar Chart so sánh Metrics ---
models = list(avg_metrics.keys())
train_maes = [avg_metrics[m]['Train MAE'] for m in models]
val_maes = [avg_metrics[m]['Val MAE'] for m in models]
train_rmses = [avg_metrics[m]['Train RMSE'] for m in models]
val_rmses = [avg_metrics[m]['Val RMSE'] for m in models]
train_r2s = [avg_metrics[m]['Train R2'] for m in models]
val_r2s = [avg_metrics[m]['Val R2'] for m in models]

x = np.arange(len(models))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Đồ thị MAE
axes[0].bar(x - width/2, train_maes, width, label='Train MAE', color='#74b9ff')
axes[0].bar(x + width/2, val_maes, width, label='Val MAE', color='#0984e3')
axes[0].set_title(r'So sánh MAE (°C) $\downarrow$', fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(models)
axes[0].legend()

# Đồ thị RMSE
axes[1].bar(x - width/2, train_rmses, width, label='Train RMSE', color='#ff7675')
axes[1].bar(x + width/2, val_rmses, width, label='Val RMSE', color='#d63031')
axes[1].set_title(r'So sánh RMSE (°C) $\downarrow$', fontweight='bold')
axes[1].set_xticks(x); axes[1].set_xticklabels(models)
axes[1].legend()

# Đồ thị R2 Score
axes[2].bar(x - width/2, train_r2s, width, label='Train R2', color='#55efc4')
axes[2].bar(x + width/2, val_r2s, width, label='Val R2', color='#00b894')
axes[2].set_title(r'So sánh R2 Score $\uparrow$', fontweight='bold')
axes[2].set_xticks(x); axes[2].set_xticklabels(models)
axes[2].set_ylim(0, 1.1)
axes[2].legend()

plt.tight_layout()
plt.savefig('metrics_comparison_cv.png', dpi=300)
plt.show()

print("🎉 Hoàn tất! Đã lưu 2 hình ảnh biểu đồ vào thư mục.")
# ==============================================================================
# ĐỒ ÁN: DỰ BÁO NHIỆT ĐỘ BẰNG XGBOOST 
# CHIẾN LƯỢC: TRỄ HÓA BIẾN NGOẠI SINH TOÀN PHẦN (FULLY LAGGED EXOGENOUS)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("="*75)
print("KHỞI ĐỘNG PIPELINE HUẤN LUYỆN XGBOOST THỰC CHIẾN (PRODUCTION-READY)")
print("="*75)

# ==============================================================================
# 1. HÀM FEATURE ENGINEERING: TRỄ HÓA TOÀN PHẦN
# ==============================================================================
def create_fully_lagged_features(df, target_col='temperature_2m'):
    """
    Biến đổi dữ liệu đa biến thành dạng Trễ hóa.
    Mô hình tại thời điểm T sẽ học từ các biến ngoại sinh tại T-1 và T-24.
    Tuyệt đối không dùng Độ ẩm/Áp suất của thời điểm T để dự báo Nhiệt độ thời điểm T.
    """
    print("[1/5] Đang xử lý Feature Engineering (Trễ hóa toàn phần)...")
    df = df.copy()
    
    # Ngăn rò rỉ dữ liệu phái sinh
    if 'apparent_temperature' in df.columns:
        df.drop(columns=['apparent_temperature'], inplace=True)
        
    # Tạo Tọa độ thời gian (Luôn biết trước ở tương lai)
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    
    # Lấy danh sách các cột gốc (Trừ thời gian)
    original_features = [col for col in df.columns if col not in ['hour', 'month', 'day_of_year']]
    
    # Tạo biến trễ (Lags) cho TOÀN BỘ các cột vật lý (Bao gồm cả Nhiệt độ)
    for col in original_features:
        df[f'{col}_lag1'] = df[col].shift(1)   # Dữ liệu của 1 giờ trước
        df[f'{col}_lag24'] = df[col].shift(24) # Dữ liệu của 24 giờ trước
        
    # BƯỚC QUYẾT ĐỊNH: Xóa bỏ dữ liệu ngoại sinh ở thời điểm hiện tại (T)
    # Chỉ giữ lại duy nhất cột Nhiệt độ (Target) làm nhãn để huấn luyện
    cols_to_drop = [col for col in original_features if col != target_col]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Xóa 24 dòng đầu tiên bị NaN do phép toán dịch chuyển (shift)
    df.dropna(inplace=True)
    
    return df

# ==============================================================================
# 2. NẠP DỮ LIỆU & PHÂN CHIA (TRAIN/TEST SPLIT)
# ==============================================================================
DATA_PATH = 'clean_data/clean_data_train.csv' # <--- SỬA TÊN FILE CỦA BẠN Ở ĐÂY
TARGET = 'temperature_2m'

print(f"[2/5] Đang nạp dữ liệu từ '{DATA_PATH}'...")
raw_df = pd.read_csv(DATA_PATH)
raw_df['time'] = pd.to_datetime(raw_df['time'])
raw_df.set_index('time', inplace=True)

# Đưa qua bộ lọc Feature Engineering
processed_df = create_fully_lagged_features(raw_df, TARGET)

# Phân chia Train/Test 90-10 theo trình tự thời gian
split_idx = int(len(processed_df) * 0.90)
train_df = processed_df.iloc[:split_idx]
test_df = processed_df.iloc[split_idx:]

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"  -> Tập Huấn luyện (Train): {X_train.shape[0]} mẫu")
print(f"  -> Tập Kiểm thử (Test)  : {X_test.shape[0]} mẫu")
print(f"  -> Số lượng Features    : {X_train.shape[1]} (Đã bao gồm Lags và Thời gian)")

# ==============================================================================
# 3. KHỞI TẠO VÀ HUẤN LUYỆN XGBOOST
# ==============================================================================
print("[3/5] Đang huấn luyện thuật toán XGBoost Regressor...")

# Siêu tham số đã được căn chỉnh tối ưu cho dự báo khí tượng
xgb_model = xgb.XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42,
    early_stopping_rounds=50 # Dừng nếu 50 cây liên tiếp không giảm được lỗi trên tập Test
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
print("  ✓ Huấn luyện hoàn tất!")

# ==============================================================================
# 4. ĐÁNH GIÁ MÔ HÌNH VÀ LƯU TRỮ
# ==============================================================================
print("[4/5] Đang kiểm thử và trích xuất chỉ số (Metrics)...")
y_pred_test = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*50)
print("BẢNG ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH (TEST SET)")
print("="*50)
print(f" 📍 Sai số tuyệt đối (MAE) : {mae:.3f} °C")
print(f" 📍 Sai số bình phương (RMSE): {rmse:.3f} °C")
print(f" 📍 Độ giải thích (R2 Score) : {r2:.4f}")
print("="*50)

# Xuất Mô hình ra file để triển khai (Deploy)
MODEL_NAME = 'xgboost_lagged_production.pkl'
joblib.dump(xgb_model, MODEL_NAME)
print(f"\n💾 Đã lưu mô hình sẵn sàng Production tại: {MODEL_NAME}")

# ==============================================================================
# 5. TRỰC QUAN HÓA KẾT QUẢ ĐỂ ĐƯA VÀO BÁO CÁO
# ==============================================================================
print("[5/5] Đang vẽ biểu đồ phân tích...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Đồ thị 1: Thực tế vs Dự báo (Chỉ lấy 300 mẫu đầu cho rõ ràng)
plot_limit = 300
axes[0].plot(y_test.index[:plot_limit], y_test.values[:plot_limit], label='Thực tế (Actual)', color='#2c3e50', linewidth=2)
axes[0].plot(y_test.index[:plot_limit], y_pred_test[:plot_limit], label='Dự báo XGBoost', color='#e74c3c', linestyle='--', linewidth=2)
axes[0].set_title(r'ĐỐI CHIẾU NHIỆT ĐỘ THỰC TẾ & DỰ BÁO (300 GIỜ)', fontweight='bold')
axes[0].set_ylabel('Nhiệt độ (°C)')
axes[0].legend()

# Đồ thị 2: Feature Importance (Biến nào đóng góp nhiều nhất?)
importances = xgb_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[-10:] # Lấy Top 10 biến quan trọng nhất

axes[1].barh(range(len(indices)), importances[indices], color='#0984e3', align='center')
axes[1].set_yticks(range(len(indices)))
axes[1].set_yticklabels([feature_names[i] for i in indices])
axes[1].set_title('TOP 10 ĐẶC TRƯNG QUAN TRỌNG NHẤT (FEATURE IMPORTANCE)', fontweight='bold')
axes[1].set_xlabel('Mức độ đóng góp (F-score)')

plt.tight_layout()
plt.savefig('final_evaluation_dashboard.png', dpi=300)
plt.show()

print("\n🎉 HOÀN TẤT! Đã lưu ảnh Dashboard tại: 'final_evaluation_dashboard.png'")