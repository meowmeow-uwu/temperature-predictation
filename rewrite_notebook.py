import json

with open('4_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

idx_corr = -1
idx_pca = -1
idx_lag_md = -1
idx_lag_c1 = -1
idx_lag_c2 = -1
idx_tonghop = -1
idx_3versions = -1
idx_scaler = -1

for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if '1.1 Phân tích ma trận tương quan' in src:
        idx_corr = i
    if '2. Phân tích thành phần chính (PCA)' in src:
        idx_pca = i
    if 'Tạo đặc trưng trễ (Lag Features)' in src:
        idx_lag_md = i
    if '4. Tạo Lag Features' in src:
        idx_lag_c1 = i
    if 'So sánh Trước và Sau tạo Lag Features' in src:
        idx_lag_c2 = i
    if '5. Tổng hợp và thống kê bộ đặc trưng cuối cùng' in src:
        idx_tonghop = i
    if 'CHUẨN BỊ 3 PHIÊN BẢN DATASET' in src:
        idx_3versions = i
    if '2.1 Chuẩn hóa dữ liệu (StandardScaler)' in src and 'feature_cols' in src:
        idx_scaler = i

# Modify markdown of PCA to be 3.
for i in range(len(cells)):
    if cells[i]['cell_type'] == 'markdown':
        for j in range(len(cells[i]['source'])):
            cells[i]['source'][j] = cells[i]['source'][j].replace('## 2. Phân tích thành phần chính', '## 3. Phân tích thành phần chính')
            cells[i]['source'][j] = cells[i]['source'][j].replace('### 2.', '### 3.')
            cells[i]['source'][j] = cells[i]['source'][j].replace('## 3. Trích xuất', '## 4. Trích xuất')
            cells[i]['source'][j] = cells[i]['source'][j].replace('### 3.', '### 4.')
            cells[i]['source'][j] = cells[i]['source'][j].replace('## 4. Tạo đặc trưng trễ', '## 2. Tạo đặc trưng trễ')

for i in range(len(cells)):
    if cells[i]['cell_type'] == 'code':
        for j in range(len(cells[i]['source'])):
            # Replace code comments for section numbers
            if '# 2.1' in cells[i]['source'][j]: cells[i]['source'][j] = cells[i]['source'][j].replace('# 2.1', '# 3.1')
            elif '# 2.2' in cells[i]['source'][j]: cells[i]['source'][j] = cells[i]['source'][j].replace('# 2.2', '# 3.2')
            elif '# 2.3' in cells[i]['source'][j]: cells[i]['source'][j] = cells[i]['source'][j].replace('# 2.3', '# 3.3')
            elif '# 2.4' in cells[i]['source'][j]: cells[i]['source'][j] = cells[i]['source'][j].replace('# 2.4', '# 3.4')
            elif '# 2.5' in cells[i]['source'][j]: cells[i]['source'][j] = cells[i]['source'][j].replace('# 2.5', '# 3.5')
            elif '# 3.1' in cells[i]['source'][j]: cells[i]['source'][j] = cells[i]['source'][j].replace('# 3.1', '# 4.1')
            elif '# 3.2' in cells[i]['source'][j]: cells[i]['source'][j] = cells[i]['source'][j].replace('# 3.2', '# 4.2')

cells[idx_lag_c1]['source'] = [
    "# ==================================================\n",
    "# 2. Tạo Lag Features\n",
    "# Ghép train + test trước khi tạo lag để test có đủ giá trị lịch sử\n",
    "# ==================================================\n",
    "\n",
    "# Đánh dấu is_train để tách lại sau\n",
    "train_df['is_train'] = 1\n",
    "test_df['is_train']  = 0\n",
    "\n",
    "# Ghép lại và sắp xếp theo thời gian\n",
    "full_df = pd.concat([train_df, test_df], ignore_index=True)\n",
    "full_df = full_df.sort_values('time').reset_index(drop=True)\n",
    "\n",
    "# Tạo lag features\n",
    "lag_hours = [1, 2, 3, 12, 24, 72, 168]\n",
    "for lag in lag_hours:\n",
    "    full_df[f'temp_lag_{lag}'] = full_df['temperature_2m'].shift(lag)\n",
    "\n",
    "# Tách lại train và test, dropna loại bỏ các dòng không đủ lịch sử\n",
    "train_df = full_df[full_df['is_train'] == 1].dropna().reset_index(drop=True)\n",
    "test_df  = full_df[full_df['is_train'] == 0].dropna().reset_index(drop=True)\n",
    "\n",
    "# Bỏ cột is_train\n",
    "train_df = train_df.drop(columns=['is_train'])\n",
    "test_df  = test_df.drop(columns=['is_train'])\n",
    "\n",
    "print(\"=== KẾT QUẢ LAG FEATURES ===\")\n",
    "print(f\"\\nTrain shape: {train_df.shape}\")\n",
    "print(f\"Test shape:  {test_df.shape}\")\n",
    "print(f\"\\nCác cột lag: {[col for col in train_df.columns if 'lag' in col]}\")\n",
    "print(f\"\\nTất cả cột: {train_df.columns.tolist()}\")\n",
    "print(f\"\\nTrain time range: {train_df['time'].min()} → {train_df['time'].max()}\")\n",
    "print(f\"Test time range:  {test_df['time'].min()} → {test_df['time'].max()}\")\n"
]

cells[idx_lag_c2]['source'] = [
    "# ==================================================\n",
    "# TRỰC QUAN HÓA: So sánh Trước và Sau tạo Lag Features\n",
    "# ==================================================\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 4))\n",
    "\n",
    "# Lấy 7 ngày đầu để visualize rõ\n",
    "sample = train_df.iloc[:168]\n",
    "\n",
    "# Trước lag — chỉ có temperature_2m gốc\n",
    "axes[0].plot(sample['time'], sample['temperature_2m'],\n",
    "             color='steelblue', linewidth=1)\n",
    "axes[0].set_title('Trước Lag Features\\n(chỉ có temperature_2m gốc)')\n",
    "axes[0].set_xlabel('Thời gian')\n",
    "axes[0].set_ylabel('Nhiệt độ (°C)')\n",
    "axes[0].tick_params(axis='x', rotation=45)\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Sau lag — so sánh temp và lag_24\n",
    "axes[1].plot(sample['time'], sample['temperature_2m'],\n",
    "             color='steelblue', linewidth=1, label='temperature_2m (hiện tại)')\n",
    "axes[1].plot(sample['time'], sample['temp_lag_24'],\n",
    "             color='tomato', linewidth=1, linestyle='--', label='temp_lag_24 (24h trước)')\n",
    "axes[1].plot(sample['time'], sample['temp_lag_1'],\n",
    "             color='green', linewidth=1, linestyle=':', label='temp_lag_1 (1h trước)')\n",
    "axes[1].set_title('Sau Lag Features\\n(temperature_2m vs temp_lag_1 vs temp_lag_24)')\n",
    "axes[1].set_xlabel('Thời gian')\n",
    "axes[1].set_ylabel('Nhiệt độ (°C)')\n",
    "axes[1].tick_params(axis='x', rotation=45)\n",
    "axes[1].legend(fontsize=8)\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.suptitle('So sánh Trước và Sau tạo Lag Features', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

cells[idx_scaler]['source'] = [
    "# ==================================================\n",
    "# 3.1 Chuẩn hóa dữ liệu (StandardScaler)\n",
    "# ==================================================\n",
    "feature_cols = ['wind_speed_10m', 'cloud_cover', 'relative_humidity_2m',\n",
    "                'surface_pressure', 'precipitation', 'vapour_pressure_deficit']\n",
    "\n",
    "lag_hours = [1, 2, 3, 12, 24, 72, 168]\n",
    "for lag in lag_hours:\n",
    "    feature_cols.append(f'temp_lag_{lag}')\n",
    "\n",
    "X_train = train_df[feature_cols].values\n",
    "X_test  = test_df[feature_cols].values\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)  # fit + transform trên train\n",
    "X_test_scaled  = scaler.transform(X_test)        # chỉ transform trên test\n",
    "\n",
    "print(\"=== KẾT QUẢ CHUẨN HÓA ===\")\n",
    "print(f\"\\nX_train_scaled shape: {X_train_scaled.shape}\")\n",
    "print(f\"X_test_scaled shape:  {X_test_scaled.shape}\")\n",
    "print(f\"\\nTrước chuẩn hóa — Mean: {X_train.mean(axis=0).round(3)}\")\n",
    "print(f\"Trước chuẩn hóa — Std:  {X_train.std(axis=0).round(3)}\")\n",
    "print(f\"\\nSau chuẩn hóa  — Mean: {X_train_scaled.mean(axis=0).round(4)}\")\n",
    "print(f\"Sau chuẩn hóa  — Std:  {X_train_scaled.std(axis=0).round(4)}\")\n"
]

cells[idx_tonghop]['source'] = [
    "# ==================================================\n",
    "# 5. Tổng hợp và thống kê bộ đặc trưng cuối cùng\n",
    "# ==================================================\n",
    "feature_cols_final = [col for col in train_pca_df.columns\n",
    "                      if col not in ['time', 'temperature_2m']]\n",
    "\n",
    "print(\"=\" * 55)\n",
    "print(\"TỔNG HỢP BỘ ĐẶC TRƯNG CUỐI CÙNG (PCA + TIME)\")\n",
    "print(\"=\" * 55)\n",
    "print(f\"\\nBiến mục tiêu: temperature_2m\")\n",
    "print(f\"\\nCác features đầu vào ({len(feature_cols_final)} biến):\")\n",
    "for i, col in enumerate(feature_cols_final, 1):\n",
    "    print(f\"  {i:2d}. {col}\")\n",
    "\n",
    "print(f\"\\n{'Tập dữ liệu':<15} {'Số mẫu':>10} {'Số features':>12} {'Khoảng thời gian'}\")\n",
    "print(\"-\" * 70)\n",
    "print(f\"{'Train':<15} {len(train_pca_df):>10} {len(feature_cols_final):>12} \"\n",
    "      f\"{train_pca_df['time'].min().date()} → {train_pca_df['time'].max().date()}\")\n",
    "print(f\"{'Test':<15} {len(test_pca_df):>10} {len(feature_cols_final):>12} \"\n",
    "      f\"{test_pca_df['time'].min().date()} → {test_pca_df['time'].max().date()}\")\n",
    "\n",
    "print(\"\\n5 dòng đầu Train:\")\n",
    "print(train_pca_df.head())\n"
]

cells[idx_3versions]['source'] = [
    "# =======================================================\n",
    "# CHUẨN BỊ 3 PHIÊN BẢN DATASET TỪ CÁC BIẾN ĐÃ CÓ\n",
    "# =======================================================\n",
    "feature_cols_stats = ['wind_speed_10m', 'cloud_cover', 'relative_humidity_2m',\n",
    "                      'surface_pressure', 'precipitation', 'vapour_pressure_deficit']\n",
    "\n",
    "# 1. Dataset FE_PCA (Chính là tập từ PCA, đã gồm lag/time vì PCA chạy trên lag, time thêm sau)\n",
    "X_train_fe_pca = train_pca_df[[col for col in train_pca_df.columns if col not in ['time', 'temperature_2m']]].copy()\n",
    "y_train_fe_pca = train_pca_df['temperature_2m'].copy()\n",
    "X_test_fe_pca = test_pca_df[[col for col in test_pca_df.columns if col not in ['time', 'temperature_2m']]].copy()\n",
    "y_test_fe_pca = test_pca_df['temperature_2m'].copy()\n",
    "\n",
    "# 2. Dataset RAW (Không qua chuẩn hóa, không Lag/Time)\n",
    "X_train_raw = train_df[feature_cols_stats].copy()\n",
    "y_train_raw = train_df['temperature_2m'].copy()\n",
    "X_test_raw = test_df[feature_cols_stats].copy()\n",
    "y_test_raw = test_df['temperature_2m'].copy()\n",
    "\n",
    "# 3. Dataset FE_no_PCA (Có Chuẩn hóa + Lag/Time, nhưng KHÔNG dùng PCA)\n",
    "time_lag_cols = ['sin_hour', 'cos_hour', 'sin_month', 'cos_month'] + [f'temp_lag_{l}' for l in [1, 2, 3, 12, 24, 72, 168]]\n",
    "X_train_scaled_df = pd.DataFrame(X_train_scaled[:, :6], columns=feature_cols_stats)\n",
    "X_test_scaled_df = pd.DataFrame(X_test_scaled[:, :6], columns=feature_cols_stats)\n",
    "X_train_lag_time = train_pca_df[['sin_hour', 'cos_hour', 'sin_month', 'cos_month']].copy()\n",
    "for l in [1, 2, 3, 12, 24, 72, 168]: X_train_lag_time[f'temp_lag_{l}'] = train_df[f'temp_lag_{l}'].values\n",
    "X_test_lag_time = test_pca_df[['sin_hour', 'cos_hour', 'sin_month', 'cos_month']].copy()\n",
    "for l in [1, 2, 3, 12, 24, 72, 168]: X_test_lag_time[f'temp_lag_{l}'] = test_df[f'temp_lag_{l}'].values\n",
    "\n",
    "X_train_fe_no_pca = pd.concat([X_train_scaled_df, X_train_lag_time], axis=1)\n",
    "X_test_fe_no_pca = pd.concat([X_test_scaled_df, X_test_lag_time], axis=1)\n",
    "\n",
    "print(\"Đã tạo thành công 3 phiên bản dữ liệu:\")\n",
    "print(f\"   • Raw: {X_train_raw.shape} (chỉ 6 cột thống kê gốc)\")\n",
    "print(f\"   • FE_no_PCA: {X_train_fe_no_pca.shape} (chuẩn hóa + lag/time)\")\n",
    "print(f\"   • FE_PCA: {X_train_fe_pca.shape} (PCA + lag/time)\")\n",
    "print(f\"\\n   y_train_raw: {y_train_raw.shape}\")\n",
    "print(f\"   y_train_fe_pca: {y_train_fe_pca.shape}\")\n"
]

# Extract lag cells
c24 = cells.pop(idx_lag_c2)
c23 = cells.pop(idx_lag_c1)
c22 = cells.pop(idx_lag_md)

# We want to insert them AFTER idx_corr.
# Note: since we popped elements at indices > idx_corr, idx_corr is unchanged.
cells.insert(idx_corr + 1, c24)
cells.insert(idx_corr + 1, c23)
cells.insert(idx_corr + 1, c22)

# Clear outputs so the user can re-run and see it's fresh
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

with open('4_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Modified notebook saved.")
