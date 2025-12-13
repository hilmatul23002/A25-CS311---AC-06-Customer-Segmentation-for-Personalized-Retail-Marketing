# Pengelompokan Pelanggan Berbasis RFM dengan Algoritma K-Means untuk Strategi Marketing Terpersonalisasi di Industri Ritel
## Proyek akhir (Capstone Project Asah 2025)
### Pada proyek ini dilakukan segmentasi pelanggan menggunakan RFM dan algoritma K-Means untuk memahami perilaku pelanggan guna memaksimalkan strategi pemasaran pada tiap segmennya. Data yang digunakan pada proyek ini bersumber dari kaggle dengan kumpulan data tahun 2010-2011.


Rumusan masalah proyek ini: 
* Segmen pelanggan seperti apa yang dapat diidentifikasi berdasarkan data transaksi? 
* Bagaimana Algoritma K-Means Clustering secara efektif mengidentifikasi perilaku pelanggan?
* Bagaimana segmentasi ini dapat memberi informasi dan meningkatkan strategi pemasaran terarah untuk setiap segmen pelanggan yang telah diidentifikasi?

Tujuan dari proyek ini:
* Memperoleh segmentasi pelanggan untuk optimalisasi marketing

### Berikut adalah beberapa langkah yang dilakukan dalam menyelesaikan proyek ini
### 1. Siapkan Tools
Proyek dijalankan menggunakan Google Colab, namun apabila ingin menjalankan secara lokal, perlu dilakukan install dependency berikut:
* pip install pandas
* pip install numpy
* pip install matplotlib
* pip install feature_engine
* pip install kagglehub
* pip install seaborn
* pip install scikit-learn
### 2. Import Library yang adakan di gunakan
## Dependencies / Libraries

Berikut adalah library Python yang digunakan dalam proyek ini:

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from feature_engine.outliers import Winsorizer
import kagglehub
import math
import streamlit as st
from google.colab import files
```

#### Catatan

* Pastikan seluruh library telah terpasang sebelum menjalankan proyek.
* Beberapa library seperti `google.colab` hanya relevan jika dijalankan di lingkungan Google Colab.
* `streamlit` digunakan jika aplikasi dijalankan sebagai web app.
### 3. Data Loading
Pada proses ini, data langsung diambil dari kaggle tanpa harus mengunduh file dan menyimpah file pada perangkat terlebih dahulu.

Tahap ini bertujuan untuk mengunduh dataset *Online Retail* dari Kaggle, membaca file Excel, lalu mengonversinya ke format CSV agar lebih mudah diproses pada tahap selanjutnya.

3.1. **Mengunduh dataset dari Kaggle** menggunakan `kagglehub`:

```python
# Mengunduh dataset dari Kaggle

dataset_path = kagglehub.dataset_download(
    'jihyeseo/online-retail-data-set-from-uci-ml-repo'
)
print("Dataset berhasil diunduh di folder:", dataset_path)
```

3.2. **Menampilkan isi folder dataset dan memilih file**:

```python
import os

files = os.listdir(dataset_path)
file_path = os.path.join(dataset_path, files[0])
```

3.3. **Membaca file Excel ke dalam DataFrame**:

```python

df = pd.read_excel(file_path)
```

3.4. **Menyimpan dataset ke format CSV**:

```python
csv_output_path = 'Online_Retail.csv'
df.to_csv(csv_output_path, index=False)
print("CSV berhasil disimpan di:", csv_output_path)
```

3.5. **Menampilkan data awal** untuk verifikasi:

```python
df.head(10)
```

#### Output

* Dataset berhasil diunduh dan diekstrak secara otomatis.
* File CSV `Online_Retail.csv` tersimpan dan siap digunakan untuk tahap *data preprocessing* dan *modeling*.

### 4. Data Cleaning
Dataset Online Retail memiliki beberapa masalah umum yang harus dibersihkan terlebih dahulu. Beberapa yang dilakukan pada proses ini:

4.1. Menghapus data duplikat

```python
df = df.drop_duplicates().copy()
df.duplicated().sum()
```

4.2. Menghapus baris yang memiliki missing values (nilai kosong/NaN)

```python
df_clean = df.dropna(subset=["CustomerID"])
df_clean.head(10)
```

4.3. Menghapus transaksi yang bernilai negatif

```python
df_clean = df_clean[df_clean['Quantity'] > 0]
df_clean = df_clean[df_clean['UnitPrice'] > 0]
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
df_clean.head(10)
```
4.4. Mengubah tipe data

```python
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
df_clean.head(10)
```
4.5. Menghapus transaksi dengan Invoice bernilai Cancel (prefix "C")

```python
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
```
4.6. Menghapus outlier

```python
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    variables=['Quantity', 'UnitPrice', 'TotalPrice'],
    fold=1.5
)

df = winsor.fit_transform(df_clean)

plt.figure(figsize=(12, 6))
plt.subplot
sns.boxplot(data=df[['Quantity', 'UnitPrice', 'TotalPrice']])
plt.title("Boxplot of Outliers")
plt.show()
```
4.7. Membagi dataset menjadi global dan local

```python
df_global = df.copy()
df_local = df[df['Country'] == 'United Kingdom'].copy()
```
### 5. Exploratory Data Analysis (EDA)
Tujuan Utama dari EDA ini adalah untuk memahami karakteristik data transaksi penjualan, mengidentifikasi pola pembelian, insight terkait produk, harga, pelanggan, serta tren penjualan dari waktu ke waktu.

Analisis mencakup beberapa bagian utama:
* Analisis harga barang (distribusi, barang termurah & termahal)
* Analisis performa produk (produk paling laku & paling jarang dibeli)
* Analisis perilaku pelanggan (frekuensi pembelian dan total belanja)
* Analisis tren waktu (jumlah transaksi & total belanja per bulan)
  
EDA ini dilakukan pada dua data:
* Data Global (all customers)
* Data Local (local customers)

5.1. Menampilkan Informasi Data Global dan Data Local

```python
df_global.info()
```
```python
df_local.info()
```
```python
df_global.describe(include='all')
```
```python
df_local.describe(include='all')
```
```python
df_global.dtypes
```
```python
df_local.dtypes
```
```python
df_global.isnull().sum()
```
```python
df_local.isnull().sum()
```
```python
df_global.shape
```
```python
df_local.shape
```

5.2. Menampilkan Insight Harga

5.2.1. Menampilkan Distribusi Harga
```python
#Menampilkan grafik distribusi harga
# ---- Grafik kiri: Global ----
plt.subplot(1, 2, 1)   # 1 baris, 2 kolom, grafik ke-1
sns.histplot(df_global['UnitPrice'], bins=50, kde=True)
plt.title("Distribusi Harga Barang (Global)")
plt.xlabel("Unit Price")
plt.ylabel("Frequency")

# ---- Grafik kanan: Local ----
plt.subplot(1, 2, 2)   # 1 baris, 2 kolom, grafik ke-2
sns.histplot(df_local['UnitPrice'], bins=50, kde=True)
plt.title("Distribusi Harga Barang (Local)")
plt.xlabel("Unit Price")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
```
5.2.2. Menampilkan Harga Barang Tertinggi

```python
# Barang dengan harga termahal
# --- Menentukan data dulu ---
top_10_highest_global = df_global[['Description', 'UnitPrice']].drop_duplicates().sort_values(
    by='UnitPrice', ascending=False).head(10)

top_10_highest_local = df_local[['Description', 'UnitPrice']].drop_duplicates().sort_values(
    by='UnitPrice', ascending=False).head(10)

# Membuat color palette berbeda untuk tiap barang
colors_global = sns.color_palette("tab20", len(top_10_highest_global))
colors_local = sns.color_palette("tab20", len(top_10_highest_local))

# --- Plot berdampingan ---
plt.figure(figsize=(18,7))  # diperlebar

# Grafik kiri: Global
plt.subplot(1, 2, 1)
sns.barplot(
    data=top_10_highest_global,
    y='Description', x='UnitPrice',
    hue='Description',              # <--- warna berdasarkan nama barang
    palette=colors_global,
    legend=False                    # <--- legend tidak perlu
)

plt.title("Top 10 Barang Dengan Harga Tertinggi (Global)", fontsize=14)
plt.xlabel("Harga (Unit Price)")
plt.ylabel("Nama Barang")

# Tambahkan angka harga di ujung bar
for i, value in enumerate(top_10_highest_global['UnitPrice']):
    plt.text(value + 0.5, i, f"{value:.2f}", va='center', fontsize=9)

# Grafik kanan: Local
plt.subplot(1, 2, 2)
sns.barplot(
    data=top_10_highest_local,
    y='Description', x='UnitPrice',
    hue='Description',
    palette=colors_local,
    legend=False
)
plt.title("Top 10 Barang Dengan Harga Tertinggi (Local)", fontsize=14)
plt.xlabel("Harga (Unit Price)")
plt.ylabel("Nama Barang")

# Tambahkan angka harga di ujung bar
for i, value in enumerate(top_10_highest_local['UnitPrice']):
    plt.text(value + 0.5, i, f"{value:.2f}", va='center', fontsize=9)

plt.tight_layout()
plt.show()
```

5.2.3. Menampilkan Harga Barang Terendah
```python
# Barang dengan harga terendah
# --- Siapkan data barang terendah ---
top_10_lowest_global = df_global[df_global['UnitPrice'] > 0][['Description', 'UnitPrice']] \
    .drop_duplicates().sort_values(by='UnitPrice', ascending=True).head(10)

top_10_lowest_local = df_local[df_local['UnitPrice'] > 0][['Description', 'UnitPrice']] \
    .drop_duplicates().sort_values(by='UnitPrice', ascending=True).head(10)

# Buat warna berbeda untuk setiap barang
colors_global = sns.color_palette("tab10", len(top_10_lowest_global))
colors_local = sns.color_palette("tab10", len(top_10_lowest_local))

# --- Plot berdampingan ---
plt.figure(figsize=(18, 7))

# Grafik kiri: Global
plt.subplot(1, 2, 1)
sns.barplot(
    data=top_10_lowest_global,
    y='Description', x='UnitPrice',
    hue='Description',              # <--- warna berdasarkan nama barang
    palette=colors_global,
    legend=False                    # <--- legend tidak perlu
)
plt.title("Top 10 Barang Dengan Harga Termurah (Global)", fontsize=14)
plt.xlabel("Harga (Unit Price)")
plt.ylabel("Nama Barang")

# Tambahkan label harga di ujung bar
for i, value in enumerate(top_10_lowest_global['UnitPrice']):
    plt.text(value + 0.02, i, f"{value:.2f}", va='center', fontsize=9)

# Tambahkan informasi jumlah produk
plt.text(0.95, -0.15, f"Jumlah produk pada grafik ini: {len(top_10_lowest_global)}",
         ha='right', va='center', transform=plt.gca().transAxes, fontsize=10, style='italic')

# Grafik kanan: Local
plt.subplot(1, 2, 2)
sns.barplot(
    data=top_10_lowest_local,
    y='Description', x='UnitPrice',
    hue='Description',
    palette=colors_local,
    legend=False
)
plt.title("Top 10 Barang Dengan Harga Termurah (Local)", fontsize=14)
plt.xlabel("Harga (Unit Price)")
plt.ylabel("Nama Barang")

# Tambahkan label harga di ujung bar
for i, value in enumerate(top_10_lowest_local['UnitPrice']):
    plt.text(value + 0.02, i, f"{value:.2f}", va='center', fontsize=9)

# Tambahkan informasi jumlah produk
plt.text(0.95, -0.15, f"Jumlah produk pada grafik ini: {len(top_10_lowest_local)}",
         ha='right', va='center', transform=plt.gca().transAxes, fontsize=10, style='italic')

plt.tight_layout()
plt.show()
```

5.3. Insight Barang

5.3.1. Barang Paling Diminati
```python
# --- Data barang yang paling diminati ---
top_products_global = df_global.groupby('Description', as_index=False)['Quantity'].sum() \
                               .sort_values(by='Quantity', ascending=False).head(10)

top_products_local = df_local.groupby('Description', as_index=False)['Quantity'].sum() \
                             .sort_values(by='Quantity', ascending=False).head(10)

# Palet warna berbeda untuk tiap barang
colors_global = sns.color_palette("tab20", len(top_products_global))
colors_local = sns.color_palette("tab20", len(top_products_local))

# --- Visualisasi berdampingan ---
plt.figure(figsize=(18, 7))

# Grafik kiri: Global
plt.subplot(1, 2, 1)
sns.barplot(
    data=top_products_global,
    y='Description', x='Quantity',
    hue='Description', palette=colors_global, legend=False  # <–– warna tiap barang, tanpa warning
)
plt.title("Top 10 Barang Paling Diminati (Global)", fontsize=14)
plt.xlabel("Jumlah Quantity Terjual")
plt.ylabel("Nama Barang")

# Tambahkan angka quantity di ujung bar
for i, qty in enumerate(top_products_global['Quantity']):
    plt.text(qty + 0.5, i, f"{qty}", va='center', fontsize=9)

# Tambahkan informasi jumlah produk
plt.text(
    0.95, -0.15,
    f"Jumlah produk pada grafik ini: {len(top_products_global)}",
    ha='right', va='center', transform=plt.gca().transAxes, fontsize=10, style='italic'
)

# Grafik kanan: Local
plt.subplot(1, 2, 2)
sns.barplot(
    data=top_products_local,
    y='Description', x='Quantity',
    hue='Description', palette=colors_local, legend=False
)
plt.title("Top 10 Barang Paling Diminati (Local)", fontsize=14)
plt.xlabel("Jumlah Quantity Terjual")
plt.ylabel("Nama Barang")

# Tambahkan angka quantity di ujung bar
for i, qty in enumerate(top_products_local['Quantity']):
    plt.text(qty + 0.5, i, f"{qty}", va='center', fontsize=9)

# Tambahkan informasi jumlah produk
plt.text(
    0.95, -0.15,
    f"Jumlah produk pada grafik ini: {len(top_products_local)}",
    ha='right', va='center', transform=plt.gca().transAxes, fontsize=10, style='italic'
)

plt.tight_layout()
plt.show()

```

5.3.2. Barang Kurang Diminati
```python
# --- Data barang yang paling kurang diminati ---
least_products_global = df_global[df_global['Quantity'] > 0].groupby('Description', as_index=False)['Quantity'].sum() \
                                      .sort_values(by='Quantity', ascending=True).head(10)

least_products_local = df_local[df_local['Quantity'] > 0].groupby('Description', as_index=False)['Quantity'].sum() \
                                    .sort_values(by='Quantity', ascending=True).head(10)

# Buat warna berbeda untuk setiap barang
colors_global = sns.color_palette("tab20", len(least_products_global))
colors_local = sns.color_palette("tab20", len(least_products_local))

# --- Visualisasi berdampingan ---
plt.figure(figsize=(18, 7))

# Grafik kiri: Global
plt.subplot(1, 2, 1)
sns.barplot(
    data=least_products_global,
    y='Description', x='Quantity',
    hue='Description', palette=colors_global, legend=False
)
plt.title("Top 10 Barang yang Paling Kurang Diminati (Global)", fontsize=14)
plt.xlabel("Jumlah Quantity Dibeli")
plt.ylabel("Nama Barang")

# Tambahkan angka quantity di ujung bar
for i, qty in enumerate(least_products_global['Quantity']):
    plt.text(qty + 0.3, i, f"{qty}", va='center', fontsize=9)

# Tambahkan informasi jumlah produk
plt.text(
    0.95, -0.15,
    f"Jumlah produk pada grafik ini: {len(least_products_global)}",
    ha='right', va='center', transform=plt.gca().transAxes, fontsize=10, style='italic'
)

# Grafik kanan: Local
plt.subplot(1, 2, 2)
sns.barplot(
    data=least_products_local,
    y='Description', x='Quantity',
    hue='Description', palette=colors_local, legend=False
)
plt.title("Top 10 Barang yang Paling Kurang Diminati (Local)", fontsize=14)
plt.xlabel("Jumlah Quantity Dibeli")
plt.ylabel("Nama Barang")

# Tambahkan angka quantity di ujung bar
for i, qty in enumerate(least_products_local['Quantity']):
    plt.text(qty + 0.3, i, f"{qty}", va='center', fontsize=9)

# Tambahkan informasi jumlah produk
plt.text(
    0.95, -0.15,
    f"Jumlah produk pada grafik ini: {len(least_products_local)}",
    ha='right', va='center', transform=plt.gca().transAxes, fontsize=10, style='italic'
)

plt.tight_layout()
plt.show()

```

5.3.3. Hubungan Antara Banyak Barang dengan Harga Barang
```python
# --- Mengelompokkan harga rata-rata per produk & total quantity per produk ---
price_quantity_global = df_global.groupby('Description', as_index=False).agg({
    'UnitPrice': 'mean',
    'Quantity': 'sum'
})
price_quantity_local = df_local.groupby('Description', as_index=False).agg({
    'UnitPrice': 'mean',
    'Quantity': 'sum'
})

# --- Visualisasi hubungan harga vs jumlah pembelian ---
plt.figure(figsize=(16,6))

# Grafik Global
plt.subplot(1, 2, 1)
sns.regplot(data=price_quantity_global, x='UnitPrice', y='Quantity', scatter_kws={'alpha':0.5})
plt.title("Hubungan Harga vs Jumlah Pembelian (Global)")
plt.xlabel("Harga Rata-rata Produk")
plt.ylabel("Total Quantity Terjual")

# Grafik Local
plt.subplot(1, 2, 2)
sns.regplot(data=price_quantity_local, x='UnitPrice', y='Quantity', scatter_kws={'alpha':0.5}, color='red')
plt.title("Hubungan Harga vs Jumlah Pembelian (Local)")
plt.xlabel("Harga Rata-rata Produk")
plt.ylabel("Total Quantity Terjual")

plt.tight_layout()
plt.show()
```

5.4. Insight Pelanggan

5.4.1. Pelanggan Paling Sering Melakukan Transaksi
```python
# Menghitung frekuensi pembelian pelanggan (jumlah transaksi per CustomerID)
freq_global = df_global.groupby('CustomerID', as_index=False)['InvoiceNo'].count()
freq_global.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)

freq_local = df_local.groupby('CustomerID', as_index=False)['InvoiceNo'].count()
freq_local.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)

# Pelanggan paling sering membeli (Top 10)
top_customers_global = freq_global.sort_values(by='Frequency', ascending=False).head(10)
top_customers_local = freq_local.sort_values(by='Frequency', ascending=False).head(10)

# Pelanggan jarang membeli (Bottom 10)
least_customers_global = freq_global.sort_values(by='Frequency', ascending=True).head(10)
least_customers_local = freq_local.sort_values(by='Frequency', ascending=True).head(10)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.barplot(data=top_customers_global, y='CustomerID', x='Frequency', color='blue', ax=axes[0])
axes[0].set_title("Top 10 Pelanggan Paling Sering Membeli - Global")
axes[0].set_xlabel("Frekuensi Pembelian")
axes[0].set_ylabel("Customer ID")

sns.barplot(data=top_customers_local, y='CustomerID', x='Frequency', color='red', ax=axes[1])
axes[1].set_title("Top 10 Pelanggan Paling Sering Membeli - Local")
axes[1].set_xlabel("Frekuensi Pembelian")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()


```
5.4.2. Pelanggan Jarang Melakukan Transaksi
```python
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Global (warna biru)
sns.barplot(data=least_customers_global, y='CustomerID', x='Frequency', color='blue', ax=axes[0])
axes[0].set_title("Top 10 Pelanggan yang Jarang Membeli - Global")
axes[0].set_xlabel("Frekuensi Pembelian")
axes[0].set_ylabel("Customer ID")

# Local (warna merah)
sns.barplot(data=least_customers_local, y='CustomerID', x='Frequency', color='red', ax=axes[1])
axes[1].set_title("Top 10 Pelanggan yang Jarang Membeli - Local")
axes[1].set_xlabel("Frekuensi Pembelian")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

```
5.4.3. Banyak Pelanggan
```python
freq_global.head(10)
```

```python
jumlah_pengunjung_global = df_global['CustomerID'].nunique()
print(jumlah_pengunjung_global)
```

```python
freq_local.head(10)
```

```python
jumlah_pengunjung_local = df_local['CustomerID'].nunique()
print(jumlah_pengunjung_local)
```

5.5.  Insight Trend Penjualan dari Waktu ke Waktu

5.5.1 Total Penjualan dalam Tiap Bulan
```python
# Membuat kolom total belanja
df_global['TotalAmount'] = df_global['Quantity'] * df_global['UnitPrice']
df_local['TotalAmount'] = df_local['Quantity'] * df_local['UnitPrice']

# Membuat kolom bulan (format YYYY-MM)
df_global['Month'] = df_global['InvoiceDate'].dt.to_period('M').astype(str)
df_local['Month'] = df_local['InvoiceDate'].dt.to_period('M').astype(str)

# Hitung total belanja tiap bulan
monthly_spend_global = df_global.groupby('Month', as_index=False)['TotalAmount'].sum()
monthly_spend_local = df_local.groupby('Month', as_index=False)['TotalAmount'].sum()

# Convert 'Month' column to datetime objects AFTER creating the dataframes
monthly_spend_global['Month'] = pd.to_datetime(monthly_spend_global['Month'])
monthly_spend_local['Month'] = pd.to_datetime(monthly_spend_local['Month'])

plt.figure(figsize=(12,6))

plt.plot(monthly_spend_global['Month'], monthly_spend_global['TotalAmount'],
         marker='o', label='Global', color='blue')

plt.plot(monthly_spend_local['Month'], monthly_spend_local['TotalAmount'],
         marker='o', label='Local', color='red')

plt.title("Total Belanja Pelanggan per Bulan")
plt.xlabel("Bulan (Year-Month)")
plt.ylabel("Total Belanja")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```
5.5.2 Jumlah Pelanggan dalam Tiap Bulan
```python
# Membuat kolom bulan (jika belum ada)
df_global['Month'] = df_global['InvoiceDate'].dt.to_period('M').astype(str)
df_local['Month'] = df_local['InvoiceDate'].dt.to_period('M').astype(str)

# Menghitung jumlah pelanggan unik tiap bulan
cust_month_global = df_global.groupby('Month')['CustomerID'].nunique().reset_index()
cust_month_global.rename(columns={'CustomerID': 'CustomerCount'}, inplace=True)

cust_month_local = df_local.groupby('Month')['CustomerID'].nunique().reset_index()
cust_month_local.rename(columns={'CustomerID': 'CustomerCount'}, inplace=True)

# Convert ke datetime agar grafik berurutan
cust_month_global['Month'] = pd.to_datetime(cust_month_global['Month'])
cust_month_local['Month'] = pd.to_datetime(cust_month_local['Month'])

# Sorting
cust_month_global = cust_month_global.sort_values('Month')
cust_month_local = cust_month_local.sort_values('Month')

plt.figure(figsize=(12,6))

plt.plot(cust_month_global['Month'], cust_month_global['CustomerCount'],
         marker='o', label='Global', color='blue')

plt.plot(cust_month_local['Month'], cust_month_local['CustomerCount'],
         marker='o', label='Local', color='red')

plt.title("Jumlah Pelanggan yang Berbelanja per Bulan")
plt.xlabel("Bulan (Year-Month)")
plt.ylabel("Jumlah Pelanggan")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


```
5.5.3 Jam Pembelian Terbanyak
```python
# Membuat kolom jam (0–23)
df_global['Hour'] = df_global['InvoiceDate'].dt.hour
df_local['Hour'] = df_local['InvoiceDate'].dt.hour

transactions_hour_global = df_global.groupby('Hour', as_index=False)['InvoiceNo'].count()
transactions_hour_global.rename(columns={'InvoiceNo': 'TotalTransactions'}, inplace=True)

transactions_hour_local = df_local.groupby('Hour', as_index=False)['InvoiceNo'].count()
transactions_hour_local.rename(columns={'InvoiceNo': 'TotalTransactions'}, inplace=True)

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
sns.barplot(data=transactions_hour_global, x='Hour', y='TotalTransactions', color='blue')
plt.title("Distribusi Pembelian per Jam - Global")
plt.xlabel("Jam (0-23)")
plt.ylabel("Jumlah Transaksi")

plt.subplot(1,2,2)
sns.barplot(data=transactions_hour_local, x='Hour', y='TotalTransactions', color='red')
plt.title("Distribusi Pembelian per Jam - Local")
plt.xlabel("Jam (0-23)")
plt.ylabel("Jumlah Transaksi")

plt.tight_layout()
plt.show()

```
5.5.4 Hari Pembelian Terbanyak
```python
df_global['DayOfWeek'] = df_global['InvoiceDate'].dt.day_name()
df_local['DayOfWeek'] = df_local['InvoiceDate'].dt.day_name()

visits_per_day_global = df_global['DayOfWeek'].value_counts()
most_visited_day_global = visits_per_day_global.idxmax()
max_visits = visits_per_day_global.max()

visits_per_day_local = df_local['DayOfWeek'].value_counts()
most_visited_day_local = visits_per_day_local.idxmax()
max_visits_local = visits_per_day_local.max()

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
sns.barplot(x=visits_per_day_global.index, y=visits_per_day_global.values, hue=visits_per_day_global.index, palette='viridis', legend=False)
plt.title('Kunjungan Pelanggan per Hari dalam Seminggu (Global)')
plt.xlabel('Hari')
plt.ylabel('Jumlah Kunjungan')

plt.subplot(1,2,2)
sns.barplot(x=visits_per_day_local.index, y=visits_per_day_local.values, hue=visits_per_day_global.index, palette='viridis', legend=False)
plt.title('Kunjungan Pelanggan per Hari dalam Seminggu (Local)')
plt.xlabel('Hari')
plt.ylabel('Jumlah Kunjungan')

plt.tight_layout()
plt.show()

```

```python
customers_per_country = df.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False)

top_countries = customers_per_country.head(10)
plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(x=top_countries.index, y=top_countries.values, hue=top_countries.index, palette='viridis', legend=False)
plt.title('Distribusi Pelanggan per Negara (Top 10)')
plt.xlabel('Negara')
plt.ylabel('Jumlah Pelanggan')
plt.xticks(rotation=45)
# Tambahkan label persentase di atas batang (hitung persentase relatif terhadap total top 10)
total_top = top_countries.sum()
for p in bar_plot.patches:
    height = p.get_height()
    percentage = (height / total_top) * 100
    bar_plot.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2., height),
                      ha='center', va='bottom', fontsize=10, color='black')
```


```python
df_filtered = df[df['Country'] != 'United Kingdom']
customers_per_country = df_filtered.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False)
top_countries = customers_per_country.head(10)
plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(x=top_countries.index, y=top_countries.values, hue=top_countries.index, palette='viridis', legend=False)
plt.title('Distribusi Pelanggan per Negara (Top 10, Tanpa UK)')
plt.xlabel('Negara')
plt.ylabel('Jumlah Pelanggan')
plt.xticks(rotation=45)
# Tambahkan label persentase di atas batang (hitung persentase relatif terhadap total top 10)
total_top = top_countries.sum()
for p in bar_plot.patches:
    height = p.get_height()
    percentage = (height / total_top) * 100
    bar_plot.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2., height),
                      ha='center', va='bottom', fontsize=10, color='black')
plt.show()
```
### 6. Model RFM
Analisis RFM (Recency, Frequency, Monetary) adalah metode segmentasi pelanggan yang digunakan untuk memahami nilai dan perilaku pelanggan berdasarkan aktivitas pembelian mereka. RFM membantu mengidentifikasi pelanggan paling berharga, pelanggan baru, hingga pelanggan yang mulai tidak aktif.

--> Recency (R) — Seberapa baru pelanggan melakukan transaksi terakhir. Pelanggan yang baru berbelanja cenderung memiliki minat lebih tinggi.

--> Frequency (F) — Seberapa sering pelanggan melakukan transaksi. Pelanggan yang sering membeli menunjukkan loyalitas yang kuat.

--> Monetary (M) — Seberapa besar total uang yang dibelanjakan pelanggan. Pelanggan yang menghabiskan banyak uang memberikan nilai finansial besar bagi bisnis.

Dengan memberi skor pada setiap R, F, dan M kemudian menggabungkannya, dapat dilakukan pengelompokkan pelanggan menjadi segmen

```python
# menentukan tanggal acuan (snapshot date) pada perhitungan Recency dalam analisis RFM.
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
```

6.1 RFM Data Global

6.1.1 Menghitung Nilai RFM Data Global

```python
# Menghitung nilai RFM
rfm_global = df_global.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda date: (snapshot_date - date.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
)

rfm_global.columns = ['Recency', 'Frequency', 'Monetary']

rfm_global = rfm_global.reset_index()

rfm_global.head(10)
```

6.1.2 Mengubah Nilai RFM Menjadi Skor RFM (R-Score, F-Score, M-Score) dengan skala 1–5
```python
# Recency: makin kecil makin bagus → score tinggi
rfm_global['R_Score'] = pd.qcut(rfm_global['Recency'], 5, labels=[5,4,3,2,1]).astype(int)

# Frequency: makin besar makin bagus
rfm_global['F_Score'] = pd.qcut(rfm_global['Frequency'].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)

# Monetary: makin besar makin bagus
rfm_global['M_Score'] = pd.qcut(rfm_global['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)

```

6.1.3 Mengelompokkan Pelanggan (Customer Segmentation) Berdasarkan Perilaku
```python
rfm_global['RFM_Score'] = rfm_global[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

rfm_global['RFM_Score_Str'] = (
    rfm_global['R_Score'].astype(str) +
    rfm_global['F_Score'].astype(str) +
    rfm_global['M_Score'].astype(str)
)
```


```python
def segment_customer(row):

    # 1″ NEW CUSTOMER → Recency tinggi (baru belanja), Frequency dan Monetary masih rendah
    if row['R_Score'] >= 4 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
        return 'New Customer'

    # 2″ LOYAL → Belanja sering
    if row['F_Score'] >= 4 and row['M_Score'] >= 3:
        return 'Loyalty'

    # 3″ BIG SPENDER → Belanja besar (Monetary tinggi), meskipun tidak sering
    if row['M_Score'] >= 4 and row['F_Score'] >= 2:
        return 'Big Spenders'

    # 4″ AT RISK → Sudah lama tidak belanja (Recency rendah)
    if row['R_Score'] <= 2 and row['F_Score'] >= 2:
        return 'At Risk'

    # Default fallback
    return 'Need Attention'

# Terapkan ke RFM
rfm_global['Segment'] = rfm_global.apply(segment_customer, axis=1)

rfm_global.head()
```

6.1.4 Visualisasi RFM Data Global
```python
plt.figure(figsize=(20, 20))   # ukuran keseluruhan diperbesar

# ==========================================
# 1. BARPLOT SEGMENT (diputar horizontal & diperbesar)
# ==========================================
plt.subplot(3, 1, 1)  # <-- barplot mengisi 1 baris penuh
segment_table = rfm_global['Segment'].value_counts().sort_values(ascending=False)
segment_percent = (segment_table / segment_table.sum()) * 100

colors1 = plt.cm.tab20(range(len(segment_table)))
bars = plt.barh(segment_table.index, segment_table.values, color=colors1)  # horizontal bar

# Tambah persentase
for bar, pct in zip(bars, segment_percent):
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{pct:.1f}%", va='center', fontsize=10)

plt.title('Jumlah Customer per Segment', fontsize=16)
plt.xlabel('Jumlah Customer')
plt.ylabel('Segment')
plt.gca().invert_yaxis()  # biar ranking terbesar di atas

# ==========================================
# 2. SCATTER R vs M (warna = Segment)
# ==========================================
plt.subplot(3, 2, 3)
colors2 = {
    "Loyalty": "green",
    "At Risk": "red",
    "Big Spenders": "blue",
    "New Customer": "orange",
    "Need Attention": "grey"
}
plt.scatter(
    rfm_global['Recency'],
    rfm_global['Monetary'],
    s=rfm_global['Frequency'] * 3,
    c=rfm_global['Segment'].map(colors2),
    alpha=0.6
)
plt.title('Scatter: Recency vs Monetary (Size = Frequency)')
plt.xlabel('Recency')
plt.ylabel('Monetary')
for seg, col in colors2.items():
    plt.scatter([], [], c=col, label=seg)
plt.legend(title="Segment")

# ==========================================
# 3. SCATTER R vs M (warna = Segment & size = Frequency)
# ==========================================
plt.subplot(3, 2, 4)
plt.scatter(
    rfm_global['Recency'], rfm_global['Monetary'],
    s=rfm_global['Frequency'] * 4,
    c=rfm_global['Segment'].map(colors2),
    alpha=0.6
)
plt.title('Scatter: Recency vs Monetary (Size = Frequency)')
plt.xlabel('Recency')
plt.ylabel('Monetary')
for seg, col in colors2.items():
    plt.scatter([], [], c=col, label=seg)
plt.legend(title="Segment")

plt.tight_layout()
plt.show()
```


```python
# menampilkan tiga boxplot (Recency, Frequency, Monetary)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.boxplot(y=rfm_global['Recency'])
plt.title("Boxplot Recency")

plt.subplot(1, 3, 2)
sns.boxplot(y=rfm_global['Frequency'])
plt.title("Boxplot Frequency")

plt.subplot(1, 3, 3)
sns.boxplot(y=rfm_global['Monetary'])
plt.title("Boxplot Monetary")

plt.tight_layout()
plt.show()

```


```python
# Melihat distribusi nilai

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(rfm_global['Recency'], kde=True)
plt.title("Distribusi Recency")

plt.subplot(1, 3, 2)
sns.histplot(rfm_global['Frequency'], kde=True)
plt.title("Distribusi Frequency")

plt.subplot(1, 3, 3)
sns.histplot(rfm_global['Monetary'], kde=True)
plt.title("Distribusi Monetary")

plt.tight_layout()
plt.show()

```


```python
# Melihat hubungan antar variabel
plt.figure(figsize=(12, 4))
sns.pairplot(rfm_global[['Recency', 'Frequency', 'Monetary']])
plt.suptitle("Pairwise Relationship RFM", y=1.02)
plt.show()

```


```python
# Melihat korelasinya
plt.figure(figsize=(5,4))
sns.heatmap(rfm_global[['Recency','Frequency','Monetary']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap RFM Data Global")
plt.show()
```


```python
print("===== Analisis Otomatis Berdasarkan Segment =====")
for seg, count in segment_table.items():
    pct = (count / segment_table.sum()) * 100
    print(f"- {seg}: {count} customer ({pct:.1f}%)")

print("\n===== Insight =====")
print(f"Segmen terbesar  : {segment_table.idxmax()} ({segment_percent.max():.1f}%)")
print(f"Segmen terkecil  : {segment_table.idxmin()} ({segment_percent.min():.1f}%)")

if 'At Risk' in segment_table.index:
    print("Pelanggan yang berisiko hilang (At Risk) perlu kampanye win back.")
if 'Loyalty' in segment_table.index:
    print("Pelanggan Loyal perlu dijaga dengan program reward / membership.")
if 'Big Spenders' in segment_table.index:
    print("Big Spenders berpotensi untuk upselling & cross-selling.")
if 'New Customer' in segment_table.index:
    print("Pelanggan baru butuh onboarding & edukasi produk.")
if 'Need Attention' in segment_table.index:
    print("Need Attention butuh follow-up & promosi personalisasi.")
```

6.2 RFM untuk Data Lokal

6.2.1 Menghitung Nilai RFM Data Lokal
```python
# menghitung RFM data local
rfm_local = df_local.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm_local.columns = ['CustomerID','Recency','Frequency','Monetary']
rfm_local.head(10)
```

6.2.2 Mengubah Nilai RFM Menjadi Skor RFM (R-Score, F-Score, M-Score) dengan Skala 1–5
```python
# Recency: makin kecil makin bagus → score tinggi
rfm_local['R_Score'] = pd.qcut(rfm_local['Recency'], 5, labels=[5,4,3,2,1]).astype(int)

# Frequency: makin besar makin bagus
rfm_local['F_Score'] = pd.qcut(rfm_local['Frequency'].rank(method="first"),
                               5, labels=[1,2,3,4,5]).astype(int)

# Monetary: makin besar makin bagus
rfm_local['M_Score'] = pd.qcut(rfm_local['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)

```

6.2.3 Mengelompokkan Pelanggan (Customer Segmentation) Berdasarkan Perilaku
```python
# Menjumlahkan score R + F + M
rfm_local['RFM_Score'] = rfm_local[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

# Menggabungkan R, F, M menjadi satu string
rfm_local['RFM_Score_Str'] = (
    rfm_local['R_Score'].astype(str) +
    rfm_local['F_Score'].astype(str) +
    rfm_local['M_Score'].astype(str)
)

```


```python
def segment_customer(row):

    # 1″ NEW CUSTOMER → Recency tinggi (baru belanja), Frequency dan Monetary masih rendah
    if row['R_Score'] >= 4 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
        return 'New Customer'

    # 2″ LOYAL → Belanja sering
    if row['F_Score'] >= 4 and row['M_Score'] >= 3:
        return 'Loyalty'

    # 3″ BIG SPENDER → Belanja besar (Monetary tinggi), meskipun tidak sering
    if row['M_Score'] >= 4 and row['F_Score'] >= 2:
        return 'Big Spenders'

    # 4″ AT RISK → Sudah lama tidak belanja (Recency rendah)
    if row['R_Score'] <= 2 and row['F_Score'] >= 2:
        return 'At Risk'

    # Default fallback
    return 'Need Attention'

# Terapkan ke RFM
rfm_local['Segment'] = rfm_local.apply(segment_customer, axis=1)

rfm_local.head()
```

6.2.4 Visualisasi RFM Data Lokal
```python
plt.figure(figsize=(10, 10))   # ukuran keseluruhan diperbesar

# ==========================================
# 1. BARPLOT SEGMENT (diputar horizontal & diperbesar)
# ==========================================
plt.subplot(3, 1, 1)
segment_table = rfm_local['Segment'].value_counts().sort_values(ascending=False)
segment_percent = (segment_table / segment_table.sum()) * 100

colors1 = plt.cm.tab20(range(len(segment_table)))
bars = plt.barh(segment_table.index, segment_table.values, color=colors1)

# Tambah persentase
for bar, pct in zip(bars, segment_percent):
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{pct:.1f}%", va='center', fontsize=10)

plt.title('Jumlah Customer per Segment (Local)', fontsize=16)
plt.xlabel('Jumlah Customer')
plt.ylabel('Segment')
plt.gca().invert_yaxis()

# ==========================================
# 2. SCATTER R vs M (warna = Segment)
# ==========================================
plt.subplot(3, 2, 3)
colors2 = {
    "Loyalty": "green",
    "At Risk": "red",
    "Big Spenders": "blue",
    "New Customer": "orange",
    "Need Attention": "grey"
}

plt.scatter(
    rfm_local['Recency'],
    rfm_local['Monetary'],
    s=rfm_local['Frequency'] * 3,
    c=rfm_local['Segment'].map(colors2),
    alpha=0.6
)

plt.title('Scatter: Recency vs Monetary (Local) — Size = Frequency')
plt.xlabel('Recency')
plt.ylabel('Monetary')

for seg, col in colors2.items():
    plt.scatter([], [], c=col, label=seg)
plt.legend(title="Segment")

# ==========================================
# 3. SCATTER R vs M (warna = Segment & size = Frequency)
# ==========================================
plt.subplot(3, 2, 4)
plt.scatter(
    rfm_local['Recency'], rfm_local['Monetary'],
    s=rfm_local['Frequency'] * 4,
    c=rfm_local['Segment'].map(colors2),
    alpha=0.6
)

plt.title('Scatter: Recency vs Monetary (Local) — Size = Frequency (Bigger)')
plt.xlabel('Recency')
plt.ylabel('Monetary')

for seg, col in colors2.items():
    plt.scatter([], [], c=col, label=seg)
plt.legend(title="Segment")

plt.tight_layout()
plt.show()

```

```python
# Menampilkan boxplot dari Recency, Frequency, Monetary (Local Data)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.boxplot(y=rfm_local['Recency'])
plt.title("Boxplot Recency (Local)")

plt.subplot(1, 3, 2)
sns.boxplot(y=rfm_local['Frequency'])
plt.title("Boxplot Frequency (Local)")

plt.subplot(1, 3, 3)
sns.boxplot(y=rfm_local['Monetary'])
plt.title("Boxplot Monetary (Local)")

plt.tight_layout()
plt.show()

```

```python
# Melihat distribusi nilai (Local Data)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(rfm_local['Recency'], kde=True)
plt.title("Distribusi Recency (Local)")

plt.subplot(1, 3, 2)
sns.histplot(rfm_local['Frequency'], kde=True)
plt.title("Distribusi Frequency (Local)")

plt.subplot(1, 3, 3)
sns.histplot(rfm_local['Monetary'], kde=True)
plt.title("Distribusi Monetary (Local)")

plt.tight_layout()
plt.show()

```

```python
# Melihat hubungan antar variabel
plt.figure(figsize=(6, 2))
sns.pairplot(rfm_local[['Recency', 'Frequency', 'Monetary']])
plt.suptitle("Pairwise Relationship RFM (Local)", y=1.02)
plt.show()

```

```python
# Melihat korelasinya
plt.figure(figsize=(5,4))
sns.heatmap(rfm_local[['Recency','Frequency','Monetary']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap RFM Data Local")
plt.show()
```

```python
# ==========================================
# ANALISIS OTOMATIS RFM LOCAL
# ==========================================
segment_table = rfm_local['Segment'].value_counts().sort_values(ascending=False)
segment_percent = (segment_table / segment_table.sum()) * 100

print("===== Analisis Otomatis Berdasarkan Segment (RFM Local) =====")
for seg, count in segment_table.items():
    pct = (count / segment_table.sum()) * 100
    print(f"- {seg}: {count} customer ({pct:.1f}%)")

print("\n===== Insight =====")
print(f"Segmen terbesar  : {segment_table.idxmax()} ({segment_percent.max():.1f}%)")
print(f"Segmen terkecil  : {segment_table.idxmin()} ({segment_percent.min():.1f}%)")

# Insight tambahan berdasarkan keberadaan segmen
if 'At Risk' in segment_table.index:
    print("Pelanggan yang berisiko hilang (At Risk) perlu kampanye win-back, seperti diskon reaktivasi atau reminder pembelian.")
if 'Loyalty' in segment_table.index:
    print("Pelanggan Loyal perlu dijaga dengan program reward, prioritas layanan, atau membership eksklusif.")
if 'Big Spenders' in segment_table.index:
    print("Big Spenders berpotensi untuk upselling & cross-selling — tawarkan bundling atau produk premium.")
if 'New Customer' in segment_table.index:
    print("Pelanggan baru membutuhkan onboarding & edukasi produk agar pembelian tidak hanya satu kali.")
if 'Need Attention' in segment_table.index:
    print("Need Attention memerlukan follow-up personal, kupon penawaran, atau WhatsApp direct promo untuk menghindari churn.")
```

6.3 Standarisasi Nilai RFM
```python
# menstandarkan (menormalisasi) nilai RFM sebelum dilakukan clustering
scaler = MinMaxScaler()

rfm_global_scaled = scaler.fit_transform(rfm_global[['Recency','Frequency','Monetary']])
rfm_local_scaled = scaler.fit_transform(rfm_local[['Recency','Frequency','Monetary']])
```

```python
# ubah hasil scaler menjadi dataframe agar mudah divisualisasikan
rfm_global_scaled_df = pd.DataFrame(rfm_global_scaled, columns=["Recency_S", "Frequency_S", "Monetary_S"])
rfm_local_scaled_df = pd.DataFrame(rfm_local_scaled, columns=["Recency_S", "Frequency_S", "Monetary_S"])

plt.figure(figsize=(7.5,3))

plt.subplot(1,2,1)
sns.histplot(rfm_global_scaled_df, kde=True, palette="tab10")
plt.title("Distribusi RFM (Global, Scaled)", fontsize=14)
plt.xlabel("Skor Scaled (0 - 1)")
plt.ylabel("Jumlah Pelanggan")

plt.subplot(1,2,2)
sns.histplot(rfm_local_scaled_df, kde=True, palette="tab10")
plt.title("Distribusi RFM (Local, Scaled)", fontsize=14)
plt.xlabel("Skor Scaled (0 - 1)")
plt.ylabel("Jumlah Pelanggan")

plt.tight_layout()
plt.show()
```

### 7. K-Means Clustering

7.1 Mencari Nilai K
```python
K_range = range(3, 11)

def get_best_k(data_scaled):
    silhouette_scores = []
    inertia_scores = [] # To store SSE values

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        silhouette_scores.append(score)
        inertia_scores.append(kmeans.inertia_) # Store SSE

    best_k_original = K_range[np.argmax(silhouette_scores)]
    return best_k_original, silhouette_scores, inertia_scores # Return SSE as well

# Get silhouette and SSE scores
best_k_global_original, scores_global, sse_global = get_best_k(rfm_global_scaled)
best_k_local_original, scores_local, sse_local = get_best_k(rfm_local_scaled)

# Use the best k identified by the silhouette score
best_k_global = best_k_global_original
best_k_local = best_k_local_original


plt.figure(figsize=(8,4))
plt.plot(K_range, scores_global, marker='o')
plt.axvline(x=best_k_global, linestyle='--', color='red', label=f'Best k = {best_k_global}')
plt.title("Silhouette Scores (Global Data)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(K_range, scores_local, marker='o')
plt.axvline(x=best_k_local, linestyle='--', color='blue', label=f'Best k = {best_k_local}')
plt.title("Silhouette Scores (Local Data)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.show()

```
7.2 Clustering dengan K-Means
```python
kmeans_global = KMeans(n_clusters=best_k_global, random_state=42, n_init='auto')
kmeans_local = KMeans(n_clusters=best_k_local, random_state=42, n_init='auto')

rfm_global['Cluster'] = kmeans_global.fit_predict(rfm_global_scaled)
rfm_local['Cluster'] = kmeans_local.fit_predict(rfm_local_scaled)
```
7.2.1 Cluster K-Means – Data Global
```python
cluster_characteristics_global = rfm_global.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
print("Rata-rata Nilai RFM per Cluster (Global Data):")
display(cluster_characteristics_global)
```

Interpretasi Cluster K-Means – Data Global

Cluster 0

Recency rata-rata: 32 hari → pelanggan baru belanja

Frequency: 5.47 → cukup sering

Monetary: 1.726 → sangat tinggi

Kesimpulan: Ini adalah kelompok pelanggan aktif bernilai tinggi. Mereka baru belanja, sering bertransaksi, dan mengeluarkan total belanja terbesar. Kelompok ini sejalan dengan segmen seperti Loyalty atau Big Spenders.
Cluster 1

Recency rata-rata: 294 hari → sudah lama sekali tidak belanja

Frequency: 1.35 → jarang

Monetary: 324 → terendah

Kesimpulan: Kelompok ini masuk kategori pelanggan berisiko tinggi. Mereka hampir tidak aktif lagi dan cenderung mendekati churn. Mirip dengan segmen At Risk atau Need Attention.
Cluster 2

Recency rata-rata: 153 hari → cukup lama tidak bertransaksi

Frequency: 2.21 → jarang

Monetary: 580 → menengah

Kesimpulan: Kelompok ini adalah pelanggan kurang aktif. Aktivitas mereka menurun dalam beberapa waktu terakhir dan total belanja mereka tidak terlalu besar.

7.2.2 Confusion Matrix Data Global
```python
# Buat confusion matrix
segment_labels_global = sorted(rfm_global['Segment'].unique().tolist())
cluster_labels_global = sorted(rfm_global['Cluster'].unique().astype(str).tolist())

cm_global = pd.crosstab(index=rfm_global['Segment'],
                        columns=rfm_global['Cluster'].astype(str))

# Tampilkan tabel
display(cm_global)

# Heatmap
plt.figure(figsize=(8,4))
sns.heatmap(cm_global, annot=True, fmt='d', cmap='Blues',
            xticklabels=cluster_labels_global, yticklabels=segment_labels_global)
plt.title("Confusion Matrix (Global): RFM Segment vs K-Means Cluster")
plt.xlabel("K-Means Cluster")
plt.ylabel("RFM Segment")
plt.show()
```
7.2.3 Cluster K-Means – Data Lokal
```python
cluster_characteristics_local = rfm_local.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
print("Rata-rata Nilai RFM per Cluster (Local Data):")
display(cluster_characteristics_local)
```

Interpretasi Cluster K-Means – Data Lokal

Cluster 0

Recency: 293 hari → sangat lama tidak belanja

Frequency: 1.36 → rendah

Monetary: 306 → terendah

Kesimpulan: Karakteristiknya sangat mirip dengan Cluster 1 global. Ini adalah kelompok pelanggan berisiko tinggi yang sudah hampir tidak melakukan transaksi lagi.
Cluster 1

Recency: 154 hari

Frequency: 2.23

Monetary: 547

Kesimpulan: Ini adalah kelompok pelanggan kurang aktif. Mereka masih berpotensi diaktifkan kembali karena belum sepenuhnya hilang.
Cluster 2

Recency: 32 hari

Frequency: 5.41

Monetary: 1.592

Kesimpulan: Ini kelompok pelanggan aktif bernilai tinggi, sama seperti Cluster 0 pada data global. Mereka rutin melakukan transaksi dan memberikan kontribusi nilai terbesar.

7.2.4 Confusion Matrix Data Local
```python
# === CONFUSION MATRIX LOCAL ===

# Pastikan cluster sudah dibuat
kmeans_local = KMeans(n_clusters=best_k_local, random_state=42, n_init='auto')
rfm_local['Cluster'] = kmeans_local.fit_predict(rfm_local_scaled)

# Buat confusion matrix
segment_labels_local = sorted(rfm_local['Segment'].unique().tolist())
cluster_labels_local = sorted(rfm_local['Cluster'].unique().astype(str).tolist())

cm_local = pd.crosstab(index=rfm_local['Segment'],
                       columns=rfm_local['Cluster'].astype(str))

# Tampilkan tabel
display(cm_local)

# Heatmap
plt.figure(figsize=(8,4))
sns.heatmap(cm_local, annot=True, fmt='d', cmap='Oranges',
            xticklabels=cluster_labels_local, yticklabels=segment_labels_local)
plt.title("Confusion Matrix (Local): RFM Segment vs K-Means Cluster")
plt.xlabel("K-Means Cluster")
plt.ylabel("RFM Segment")
plt.show()

```

### 8. Interpretasi Hasil

8.1 Interpretasi Data Global
```python
plt.figure(figsize=(18, 5))

# Scatter Plot Recency vs Monetary (Global)
plt.subplot(1, 3, 1)
plt.scatter(rfm_global['Recency'], rfm_global['Monetary'], c=rfm_global['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Global: Recency vs Monetary")
plt.colorbar(label='Cluster')

# Scatter Plot Recency vs Frequency (Global)
plt.subplot(1, 3, 2)
plt.scatter(rfm_global['Recency'], rfm_global['Frequency'], c=rfm_global['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.title("Global: Recency vs Frequency")
plt.colorbar(label='Cluster')

# Scatter Plot Frequency vs Monetary (Global)
plt.subplot(1, 3, 3)
plt.scatter(rfm_global['Frequency'], rfm_global['Monetary'], c=rfm_global['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Global: Frequency vs Monetary")
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()
```
8.2 Interpretasi Data lokal
```python
plt.figure(figsize=(18, 5))

# Scatter Plot Recency vs Monetary (Local)
plt.subplot(1, 3, 1)
plt.scatter(rfm_local['Recency'], rfm_local['Monetary'], c=rfm_local['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Local: Recency vs Monetary")
plt.colorbar(label='Cluster')

# Scatter Plot Recency vs Frequency (Local)
plt.subplot(1, 3, 2)
plt.scatter(rfm_local['Recency'], rfm_local['Frequency'], c=rfm_local['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.title("Local: Recency vs Frequency")
plt.colorbar(label='Cluster')

# Scatter Plot Frequency vs Monetary (Local)
plt.subplot(1, 3, 3)
plt.scatter(rfm_local['Frequency'], rfm_local['Monetary'], c=rfm_local['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Local: Frequency vs Monetary")
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()
```

8.3. Mengunduh hasil klaster dalam bentuk CSV
```python
from google.colab import files
rfm_global.to_csv('hasil_rfm_kmeans.csv', index=False)
files.download('hasil_rfm_kmeans.csv')
rfm_local.to_csv('hasil_rfm_kmeans_lokal.csv', index=False)
files.download('hasil_rfm_kmeans_lokal.csv')
```
### 9. Strategi Pemasaran

9.1 Strategi Pelanggan Global

Membuat program VIP yang menawarkan berbagai keuntungan khusus, seperti akses lebih awal (early bird) untuk produk baru, voucher eksklusif, serta promo yang hanya tersedia untuk pelanggan VIP.
Berikan promo pemicu impulsif seperti diskon besar, flash sale, event comeback, dan voucher pembelian berikutnya, serta program tebus murah untuk meningkatkan nilai transaksi.
Bundling Produk dengan menggabungkan produk yang paling diminati dengan produk yang kurang diminati dalam satu paket penjualan.
Tawarkan hadiah khusus bagi pelanggan yang berbelanja pada momen tertentu seperti akhir tahun atau event khusus brand.
Tampilkan testimoni dan konten edukasi yang menunjukkan nilai serta cara penggunaan produk untuk meningkatkan kepercayaan dan membantu pelanggan memahami manfaatnya.

9.2 Strategi Pelanggan Lokal

Promosi dan social proof dengan menggabungkan penggunaan testimoni, social proof, dan konten edukasi produk untuk meningkatkan kepercayaan pelanggan dan membantu mereka memahami manfaat produk.
Promo penjualan & diskon seperti diskon tengah malam, voucher untuk pembelian berikutnya, dan bundling produk populer dengan produk kurang diminati untuk meningkatkan nilai transaksi.
Mengadakan kompetisi berhadiah dengan syarat tertentu (misal minimum pembelian) dan program belanja berhadiah, terutama pada periode akhir tahun, untuk mendorong interaksi dan loyalitas pelanggan.
Membuat program loyalty VIP lokal yang memberikan keuntungan eksklusif, seperti akses awal ke produk baru dan penawaran khusus bagi member.
Berikan hadiah atau penghargaan tahunan untuk pelanggan sebagai bentuk apresiasi, memperkuat hubungan jangka panjang, dan meningkatkan retensi.



