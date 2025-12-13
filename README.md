# Judul Proyek : Pengelompokan Pelanggan Berbasis RFM dengan Algoritma K-Means untuk Strategi Marketing Terpersonalisasi di Industri Ritel

## Deskripsi Singkat

Pada proyek ini dilakukan segmentasi pelanggan menggunakan RFM dan algoritma K-Means untuk memahami perilaku pelanggan guna memaksimalkan strategi pemasaran pada tiap segmennya. Data yang digunakan pada proyek ini bersumber dari kaggle dengan kumpulan data tahun 2010-2011. Dataset dibagi menjadi 2, yaitu data global yang mencakup seluruh data, serta data lokal yang berisikan pelanggan dari negara United Kingdom (UK)
## Tujuan Proyek

Tujuan dari proyek ini:
* Memperoleh segmentasi pelanggan untuk optimalisasi marketing

## Rumusan Masalah
Rumusan masalah proyek ini: 
* Segmen pelanggan seperti apa yang dapat diidentifikasi berdasarkan data transaksi? 
* Bagaimana Algoritma K-Means Clustering secara efektif mengidentifikasi perilaku pelanggan?
* Bagaimana segmentasi ini dapat memberi informasi dan meningkatkan strategi pemasaran terarah untuk setiap segmen pelanggan yang telah diidentifikasi?
## Dataset
Dataset diambil melalui Kaggle.

Berikut adalah tautan dataset yang digunakan

https://www.kaggle.com/code/akashchola/customer-segmentation-rfm-model-k-means/input
## Tools & Dependencies
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
## Struktur Repository
├── public/

│ └── hasil_rfm_kmeans.csv

│ └── hasil_rfm_kmeans_lokal.csv

├── A25_CS311_CAPSTONE (1).ipynb

├── Online Retail.xlsx 

├── hasil_rfm_kmeans.csv

├── hasil_rfm_kmeans_lokal.csv

├── index.html

└── README.md
## Cara Menjalankan Proyek

1. Buka file notebook `A25_CS311_CAPSTONE (1).ipynb)` menggunakan Google Colab atau melalui tautan beriku

   https://github.com/hilmatul23002/A25-CS311---AC-06-Customer-Segmentation-for-Personalized-Retail-Marketing/blob/main/A25_CS311_CAPSTONE%20(1).ipynb
3. Jalankan seluruh cell secara berurutan dari atas ke bawah
4. Dataset akan otomatis diunduh dari Kaggle menggunakan library `kagglehub`
5. Hasil segmentasi pelanggan ditampilkan dalam bentuk:
   - Segmentasi RFM
   - Cluster K-Means
   - Confusion matriks untuk RFM dan K-Means
   - Visualisasi dan insight marketing

## Metodologi Singkat

Analisis segmentasi pelanggan pada proyek ini dilakukan melalui tahapan berikut:

1. **Data Loading**  
    Pada proses ini, data langsung diambil dari kaggle tanpa harus mengunduh file dan menyimpah file pada perangkat terlebih dahulu.

    Tahap ini bertujuan untuk mengunduh dataset *Online Retail* dari Kaggle, membaca file Excel, lalu mengonversinya ke format CSV agar lebih mudah diproses pada tahap selanjutnya.

2. **Data Cleaning**  
   Melakukan pembersihan data duplikat, data kosong, data bernilai negatif, data dengan status pembayaran cancel, serta membersihkan outlier
   
4. **Membagi Data Global dan Lokal**  
  Dataset dibagi menjadi 2, yaitu data global mencakup seluruh data, dan data lokal yang berisikan pelanggan dari negara United Kingdom (UK)

6. **Exploratory Data Analysis (EDA)**  
   Tujuan Utama dari EDA ini adalah untuk memahami karakteristik data transaksi penjualan, mengidentifikasi pola pembelian, insight terkait produk, harga, pelanggan, serta tren penjualan dari     waktu ke waktu.

    Analisis mencakup beberapa bagian utama:
    * Analisis harga barang (distribusi, barang termurah & termahal)
    * Analisis performa produk (produk paling laku & paling jarang dibeli)
    * Analisis perilaku pelanggan (frekuensi pembelian dan total belanja)
    * Analisis tren waktu (jumlah transaksi & total belanja per bulan)
      
    EDA ini dilakukan pada dua data:
    * Data Global (all customers)
    * Data Local (local customers)

7. **Menentukan Nilai RFM dan Standarisasi RFM**  
    Membentuk variabel RFM (Recency, Frequency, Monetary) dari data transaksi pelanggan.
   Membagi pelanggan menjadi beberapa segmen berdasarkan nilai RFM-nya.
   Standarisasi nilai RFM.
9. **Clustering Menggunakan K-Means**  
   Menentukan nilai best K menggunakan elbow method dan silhoutte score.
   Menerapkan algoritma K-Means untuk mengelompokkan pelanggan berdasarkan nilai RFM.
    Hasilclustering divisualisasikan dalam bentuk grafik dan confusion matrix.
11. **Insight dan Rekomendasi**
    Berisi hasiol / output yang diperoleh dan strategi marketing yang disarankan.
    
## Output / Hasil
**Interpretasi Cluster K-Means – Data Global**

**Cluster 0**

* Recency rata-rata: 32 hari → pelanggan baru belanja
* Frequency: 5.47 → cukup sering
* Monetary: 1.726 → sangat tinggi

Kesimpulan: Ini adalah kelompok pelanggan aktif bernilai tinggi. Mereka baru belanja, sering bertransaksi, dan mengeluarkan total belanja terbesar. Kelompok ini sejalan dengan segmen seperti Loyalty atau Big Spenders.

**Cluster 1**

* Recency rata-rata: 294 → sudah lama sekali tidak belanja
* Frequency: 1.35 → jarang
* Monetary: 324 → terendah

Kesimpulan: Kelompok ini masuk kategori pelanggan berisiko tinggi. Mereka hampir tidak aktif lagi dan cenderung mendekati churn. Mirip dengan segmen At Risk atau Need Attention.

**Cluster 2**

* Recency rata-rata: 153 hari → cukup lama tidak bertransaksi
* Frequency: 2.21 → jarang
* Monetary: 580 → menengah

Kesimpulan: Kelompok ini adalah pelanggan kurang aktif. Aktivitas mereka menurun dalam beberapa waktu terakhir dan total belanja mereka tidak terlalu besar.


**Interpretasi Cluster K-Means – Data Lokal**

**Cluster 0**

* Recency: 293 hari → sangat lama tidak belanja
* Frequency: 1.36 → rendah
* Monetary: 306 → terendah

Kesimpulan: Karakteristiknya sangat mirip dengan Cluster 1 global. Ini adalah kelompok pelanggan berisiko tinggi yang sudah hampir tidak melakukan transaksi lagi.

**Cluster 1**

* Recency: 154 hari
* Frequency: 2.23
* Monetary: 547

Kesimpulan: Ini adalah kelompok pelanggan kurang aktif. Mereka masih berpotensi diaktifkan kembali karena belum sepenuhnya hilang.

**Cluster 2**

* Recency: 32 hari
* Frequency: 5.41
*Monetary: 1.592

Kesimpulan: Ini kelompok pelanggan aktif bernilai tinggi, sama seperti Cluster 0 pada data global. Mereka rutin melakukan transaksi dan memberikan kontribusi nilai terbesar.

## Strategi Marketing
**1. Strategi Pelanggan Global**
* Membuat program VIP yang menawarkan berbagai keuntungan khusus, seperti akses lebih awal (early bird) untuk produk baru, voucher eksklusif, serta promo yang hanya tersedia untuk pelanggan VIP.
* Berikan promo pemicu impulsif seperti diskon besar, flash sale, event comeback, dan voucher pembelian berikutnya, serta program tebus murah untuk meningkatkan nilai transaksi.
* Bundling Produk dengan menggabungkan produk yang paling diminati dengan produk yang kurang diminati dalam satu paket penjualan.
* Tawarkan hadiah khusus bagi pelanggan yang berbelanja pada momen tertentu seperti akhir tahun atau event khusus brand.
* Tampilkan testimoni dan konten edukasi yang menunjukkan nilai serta cara penggunaan produk untuk meningkatkan kepercayaan dan membantu pelanggan memahami manfaatnya.

**2 Strategi Pelanggan Lokal**

* Promosi dan social proof dengan menggabungkan penggunaan testimoni, social proof, dan konten edukasi produk untuk meningkatkan kepercayaan pelanggan dan membantu mereka memahami manfaat produk.
* Promo penjualan & diskon seperti diskon tengah malam, voucher untuk pembelian berikutnya, dan bundling produk populer dengan produk kurang diminati untuk meningkatkan nilai transaksi.
* Mengadakan kompetisi berhadiah dengan syarat tertentu (misal minimum pembelian) dan program belanja berhadiah, terutama pada periode akhir tahun, untuk mendorong interaksi dan loyalitas pelanggan.
* Membuat program loyalty VIP lokal yang memberikan keuntungan eksklusif, seperti akses awal ke produk baru dan penawaran khusus bagi member.
* Berikan hadiah atau penghargaan tahunan untuk pelanggan sebagai bentuk apresiasi, memperkuat hubungan jangka panjang, dan meningkatkan retensi.

## Website
Seluruh hasil dari proses ini di tampilkan melaui dashboard dalam bentuk website.

Dashboard diperoleh dari mendeploy fila index.html menggunakan netlify.

Berikut adalah tautan website:

https://segmentasic331.netlify.app/
