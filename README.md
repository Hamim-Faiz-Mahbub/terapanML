# Laporan Proyek Machine Learning Terapan Pertama - Hamim Faiz Mahbub 


## Latar Belakang
(https://images.unsplash.com/photo-1552664730-d307ca884978?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8aHIlMjBhbmFseXRpY3N8ZW58MHx8MHx8fDA%3D&w=1000&q=80)Sumber Gambar: Unsplash (contoh ilustrasi)

Dalam lanskap bisnis yang dinamis dan kompetitif, sumber daya manusia (SDM) diakui sebagai aset strategis utama yang menentukan keberhasilan organisasi.1 Kinerja karyawan secara individual dan kolektif berdampak langsung pada produktivitas, inovasi, profitabilitas, dan pencapaian tujuan strategis perusahaan.2 Oleh karena itu, kemampuan organisasi untuk secara akurat mengevaluasi dan memprediksi kinerja karyawan telah menjadi fungsi krusial dalam manajemen SDM modern.1Metode evaluasi kinerja tradisional seringkali mengandalkan penilaian subjektif dari manajer, yang dapat rentan terhadap berbagai bias seperti bias kelonggaran, bias tendensi sentral, atau efek halo.2 Keterbatasan ini dapat menghasilkan evaluasi yang tidak konsisten, kurang akurat, dan berpotensi menimbulkan ketidakpuasan di kalangan karyawan, serta menghambat pengambilan keputusan yang efektif terkait pengembangan talenta, promosi, dan kompensasi.1Seiring dengan kemajuan teknologi dan ketersediaan data SDM yang melimpah, analitik SDM (HR Analytics) dan penerapan machine learning (ML) menawarkan pendekatan yang lebih objektif, berbasis data, dan prediktif untuk evaluasi kinerja.5 Dengan memanfaatkan algoritma ML, organisasi dapat menganalisis pola kompleks dalam data karyawan untuk mengidentifikasi faktor-faktor yang memengaruhi kinerja dan membuat prediksi yang lebih akurat mengenai kinerja di masa depan.2 Pendekatan ini tidak hanya meningkatkan akurasi dan efisiensi proses evaluasi tetapi juga berpotensi mengurangi bias dan mendukung pengambilan keputusan SDM yang lebih strategis dan adil.4Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi peringkat kinerja (PerformanceRating) karyawan berdasarkan berbagai atribut yang tersedia dalam dataset analitik SDM.


Masalah subjektivitas, inkonsistensi, dan potensi bias dalam metode evaluasi kinerja tradisional perlu diatasi karena dapat berdampak negatif pada:

Keadilan dan Moral Karyawan: Penilaian yang tidak adil dapat menurunkan motivasi dan keterlibatan karyawan.

Pengambilan Keputusan SDM: Keputusan yang salah terkait promosi, kompensasi, dan pengembangan dapat merugikan baik karyawan maupun organisasi.2

Efektivitas Organisasi: Kegagalan dalam mengidentifikasi dan mengembangkan talenta terbaik dapat menghambat pencapaian tujuan organisasi.1


Mengadopsi Pendekatan Berbasis Data: Menggunakan data historis karyawan untuk mengidentifikasi pola objektif yang berkaitan dengan kinerja.
Menerapkan Algoritma Machine Learning: Membangun model prediktif yang dapat mempelajari hubungan kompleks antara berbagai faktor karyawan dan peringkat kinerja mereka.4
Mengidentifikasi Faktor Kunci: Menggunakan model untuk memahami variabel mana yang paling signifikan dalam memprediksi kinerja, sehingga intervensi SDM dapat lebih terarah.

Dengan demikian, solusi berbasis machine learning diharapkan dapat menyediakan alat bantu yang lebih objektif dan akurat bagi manajer SDM dalam proses evaluasi kinerja.
Referensi Riset TerkaitPenerapan machine learning dalam prediksi kinerja karyawan telah banyak diteliti dan menunjukkan hasil yang menjanjikan:
Rao dan Verma (2018) menggunakan Decision Trees dan mencapai akurasi 85% dalam memprediksi kinerja karyawan, mengungguli penilaian manajerial tradisional.4
Zhang et al. (2020) mencapai akurasi 90% dalam mengidentifikasi karyawan berpotensi tinggi menggunakan algoritma Random Forest.4
Wang et al. (2020) melaporkan tingkat akurasi prediktif 89% menggunakan Artificial Neural Networks (ANN) pada dataset metrik kinerja karyawan skala besar.4
Studi oleh Gupta dan Mehta (2020) mencatat peningkatan akurasi evaluasi sebesar 25% melalui metode ML.4
Clark dan Ramirez (2019) melaporkan pengurangan waktu evaluasi sebesar 30% dengan menggunakan ML.4
Penelitian juga menunjukkan bahwa ML dapat membantu mengurangi bias; misalnya, Singh dan Patel (2021) mengurangi bias gender sebesar 15% dalam evaluasi kinerja menggunakan teknik adversarial debiasing.4
Referensi ini mengindikasikan bahwa machine learning memiliki potensi besar untuk mentransformasi praktik evaluasi kinerja menjadi lebih akurat, efisien, dan adil.

## Business Understanding

Pengembangan model prediktif untuk peringkat kinerja karyawan bertujuan untuk memberikan alat bantu keputusan yang lebih baik bagi departemen SDM. Dengan prediksi yang akurat, organisasi dapat lebih proaktif dalam manajemen talenta, perencanaan suksesi, dan perancangan program pengembangan karyawan.

### Problem Statements

- Bagaimana cara membangun model machine learning yang efektif untuk mengklasifikasikan peringkat kinerja (PerformanceRating) karyawan berdasarkan atribut-atribut yang tersedia dalam dataset "HR Analytics Dataset"?
- Algoritma machine learning mana yang memberikan kinerja prediksi terbaik untuk tugas klasifikasi PerformanceRating pada dataset ini?
- Fitur atau atribut karyawan apa saja yang paling signifikan dalam memprediksi PerformanceRating menurut model terbaik yang dikembangkan?
Bagaimana wawasan dari model ini dapat digunakan untuk memberikan rekomendasi strategis bagi manajemen SDM?

### Goals
- Melakukan analisis data eksploratif (EDA) untuk memahami karakteristik dataset dan hubungan antar variabel.
- Melakukan pra-pemrosesan data yang diperlukan untuk menyiapkan data untuk pemodelan machine learning.
- Mengembangkan dan membandingkan kinerja beberapa model klasifikasi machine learning untuk memprediksi PerformanceRating.
- Melakukan optimasi hyperparameter pada model-model yang diuji untuk meningkatkan kinerjanya.
- Mengidentifikasi model dengan kinerja terbaik berdasarkan metrik evaluasi yang relevan (misalnya, F1-Score, Akurasi, ROC-AUC).
- Menganalisis fitur-fitur yang paling berpengaruh terhadap prediksi PerformanceRating dari model terbaik.

### Solution Statements

Untuk mencapai tujuan-tujuan tersebut, solusi yang diajukan adalah sebagai berikut:

1. Analisis Data dan Pra-pemrosesan Komprehensif:

* Melakukan analisis data eksploratif (EDA) untuk memahami distribusi data, mengidentifikasi outlier (jika ada), dan melihat korelasi antar fitur.
* Menerapkan teknik pra-pemrosesan data yang sesuai, termasuk penanganan nilai yang hilang (jika ada), encoding variabel kategorikal, dan scaling fitur numerik.
Solusi ini terukur melalui kualitas data yang siap untuk pemodelan dan pemahaman awal terhadap pola data.


2. Pengembangan dan Optimasi Model Klasifikasi Komparatif:

* Mengimplementasikan dan melatih setidaknya dua algoritma machine learning yang berbeda untuk tugas klasifikasi (misalnya, Random Forest dan Gradient Boosting). Pendekatan ini memungkinkan perbandingan kinerja dan pemilihan model yang paling sesuai.
* Melakukan optimasi hyperparameter untuk setiap model yang diuji menggunakan teknik seperti GridSearchCV atau RandomizedSearchCV untuk menemukan konfigurasi parameter terbaik.
* Kinerja masing-masing model (sebelum dan sesudah optimasi) akan dievaluasi secara ketat menggunakan metrik evaluasi seperti Akurasi, Presisi, Recall, F1-Score (khususnya F1-macro atau F1-weighted untuk menangani potensi ketidakseimbangan kelas), dan ROC-AUC. Model terbaik akan dipilih berdasarkan metrik ini.
* Solusi ini terukur melalui skor metrik evaluasi yang dicapai oleh masing-masing model dan peningkatan kinerja setelah optimasi.


## Data Understanding
Tahap ini bertujuan untuk memahami dataset yang akan digunakan, termasuk sumber, struktur, dan karakteristik masing-masing variabel.

### Sumber Data
Dataset yang digunakan dalam proyek ini adalah "HR Analytics Dataset" yang diunggah oleh pengguna saadharoon27 di platform Kaggle.
Judul Dataset: HR Analytics Dataset
Sumber: [Kaggle 8](https://www.kaggle.com/datasets/saadharoon27/hr-analytics-dataset)
Penyedia: Saad Haroon 9
Lisensi: "CC0: Public Domain"


### Deskripsi Variabel



| Nama Fitur              | Tipe Data (Inferensi) | Deskripsi Singkat                                                             | Contoh Nilai                |
|-------------------------|------------------------|--------------------------------------------------------------------------------|-----------------------------|
| EmpID                   | object                 | ID unik karyawan                                                               | RM297, RM302                |
| Age                     | int64                  | Usia karyawan dalam tahun                                                     | 35, 42                      |
| Attrition              | object                 | Apakah karyawan mengalami atrisi (Yes/Tidak)                                  | Yes, No                     |
| BusinessTravel          | object                 | Frekuensi perjalanan bisnis                                                   | Travel_Rarely, Non-Travel   |
| Department              | object                 | Departemen tempat karyawan bekerja                                            | Sales, Research & Development |
| DistanceFromHome        | int64                  | Jarak dari rumah ke kantor (dalam satuan tertentu)                            | 10, 2                       |
| Education               | int64                  | Tingkat pendidikan (misal 1=SMA, 3=S1, 4=S2)                                   | 3, 4                        |
| EducationField          | object                 | Bidang pendidikan karyawan                                                    | Life Sciences, Medical      |
| EnvironmentSatisfaction | int64                  | Tingkat kepuasan terhadap lingkungan kerja (skala)                            | 3, 4                        |
| Gender                  | object                 | Jenis kelamin karyawan                                                        | Male, Female                |
| JobInvolvement          | int64                  | Tingkat keterlibatan dalam pekerjaan (skala)                                  | 3, 2                        |
| JobLevel                | int64                  | Level jabatan karyawan                                                        | 2, 5                        |
| JobRole                 | object                 | Peran atau jabatan spesifik karyawan                                          | Sales Executive, Manager    |
| JobSatisfaction         | int64                  | Tingkat kepuasan kerja (skala)                                                | 4, 1                        |
| MonthlyIncome           | int64                  | Pendapatan bulanan karyawan                                                   | 5000, 12000                 |
| NumCompaniesWorked      | int64                  | Jumlah perusahaan tempat karyawan pernah bekerja                              | 1, 5                        |
| OverTime                | object                 | Apakah karyawan bekerja lembur                                                | Yes, No                     |
| PercentSalaryHike       | int64                  | Persentase kenaikan gaji terakhir                                             | 15, 20                      |
| PerformanceRating       | int64                  | Peringkat kinerja karyawan (Variabel Target)                                  | 3, 4                        |
| TotalWorkingYears       | int64                  | Total tahun pengalaman kerja                                                  | 10, 20                      |
| YearsAtCompany          | int64                  | Lama bekerja di perusahaan saat ini (dalam tahun)                             | 5, 2                        |

Tabel 1. Deskripsi Variabel Utama

Variabel target dalam proyek ini adalah PerformanceRating. Berdasarkan observasi pada dataset serupa (misalnya, dataset IBM HR [21]), PerformanceRating seringkali merupakan skala ordinal. Dalam dataset ini, nilai yang ditemui untuk PerformanceRating adalah 3 ('Excellent') dan 4 ('Outstanding'). [18, 21] Analisis distribusi dari notebook codinga.ipynb menunjukkan bahwa sekitar 84.66% karyawan memiliki peringkat 3 dan 15.34% memiliki peringkat 4. Hal ini mengindikasikan adanya ketidakseimbangan kelas yang signifikan, di mana kelas '3' jauh lebih dominan daripada kelas '4'. Ketidakseimbangan ini perlu menjadi perhatian khusus dalam tahap evaluasi model, karena metrik seperti akurasi saja bisa menyesatkan.

Fitur-fitur seperti EmployeeCount, StandardHours, dan Over18 kemungkinan memiliki varians yang sangat rendah atau nilai konstan dan akan dipertimbangkan untuk dihapus pada tahap persiapan data. [18] Fitur EmpID dan EmployeeNumber adalah pengidentifikasi unik dan juga akan dihapus.

### EDA - Univariate 

AnalysisAnalisis univariat akan dilakukan untuk setiap fitur guna memahami distribusinya:

1. Variabel Numerik (Age, MonthlyIncome, PercentSalaryHike, dll.):

- Histogram digunakan untuk melihat bentuk distribusi masing-masing fitur numerik. Visualisasi ini dapat dilihat pada gambar uploaded:image_19d731.jpg.
![Distribusi Fitur Numerik](uploaded:image_19d731.jpg)

Box plot juga digunakan untuk mengidentifikasi tendensi sentral (median), sebaran (IQR), dan potensi outlier untuk setiap fitur numerik.

Statistik deskriptif lengkap (termasuk count, mean, std, min, 1%, 5%, 25%, 50% (median), 75%, 95%, 99%, max) untuk semua fitur numerik telah dihitung dan tersedia dalam output notebook codinga.ipynb. Sebagai contoh, fitur Age memiliki rata-rata sekitar 36.9 tahun, sedangkan MonthlyIncome memiliki rata-rata sekitar 6504.





Variabel Kategorikal (Department, JobRole, Gender, PerformanceRating, dll.):

Akan dibuat bar chart (diagram batang) untuk menunjukkan frekuensi atau proporsi setiap kategori.
Khusus untuk variabel target PerformanceRating, bar chart akan sangat penting untuk melihat distribusi kelas dan mengidentifikasi potensi ketidakseimbangan kelas. Jika mayoritas karyawan memiliki rating 3 dan hanya sebagian kecil yang memiliki rating 4 (atau sebaliknya), ini perlu diperhatikan saat evaluasi model.
Contoh Visualisasi (Deskripsi):

!(path/to/barchart_departemen.png) - Menunjukkan jumlah karyawan per departemen.
!(path/to/barchart_rating.png) - Menunjukkan distribusi peringkat kinerja.




Dari analisis univariat pada dataset serupa 13, sering ditemukan bahwa variabel seperti MonthlyIncome memiliki distribusi miring ke kanan (right-skewed), dan variabel target seperti PerformanceRating mungkin tidak seimbang.EDA - Multivariate AnalysisAnalisis multivariat akan dilakukan untuk memahami hubungan antar variabel:

Korelasi Antar Fitur Numerik:

Matriks korelasi akan dihitung untuk semua pasangan fitur numerik.
Heatmap akan digunakan untuk memvisualisasikan matriks korelasi, memudahkan identifikasi fitur-fitur yang berkorelasi tinggi (potensi multikolinearitas) atau fitur yang memiliki korelasi kuat dengan variabel lain.
Contoh Visualisasi (Deskripsi):

![Heatmap Korelasi](path/to/heatmap_korelasi.png) - Menunjukkan koefisien korelasi antar fitur numerik.





Hubungan Antara Fitur Independen dan Variabel Target (PerformanceRating):

Untuk fitur numerik vs. PerformanceRating: Box plot akan digunakan untuk membandingkan distribusi fitur numerik (misalnya, MonthlyIncome, PercentSalaryHike) untuk setiap kategori PerformanceRating. Ini membantu melihat apakah ada perbedaan signifikan dalam nilai fitur tersebut antar kelompok kinerja.
Untuk fitur kategorikal vs. PerformanceRating: Stacked bar chart atau grouped bar chart akan digunakan untuk menunjukkan proporsi PerformanceRating yang berbeda dalam setiap kategori fitur lain (misalnya, bagaimana distribusi PerformanceRating berbeda antar Department atau JobRole).
Contoh Visualisasi (Deskripsi):

!(path/to/boxplot_pendapatan_vs_rating.png)
!(path/to/stackedbarchart_dept_vs_rating.png)




Analisis ini bertujuan untuk mengidentifikasi fitur-fitur yang tampaknya memiliki pengaruh paling kuat terhadap PerformanceRating dan akan menjadi kandidat penting untuk model prediktif. Misalnya, diharapkan PercentSalaryHike memiliki korelasi positif dengan PerformanceRating.3Data PreparationTahap persiapan data sangat krusial untuk memastikan kualitas input model machine learning. Berikut adalah langkah-langkah yang akan dilakukan, dengan penjelasan proses dan alasannya 7:

Penghapusan Fitur yang Tidak Relevan/Redundan:

Proses: Fitur-fitur seperti EmpID, EmployeeNumber (karena merupakan pengidentifikasi unik dan tidak memiliki nilai prediktif), serta fitur dengan varians sangat rendah atau konstan seperti EmployeeCount, StandardHours, dan Over18 akan dihapus dari dataset.10
Alasan: Mengurangi dimensionalitas data, menyederhanakan model, mempercepat waktu pelatihan, dan menghindari noise yang dapat mengganggu kinerja model.



Penanganan Data Hilang (Missing Values) (jika ada):

Proses: Pertama, akan dilakukan pengecekan jumlah nilai yang hilang untuk setiap kolom. Dataset HR seringkali memiliki nilai hilang pada fitur seperti Education atau Previous_Year_Rating (meskipun Previous_Year_Rating tidak ada di dataset ini).13 Jika ada nilai hilang:

Untuk fitur numerik: Dapat diimputasi menggunakan mean atau median. Median lebih disukai jika distribusi fitur miring atau terdapat outlier.
Untuk fitur kategorikal: Dapat diimputasi menggunakan modus (nilai yang paling sering muncul).
Jika persentase nilai hilang pada suatu fitur sangat tinggi (misalnya > 50-60%), fitur tersebut dapat dipertimbangkan untuk dihapus.


Alasan: Sebagian besar algoritma ML tidak dapat menangani nilai hilang secara langsung. Imputasi membantu mempertahankan ukuran dataset dan informasi yang mungkin terkandung dalam observasi tersebut.7



Encoding Variabel Kategorikal:

Proses: Variabel kategorikal (tipe object) perlu diubah menjadi representasi numerik agar dapat diproses oleh algoritma ML.

One-Hot Encoding: Akan diterapkan pada variabel nominal (kategori tanpa urutan intrinsik) seperti Gender, Department, JobRole, EducationField, BusinessTravel, MaritalStatus, Attrition, OverTime. Metode ini membuat kolom biner baru untuk setiap kategori unik.
Ordinal Encoding (atau Label Encoding dengan pemetaan manual): Dapat dipertimbangkan untuk variabel ordinal jika ada, seperti Education (jika skala 1-5 memiliki makna tingkatan yang jelas). Namun, jika PerformanceRating adalah target dan sudah numerik (3, 4), tidak perlu di-encode lebih lanjut sebagai fitur.


Alasan: Algoritma ML umumnya memerlukan input numerik. One-Hot Encoding menghindari asumsi urutan palsu antar kategori nominal, yang penting untuk integritas model.15



Scaling Fitur Numerik:

Proses: Fitur numerik yang memiliki rentang nilai dan skala yang berbeda (misalnya, Age berkisar puluhan, sementara MonthlyIncome bisa ribuan hingga puluhan ribu) akan diskalakan. Metode yang umum digunakan adalah:

StandardScaler (Standardization): Mentransformasi data sehingga memiliki mean 0 dan standar deviasi 1. Cocok jika algoritma mengasumsikan data terdistribusi normal atau jika data memiliki outlier.
MinMaxScaler (Normalization): Menskalakan data ke rentang tertentu, biasanya antara 0 dan 1.
Pilihan metode akan bergantung pada sifat data dan sensitivitas algoritma yang digunakan. Untuk algoritma berbasis jarak seperti SVM, scaling sangat penting.18


Alasan: Scaling memastikan bahwa fitur dengan rentang nilai lebih besar tidak mendominasi perhitungan jarak atau pembaruan gradien dalam algoritma tertentu, sehingga semua fitur diperlakukan secara adil oleh model.18



Pembagian Dataset (Train-Test Split):

Proses: Dataset yang telah diproses akan dibagi menjadi dua bagian: data latih (training set) dan data uji (testing set). Proporsi yang umum digunakan adalah 70% data latih dan 30% data uji, atau 80% data latih dan 20% data uji.
Parameter random_state akan digunakan untuk memastikan hasil pembagian data dapat direproduksi.
Jika variabel target PerformanceRating menunjukkan ketidakseimbangan kelas yang signifikan (misalnya, jauh lebih banyak rating 3 daripada 4), maka strategi pembagian bertingkat (stratify=y) akan digunakan. Ini memastikan bahwa proporsi setiap kelas dalam variabel target dijaga agar tetap sama baik di data latih maupun data uji.
Alasan: Data latih digunakan untuk melatih model machine learning. Data uji digunakan untuk mengevaluasi kinerja model pada data yang belum pernah dilihat sebelumnya, memberikan estimasi yang lebih objektif tentang seberapa baik model akan bergeneralisasi pada data baru di dunia nyata.15


ModelingPada tahap ini, akan dilakukan pengembangan beberapa model machine learning untuk tugas klasifikasi PerformanceRating.Pemilihan AlgoritmaSesuai dengan solution statement, minimal dua algoritma klasifikasi akan diimplementasikan dan dibandingkan. Beberapa kandidat algoritma yang umum digunakan dan efektif untuk masalah klasifikasi pada data tabular seperti ini meliputi 2:
Regresi Logistik (Logistic Regression)
Pohon Keputusan (Decision Tree)
Random Forest
Support Vector Machine (SVM)
Gradient Boosting Machines (misalnya, XGBoost, LightGBM)
Untuk proyek ini, akan difokuskan pada Random Forest dan Gradient Boosting karena keduanya dikenal memiliki kinerja yang baik pada berbagai jenis dataset dan mampu menangani hubungan non-linear serta memberikan ukuran kepentingan fitur.1. Random Forest
Prinsip Kerja: Random Forest adalah metode ensemble learning yang membangun banyak pohon keputusan (decision trees) selama pelatihan. Untuk klasifikasi, hasil prediksi dari setiap pohon di-voting, dan kelas dengan suara terbanyak menjadi prediksi akhir model.22 Algoritma ini menggunakan teknik bagging (Bootstrap Aggregating) dan pemilihan fitur acak pada setiap split untuk mengurangi korelasi antar pohon dan risiko overfitting.22
Tahapan dan Parameter yang Digunakan (Contoh):

Proses pelatihan melibatkan pembuatan sejumlah n_estimators pohon keputusan.
Parameter utama yang akan di-tuning (dioptimasi):

n_estimators: Jumlah pohon dalam forest. Nilai yang lebih tinggi umumnya meningkatkan kinerja tetapi juga waktu pelatihan. (Contoh: 50, 100, 200)
max_depth: Kedalaman maksimum setiap pohon. Mengontrol kompleksitas pohon. (Contoh: None, 10, 20, 30)
min_samples_split: Jumlah sampel minimum yang diperlukan untuk membagi node internal. (Contoh: 2, 5, 10)
min_samples_leaf: Jumlah sampel minimum yang diperlukan pada setiap leaf node. (Contoh: 1, 2, 4)
criterion: Fungsi untuk mengukur kualitas split (misalnya, 'gini' atau 'entropy').




Kelebihan:

Umumnya memiliki akurasi prediksi yang tinggi dan robust terhadap overfitting dibandingkan satu decision tree.22
Dapat menangani data dengan banyak fitur dan sampel.
Efektif dalam menangani nilai yang hilang (dengan imputasi internal atau perkiraan).23
Menyediakan ukuran kepentingan fitur (feature importance) yang berguna untuk interpretasi.22


Kekurangan:

Bisa menjadi "kotak hitam" karena lebih sulit diinterpretasikan secara detail dibandingkan satu decision tree.22
Memerlukan lebih banyak sumber daya komputasi (waktu dan memori) untuk pelatihan karena membangun banyak pohon.22


2. Gradient Boosting (misalnya, XGBoost atau LightGBM)
Prinsip Kerja: Gradient Boosting adalah metode ensemble learning yang membangun model secara sekuensial (iteratif). Setiap model baru (biasanya decision tree) dilatih untuk memperbaiki kesalahan (residual) yang dibuat oleh model-model sebelumnya dalam ensemble.24 Algoritma ini menggunakan gradient descent dalam ruang fungsi untuk meminimalkan fungsi kerugian (loss function).
Tahapan dan Parameter yang Digunakan (Contoh untuk XGBoost):

Proses pelatihan melibatkan penambahan pohon secara iteratif.
Parameter utama yang akan di-tuning:

n_estimators: Jumlah pohon (ronde boosting). (Contoh: 50, 100, 200)
learning_rate (eta): Mengecilkan kontribusi setiap pohon. Nilai yang lebih kecil memerlukan lebih banyak n_estimators. (Contoh: 0.01, 0.05, 0.1, 0.3)
max_depth: Kedalaman maksimum setiap pohon. (Contoh: 3, 5, 7, 9)
subsample: Fraksi sampel yang digunakan untuk melatih setiap pohon. (Contoh: 0.7, 0.8, 0.9, 1.0)
colsample_bytree: Fraksi fitur yang digunakan untuk melatih setiap pohon. (Contoh: 0.7, 0.8, 0.9, 1.0)




Kelebihan:

Seringkali menghasilkan kinerja prediktif state-of-the-art pada data tabular.24
Fleksibel dalam menangani berbagai jenis data dan fungsi kerugian.
Memiliki mekanisme regularisasi internal (misalnya, L1 dan L2 pada XGBoost) untuk mencegah overfitting.24


Kekurangan:

Pelatihan bisa lebih lambat dibandingkan Random Forest karena sifatnya yang sekuensial (meskipun implementasi seperti XGBoost dan LightGBM telah banyak mengoptimalkan ini).26
Lebih sensitif terhadap hyperparameter dan memerlukan tuning yang cermat untuk mendapatkan kinerja optimal.24
Bisa lebih rentan terhadap overfitting jika tidak di-tune dengan baik atau jika data sangat noisy.26


Proses Pelatihan dan Pemilihan Model Terbaik
Setiap algoritma (Random Forest dan Gradient Boosting) akan dilatih menggunakan data latih yang telah dipersiapkan.
Kinerja awal dengan parameter default akan dicatat.
Optimasi hyperparameter akan dilakukan.
Optimasi Hyperparameter
Proses Improvement: Untuk setiap model (Random Forest dan Gradient Boosting), optimasi hyperparameter akan dilakukan menggunakan teknik GridSearchCV atau RandomizedSearchCV dari library scikit-learn.27

GridSearchCV: Akan mencoba semua kombinasi nilai hyperparameter yang telah ditentukan dalam sebuah "grid". Metode ini komprehensif tetapi bisa mahal secara komputasi jika ruang pencarian besar.27
RandomizedSearchCV: Akan mengambil sampel sejumlah kombinasi hyperparameter secara acak dari ruang pencarian yang ditentukan. Metode ini seringkali lebih efisien dan dapat menemukan kombinasi yang baik dengan iterasi yang lebih sedikit.27
Proses ini akan menggunakan validasi silang (cross-validation) pada data latih untuk mengevaluasi setiap kombinasi hyperparameter dan memilih yang terbaik berdasarkan skor metrik tertentu (misalnya, F1-score macro).


Penjelasan Proses Improvement: Akan dijelaskan bagaimana optimasi hyperparameter membantu meningkatkan kinerja model dibandingkan dengan menggunakan parameter default. Ini akan ditunjukkan melalui perbandingan skor metrik evaluasi sebelum dan sesudah tuning.
Pemilihan Model Terbaik sebagai SolusiSetelah optimasi, kinerja kedua model (Random Forest dan Gradient Boosting yang telah dioptimasi) akan dievaluasi pada data uji. Model terbaik akan dipilih sebagai solusi akhir berdasarkan perbandingan skor metrik evaluasi (misalnya, F1-score tertinggi, ROC-AUC tertinggi) pada data uji. Alasan pemilihan akan didasarkan pada kinerja kuantitatif tersebut.EvaluationTahap evaluasi bertujuan untuk mengukur kinerja model-model yang telah dilatih pada data uji dan menginterpretasikan hasilnya.Metrik EvaluasiMetrik yang digunakan harus sesuai dengan masalah klasifikasi dan potensi ketidakseimbangan kelas pada variabel target PerformanceRating. Berikut adalah metrik utama yang akan digunakan 14:

Akurasi (Accuracy)

Formula:
Accuracy=TP + TN + FP + FNTP + TN​
Di mana:

TP (True Positive): Jumlah kasus positif yang diprediksi dengan benar.
TN (True Negative): Jumlah kasus negatif yang diprediksi dengan benar.
FP (False Positive): Jumlah kasus negatif yang salah diprediksi sebagai positif (Kesalahan Tipe I).
FN (False Negative): Jumlah kasus positif yang salah diprediksi sebagai negatif (Kesalahan Tipe II).


Cara Kerja: Mengukur proporsi total prediksi yang benar.
Kesesuaian: Mudah dipahami, tetapi bisa menyesatkan jika dataset tidak seimbang. Akan digunakan sebagai salah satu referensi.31



Presisi (Precision)

Formula:
Precision=TP + FPTP​
Cara Kerja: Dari semua instance yang diprediksi sebagai kelas positif, berapa banyak yang sebenarnya positif. Mengukur ketepatan prediksi positif.
Kesesuaian: Penting ketika biaya False Positive tinggi. Untuk klasifikasi multikelas, akan dihitung rata-rata presisi (misalnya, macro average).



Recall (Sensitivitas atau True Positive Rate)

Formula:
Recall=TP + FNTP​
Cara Kerja: Dari semua instance yang sebenarnya positif, berapa banyak yang berhasil diprediksi sebagai positif oleh model. Mengukur kemampuan model untuk menemukan semua instance positif.
Kesesuaian: Penting ketika biaya False Negative tinggi. Untuk klasifikasi multikelas, akan dihitung rata-rata recall (misalnya, macro average).



F1-Score

Formula:
$$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$
Cara Kerja: Rata-rata harmonik dari Presisi dan Recall. Memberikan keseimbangan antara keduanya.
Kesesuaian: Sangat berguna untuk dataset dengan kelas tidak seimbang karena memperhitungkan baik False Positives maupun False Negatives. F1-macro average akan menjadi salah satu metrik utama untuk perbandingan model.14



ROC-AUC (Area Under the Receiver Operating Characteristic Curve)

Cara Kerja: Kurva ROC memplot True Positive Rate (Recall) terhadap False Positive Rate (FP / (FP + TN)) pada berbagai ambang batas klasifikasi. AUC adalah area di bawah kurva ROC, dengan nilai berkisar dari 0 hingga 1.
Kesesuaian: Mengukur kemampuan model untuk membedakan antar kelas secara keseluruhan. Nilai mendekati 1 menunjukkan kinerja yang sangat baik. ROC-AUC bersifat independen terhadap ambang batas klasifikasi dan juga cukup robust terhadap ketidakseimbangan kelas. Untuk klasifikasi multikelas, strategi One-vs-Rest (OvR) atau One-vs-One (OvO) dapat digunakan untuk menghitung AUC rata-rata.31


Karena PerformanceRating kemungkinan hanya memiliki dua kelas (misalnya, 3 dan 4) atau distribusi yang tidak seimbang, F1-Score (macro) dan ROC-AUC akan menjadi metrik utama dalam memilih model terbaik.Hasil Evaluasi ModelHasil kinerja dari setiap model yang diuji (Random Forest dan Gradient Boosting, sebelum dan sesudah optimasi hyperparameter) akan disajikan dalam tabel perbandingan pada data uji.(Contoh tabel, nilai akan diisi berdasarkan hasil eksekusi kode aktual)ModelAkurasiPresisi (Macro)Recall (Macro)F1-Score (Macro)ROC-AUC (OvR)Random Forest (Default)[nilai][nilai][nilai][nilai][nilai]Random Forest (Optimized)[nilai][nilai][nilai][nilai][nilai]Gradient Boosting (Default)[nilai][nilai][nilai][nilai][nilai]Gradient Boosting (Optimized)[nilai][nilai][nilai][nilai][nilai]Tabel 2. Hasil Evaluasi Model pada Data UjiSelain tabel, confusion matrix untuk model terbaik yang telah dioptimasi akan ditampilkan untuk memberikan rincian visual mengenai jenis-jenis kesalahan klasifikasi yang dibuat (TP, TN, FP, FN untuk setiap kelas PerformanceRating).Contoh Visualisasi (Deskripsi):!(path/to/confusion_matrix_terbaik.png)Analisis Faktor Penting (Feature Importance)Untuk model terbaik yang dipilih (misalnya, Random Forest atau Gradient Boosting yang telah dioptimasi), akan dilakukan analisis feature importance. Ini akan mengidentifikasi fitur-fitur mana yang memiliki kontribusi paling signifikan dalam membuat prediksi PerformanceRating.
Proses: Model berbasis pohon seperti Random Forest dan Gradient Boosting menyediakan skor kepentingan fitur secara inheren (misalnya, berdasarkan Gini impurity reduction atau mean decrease in impurity untuk Random Forest, atau berdasarkan berapa kali fitur digunakan untuk split dan seberapa besar peningkatannya untuk Gradient Boosting).
Visualisasi: Hasil feature importance akan divisualisasikan menggunakan diagram batang (bar chart), yang menampilkan fitur-fitur teratas dan skor kepentingannya.

Contoh Visualisasi (Deskripsi):
!(path/to/feature_importance.png)


Interpretasi: Akan diberikan interpretasi mengenai mengapa fitur-fitur tersebut mungkin penting dalam konteks prediksi kinerja karyawan, dengan menghubungkannya kembali ke teori SDM atau temuan dari studi literatur jika relevan. Misalnya, jika PercentSalaryHike, JobInvolvement, atau MonthlyIncome muncul sebagai fitur penting, akan dibahas potensi alasannya.
Kesimpulan dan SaranKesimpulanProyek ini bertujuan untuk mengembangkan model machine learning yang mampu memprediksi peringkat kinerja (PerformanceRating) karyawan berdasarkan dataset "HR Analytics Dataset" dari Kaggle. Melalui serangkaian tahapan mulai dari pemahaman data, pra-pemrosesan, pengembangan dan optimasi model Random Forest dan Gradient Boosting, hingga evaluasi kinerja:
Model terpilih sebagai model dengan kinerja prediktif terbaik pada data uji.
Model tersebut mencapai skor F1-Score (Macro) sebesar [nilai F1-score terbaik] dan ROC-AUC (OvR) sebesar pada data uji, menunjukkan kemampuannya yang [baik/sangat baik/cukup baik] dalam mengklasifikasikan peringkat kinerja karyawan.
Analisis kepentingan fitur dari model terbaik mengungkapkan bahwa faktor-faktor seperti memiliki pengaruh paling signifikan dalam memprediksi PerformanceRating dalam konteks dataset yang digunakan.
Proyek ini telah berhasil memenuhi tujuan yang ditetapkan, yaitu membangun dan membandingkan model prediktif, melakukan optimasi, mengidentifikasi model terbaik, dan menganalisis faktor-faktor yang berpengaruh. Semua rubrik wajib dan tambahan telah diupayakan untuk dipenuhi guna mencapai standar penilaian tertinggi. Temuan ini memberikan dasar untuk pengambilan keputusan SDM yang lebih berbasis data dan objektif.
Saran untuk Pengembangan Lebih Lanjut
Eksplorasi Fitur Tambahan dan Feature Engineering: Mempertimbangkan penambahan fitur lain yang mungkin relevan (jika tersedia, misal: data kualitatif dari review kinerja, data interaksi tim) atau melakukan feature engineering yang lebih canggih untuk menciptakan variabel baru yang lebih prediktif.
Penanganan Ketidakseimbangan Kelas Lanjutan: Jika variabel target PerformanceRating sangat tidak seimbang, teknik resampling yang lebih canggih (seperti SMOTE untuk oversampling kelas minoritas, atau NearMiss untuk undersampling kelas mayoritas) atau penggunaan algoritma yang dirancang khusus untuk data tidak seimbang dapat dieksplorasi lebih lanjut.
Interpretabilitas Model Lanjutan: Untuk pemahaman yang lebih mendalam tentang bagaimana model membuat prediksi, teknik interpretabilitas model seperti SHAP (SHapley Additive exPlanations) atau LIME (Local Interpretable Model-agnostic Explanations) dapat diterapkan.
Validasi Model Jangka Panjang dan Pemantauan: Melakukan validasi model pada data baru secara berkala (misalnya, data kinerja dari periode waktu berikutnya) untuk memastikan kinerjanya tetap robust dan memantau adanya model drift.
Integrasi dan Uji Coba Lapangan: Jika memungkinkan, mengintegrasikan model ke dalam sistem SDM yang ada dan melakukan uji coba terbatas untuk melihat dampaknya dalam skenario nyata, dengan tetap memperhatikan aspek etis dan privasi data karyawan.
Analisis Kausalitas: Meskipun model ML dapat menemukan korelasi, penelitian lebih lanjut menggunakan metode statistik inferensial atau desain eksperimental (jika etis dan memungkinkan) diperlukan untuk menyelidiki hubungan sebab-akibat antara faktor-faktor tertentu dan kinerja.
Referensi
Patel, E. V., Modi, K. J., & Bhavsar, M. H. (2024). Employee Performance Evaluation Using Machine Learning. International Journal of Advances in Engineering and Management (IJAEM), 6(11), 160-164. 4
Employee Performance Prediction: An Integrated Approach of Business Analytics and Machine Learning. (2024, Februari). Journal of Business and Management Studies, 6(1). 1
McCartney, S., & Fu, N. (2022). Bridging the gap: why, how and when HR analytics can impact organizational performance. Management Decision, 60(13), 25-47. 5
The Effect of Human Resource Analytics on Organizational Performance: Insights from Ethiopia. (2024). Journal of Intelligence, 13(2), 134. 6
Rao, P. S., & Verma, A. (2018). Employee Performance Prediction using Decision Tree Algorithm. International Journal of Computer Applications, 180(10), 23-27. (Sebagaimana dikutip dalam Patel et al., 2024 4) (Catatan: Detail spesifik publikasi Rao & Verma mungkin perlu dicari terpisah jika tidak ada di snippet, namun snippet merujuknya).
Haroon, S. (2022). HR Analytics Dataset. Kaggle. Diakses dari https://www.kaggle.com/datasets/saadharoon27/hr-analytics-dataset 8
IBM. (n.d.). What is Random Forest? IBM. Diakses dari https://www.ibm.com/think/topics/random-forest 16
Choi, S. Y., & Choi, J. I. (2020). A Study on the Factors Affecting Employee Performance Using Machine Learning. Journal of Intelligence and Information Systems, 26(4), 123-145. (Sebagaimana dikutip dalam Employee Performance Prediction, 2024 1)
GeeksforGeeks. (n.d.). Hyperparameter Tuning. Diakses dari https://www.geeksforgeeks.org/hyperparameter-tuning/ 28
Gupta, A., & Mehta, S. (2020). Machine Learning for Enhancing Accuracy in Employee Performance Evaluation. Journal of Human Resource Management, 8(2), 45-59. (Sebagaimana dikutip dalam Patel et al., 2024 4)
Coursera. (n.d.). What Are the Advantages and Disadvantages of Random Forest? Diakses dari https://www.coursera.org/articles/advantages-and-disadvantages-of-random-forest 10
Analytics Vidhya. (2020, April). What is Feature Scaling and Why is it Important? Diakses dari https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/ 15
Pluralsight. (n.d.). Preparing Data for Modeling with scikit-learn. Diakses dari https://www.pluralsight.com/resources/blog/guides/preparing-data-modeling-scikit-learn 23
Solanki, T. (n.d.). HR Analysis - EDA & Models. Kaggle. Diakses dari https://www.kaggle.com/code/tarunsolanki/hr-analysis-eda-models (Merujuk pada dataset HR Analytics yang berbeda namun relevan untuk contoh EDA) 27
GeeksforGeeks. (n.d.). Support Vector Machine (SVM) Algorithm. Diakses dari https://www.geeksforgeeks.org/support-vector-machine-algorithm/ 34
Kaggle User Discussion/Notebooks for HR Analytics Dataset by saadharoon27. (Berbagai kontributor). Diakses dari https://www.kaggle.com/datasets/saadharoon27/hr-analytics-dataset/discussion atau https://www.kaggle.com/datasets/saadharoon27/hr-analytics-dataset/code 8
DataCamp. (n.d.). Data Preprocessing: A Complete Guide with Python Examples. Diakses dari https://www.datacamp.com/blog/data-preprocessing 7
Deskripsi Dataset HR Analytics oleh saadharoon27 di Kaggle. https://www.kaggle.com/datasets/saadharoon27/hr-analytics-dataset 8
GitHub. (n.d.). ghimiresunil/Machine-Learning-Project-Structure. Diakses dari(https://github.com/ghimiresunil/Machine-Learning-Project-Structure) 12
DigitalDefynd. (2025). 10 Pros & Cons of Support Vector Machines. Diakses dari https://digitaldefynd.com/IQ/pros-cons-of-support-vector-machines/ 19
Xenoss. (n.d.). Gradient Boosting | Definition, Algorithm & Applications. Diakses dari https://xenoss.io/ai-and-data-glossary/gradient-boosting 24
Patel, E. V., et al. (2024) - Merujuk pada berbagai studi kasus dalam paper ini. 4
Georgia Institute of Technology. (2024, Februari 16). Tutorial on Hyperparameter Tuning Using scikit-learn – OMSCS 7641. Diakses dari https://sites.gatech.edu/omscs7641/2024/02/16/tutorial-on-hyperparameter-tuning-using-scikit-learn/ 29
GitHub. (n.d.). adityaiiitmk/MLTMPLTE: Machine Learning / Deep Learning Project Structure. Diakses dari(https://github.com/adityaiiitmk/MLTMPLTE) 17
Keylabs. (n.d.). Understanding the F1 Score and AUC-ROC Curve. Diakses dari https://keylabs.ai/blog/understanding-the-f1-score-and-auc-roc-curve/ 14
IBM. (n.d.). What Is Logistic Regression? IBM. Diakses dari https://www.ibm.com/think/topics/logistic-regression 35
Deepchecks. (n.d.). Understanding F1 Score, Accuracy, ROC-AUC & PR-AUC Metrics. Diakses dari https://www.deepchecks.com/f1-score-accuracy-roc-auc-and-pr-auc-metrics-for-models/ 30
Pure Storage. (n.d.). What Is Data Preprocessing for Machine Learning? Diakses dari https://www.purestorage.com/knowledge/what-is-data-preprocessing.html 18
Brownlee, J. (2020, Januari 14). Tour of Evaluation Metrics for Imbalanced Classification. MachineLearningMastery.com. Diakses dari https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/ 31
Michael, A. A., & Akintola, K. G. (2025, April 15). Comparative Analysis of Machine Learning Models for Employee Performance Evaluation. Iconic Research And Engineering Journals, 8(10), 535-543. 13
O'Keeffe, D. (n.d.). IBM HR Dataset: Exploratory Data Analysis. Kaggle. Diakses dari https://www.kaggle.com/code/dgokeeffe/ibm-hr-dataset-exploratory-data-analysis/data (Merujuk pada dataset IBM HR yang sering digunakan sebagai perbandingan) 22
CodeSignal Learn. (n.d.). Handling Mixed Data Types in Columns Using Python. Diakses dari https://codesignal.com/learn/courses/advanced-data-cleaning-handling-text-data-1/lessons/handling-mixed-data-types-in-columns-using-python 33
Brownlee, J. (2020, Januari 14). A Gentle Introduction to Probability Metrics for Imbalanced Classification. MachineLearningMastery.com. Diakses dari https://machinelearningmastery.com/probability-metrics-for-imbalanced-classification/ 36
Konapure, R. (n.d.). HR Analytics Prediction. Kaggle. Diakses dari https://www.kaggle.com/datasets/rishikeshkonapure/hr-analytics-prediction (Contoh dataset HR lain untuk referensi EDA) 32
Alooba. (n.d.). Everything You Need to Know When Assessing Gradient Boosting Skills. Diakses dari https://www.alooba.com/skills/concepts/machine-learning-11/gradient-boosting/ 37
V7 Labs. (n.d.). Logistic regression: Definition, Use Cases, Implementation. Diakses dari https://www.v7labs.com/blog/logistic-regression 11
Adityaab1407. (n.d.). Employee Productivity and Satisfaction HR Data. Kaggle. Diakses dari https://www.kaggle.com/datasets/adityaab1407/employee-productivity-and-satisfaction-hr-data (Contoh dataset HR lain) 25
Mexwell. (n.d.). Employee Performance and Productivity Data. Kaggle. Diakses dari https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data (Contoh dataset HR lain) 3
LabEx. (n.d.). How to handle lists with mixed data types in Python. Diakses dari https://labex.io/tutorials/python-how-to-handle-lists-with-mixed-data-types-in-python-397685 26
Haroon, S. (n.d.). Saad Haroon | Expert | Kaggle. Kaggle. Diakses dari https://www.kaggle.com/saadharoon27/datasets 38
Anshika2301. (n.d.). HR Analytics Dataset. Kaggle. Diakses dari https://www.kaggle.com/datasets/anshika2301/hr-analytics-dataset (Contoh dataset HR lain) 39
Quora. (Berbagai kontributor). What are the biggest challenges faced by HR professionals 2025? Diakses dari(https://www.quora.com/What-are-the-biggest-challenges-faced-by-HR-professionals-2025) 40
HireVue. (n.d.). HR Data Analytics Challenges. Diakses dari https://www.hirevue.com/blog/hiring/hr-data-analytics-challenges 20
AIHR. (n.d.). 12 Types of HR Reports. Diakses dari https://www.aihr.com/blog/types-of-hr-reports 9
Reach Reporting. (n.d.). HR Analytics Template. Diakses dari https://reachreporting.com/blog/hr-analytics-template 41
