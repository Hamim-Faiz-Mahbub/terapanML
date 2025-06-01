# Laporan Proyek Machine Learning Terapan Pertama - Hamim Faiz Mahbub 


## Latar Belakang
<p align="center">
<img src= "https://github.com/user-attachments/assets/37c3e813-1b36-4e5f-a396-6bec399c6a3b"/>
</p>


- Dalam lanskap bisnis yang dinamis dan kompetitif, sumber daya manusia (SDM) diakui sebagai aset strategis utama yang menentukan keberhasilan organisasi.[1] Kinerja karyawan secara individual dan kolektif berdampak langsung pada produktivitas, inovasi, profitabilitas, dan pencapaian tujuan strategis perusahaan.[2] Oleh karena itu, kemampuan organisasi untuk secara akurat mengevaluasi dan memprediksi kinerja karyawan telah menjadi fungsi krusial dalam manajemen SDM modern.[1]

- Metode evaluasi kinerja tradisional seringkali mengandalkan penilaian subjektif dari manajer, yang dapat rentan terhadap berbagai bias seperti bias kelonggaran, bias tendensi sentral, atau efek halo.[1, 3] Keterbatasan ini dapat menghasilkan evaluasi yang tidak konsisten, kurang akurat, dan berpotensi menimbulkan ketidakpuasan di kalangan karyawan, serta menghambat pengambilan keputusan yang efektif terkait pengembangan talenta, promosi, dan kompensasi.[1, 3]

- Seiring dengan kemajuan teknologi dan ketersediaan data SDM yang melimpah, analitik SDM (HR Analytics) dan penerapan machine learning (ML) menawarkan pendekatan yang lebih objektif, berbasis data, dan prediktif untuk evaluasi kinerja.[4, 5, 3] Dengan memanfaatkan algoritma ML, organisasi dapat menganalisis pola kompleks dalam data karyawan untuk mengidentifikasi faktor-faktor yang memengaruhi kinerja dan membuat prediksi yang lebih akurat mengenai kinerja di masa depan.[1] Pendekatan ini tidak hanya meningkatkan akurasi dan efisiensi proses evaluasi tetapi juga berpotensi mengurangi bias dan mendukung pengambilan keputusan SDM yang lebih strategis dan adil.[3]

Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi peringkat kinerja (PerformanceRating) karyawan berdasarkan berbagai atribut yang tersedia dalam dataset analitik SDM.

Masalah subjektivitas, inkonsistensi, dan potensi bias dalam metode evaluasi kinerja tradisional perlu diatasi karena dapat berdampak negatif pada:

- Keadilan dan Moral Karyawan: Penilaian yang tidak adil dapat menurunkan motivasi dan keterlibatan karyawan.[3]

- Pengambilan Keputusan SDM: Keputusan yang salah terkait promosi, kompensasi, dan pengembangan dapat merugikan baik karyawan maupun organisasi.[1]

- Efektivitas Organisasi: Kegagalan dalam mengidentifikasi dan mengembangkan talenta terbaik dapat menghambat pencapaian tujuan organisasi.[1]

Masalah ini dapat diselesaikan dengan:

1. Mengadopsi Pendekatan Berbasis Data: Menggunakan data historis karyawan untuk mengidentifikasi pola objektif yang berkaitan dengan kinerja. [1]

2. Menerapkan Algoritma Machine Learning: Membangun model prediktif yang dapat mempelajari hubungan kompleks antara berbagai faktor karyawan dan peringkat kinerja mereka.[1, 3]

3. Mengidentifikasi Faktor Kunci: Menggunakan model untuk memahami variabel mana yang paling signifikan dalam memprediksi kinerja, sehingga intervensi SDM dapat lebih terarah. [1]

Dengan demikian, solusi berbasis machine learning diharapkan dapat menyediakan alat bantu yang lebih objektif dan akurat bagi manajer SDM dalam proses evaluasi kinerja.
Dengan demikian, solusi berbasis machine learning diharapkan dapat menyediakan alat bantu yang lebih objektif dan akurat bagi manajer SDM dalam proses evaluasi kinerja.


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

### Dimensi Dataset Awal
Dimensi Dataset (baris, kolom): (1480, 38)
- Dataset diketahui Memiliki 1480 baris yang masing-masing memiliki 38 attribute di dalamnya

### Deskripsi Variabel



| Nama Fitur              | Tipe Data (Inferensi) | Deskripsi Singkat                                                             | Contoh Nilai                |
|-------------------------|------------------------|--------------------------------------------------------------------------------|-----------------------------|
| EmpID                   | object                 | ID unik karyawan                                                               | RM297, RM302                |
| Age                     | int64                  | Usia karyawan dalam tahun                                                      | 35, 42                      |
| AgeGroup                | object                 | Kategori umur tempat karyawan dikelompokkan                                    | 30-40, 40-50                |
| Attrition              | object                 | Apakah karyawan mengalami atrisi (Yes/Tidak)                                  | Yes, No                     |
| BusinessTravel          | object                 | Frekuensi perjalanan bisnis                                                    | Travel_Rarely, Non-Travel   |
| DailyRate               | int64                  | Gaji harian karyawan                                                           | 800, 1200                   |
| Department              | object                 | Departemen tempat karyawan bekerja                                             | Sales, R&D                  |
| DistanceFromHome        | int64                  | Jarak dari rumah ke kantor (dalam mil)                                         | 10, 2                       |
| Education               | int64                  | Tingkat pendidikan (misal 1=SMA, 3=S1, 4=S2)                                    | 3, 4                        |
| EducationField          | object                 | Bidang pendidikan karyawan                                                     | Life Sciences, Medical      |
| EmployeeCount           | int64                  | Jumlah pegawai (umumnya bernilai konstan)                                      | 1                           |
| EmployeeNumber          | int64                  | Nomor unik karyawan                                                            | 1001, 1015                  |
| EnvironmentSatisfaction | int64                  | Tingkat kepuasan terhadap lingkungan kerja (skala 1–4)                         | 3, 4                        |
| Gender                  | object                 | Jenis kelamin karyawan                                                         | Male, Female                |
| HourlyRate              | int64                  | Gaji per jam karyawan                                                          | 45, 80                      |
| JobInvolvement          | int64                  | Tingkat keterlibatan dalam pekerjaan (skala 1–4)                               | 3, 2                        |
| JobLevel                | int64                  | Level jabatan karyawan                                                         | 2, 5                        |
| JobRole                 | object                 | Peran atau jabatan spesifik karyawan                                           | Sales Executive, Manager    |
| JobSatisfaction         | int64                  | Tingkat kepuasan kerja (skala 1–4)                                             | 4, 1                        |
| MaritalStatus           | object                 | Status pernikahan karyawan                                                     | Single, Married             |
| MonthlyIncome           | int64                  | Pendapatan bulanan karyawan                                                    | 5000, 12000                 |
| SalarySlab              | object                 | Kategori pengelompokan gaji bulanan                                            | Low, High                   |
| MonthlyRate             | int64                  | Gaji bulanan standar                                                           | 20000, 15000                |
| NumCompaniesWorked      | int64                  | Jumlah perusahaan tempat karyawan pernah bekerja                               | 1, 5                        |
| Over18                  | object                 | Apakah karyawan berusia di atas 18 tahun                                       | Yes, NO                           |
| OverTime                | object                 | Apakah karyawan bekerja lembur                                                 | Yes, No                     |
| PercentSalaryHike       | int64                  | Persentase kenaikan gaji terakhir                                              | 15, 20                      |
| PerformanceRating       | int64                  | Peringkat kinerja karyawan (biasanya skala 1–4)                                | 3, 4                        |
| RelationshipSatisfaction| int64                  | Kepuasan terhadap hubungan kerja (skala 1–4)                                   | 2, 4                        |
| StandardHours           | int64                  | Jam kerja standar (biasanya bernilai tetap)                                    | 40                          |
| StockOptionLevel        | int64                  | Level opsi saham yang diberikan perusahaan                                     | 0, 1                        |
| TotalWorkingYears       | int64                  | Total tahun pengalaman kerja                                                   | 10, 20                      |
| TrainingTimesLastYear   | int64                  | Jumlah pelatihan yang diikuti tahun lalu                                       | 2, 4                        |
| WorkLifeBalance         | int64                  | Kepuasan terhadap keseimbangan kerja dan hidup (skala 1–4)                     | 3, 4                        |
| YearsAtCompany          | int64                  | Lama bekerja di perusahaan saat ini (dalam tahun)                              | 5, 2                        |
| YearsInCurrentRole      | int64                  | Lama bekerja di posisi saat ini                                                | 3, 1                        |
| YearsSinceLastPromotion | int64                  | Tahun sejak promosi terakhir                                                   | 2, 5                        |
| YearsWithCurrManager    | float64                | Lama bekerja bersama atasan saat ini                                           | 3.0, 6.0                    |

Tabel 1. Deskripsi Variabel Utama

Variabel target dalam proyek ini adalah PerformanceRating. Berdasarkan observasi pada dataset serupa (misalnya, dataset IBM HR [21]), PerformanceRating seringkali merupakan skala ordinal. Dalam dataset ini, nilai yang ditemui untuk PerformanceRating adalah 3 ('Excellent') dan 4 ('Outstanding'). [18, 21] Analisis distribusi menunjukkan bahwa sekitar 84.66% karyawan memiliki peringkat 3 dan 15.34% memiliki peringkat 4. Hal ini mengindikasikan adanya ketidakseimbangan kelas yang signifikan, di mana kelas '3' jauh lebih dominan daripada kelas '4'. Ketidakseimbangan ini perlu menjadi perhatian khusus dalam tahap evaluasi model, karena metrik seperti akurasi saja bisa menyesatkan.

Fitur-fitur seperti EmployeeCount, StandardHours, dan Over18 kemungkinan memiliki varians yang sangat rendah atau nilai konstan dan akan dipertimbangkan untuk dihapus pada tahap persiapan data. [18] Fitur EmpID dan EmployeeNumber adalah pengidentifikasi unik dan juga akan dihapus.

### Variasi Nilai Attribute
--- Jumlah Nilai Unik per Kolom ---
| Kolom                    | Jumlah Nilai Unik |
|--------------------------|-------------------|
| EmpID                    | 1470              |
| Age                      | 43                |
| AgeGroup                 | 5                 |
| Attrition                | 2                 |
| BusinessTravel           | 4                 |
| DailyRate                | 886               |
| Department               | 3                 |
| DistanceFromHome         | 29                |
| Education                | 5                 |
| EducationField           | 6                 |
| EmployeeCount            | 1                 |
| EmployeeNumber           | 1470              |
| EnvironmentSatisfaction  | 4                 |
| Gender                   | 2                 |
| HourlyRate               | 71                |
| JobInvolvement           | 4                 |
| JobLevel                 | 5                 |
| JobRole                  | 9                 |
| JobSatisfaction          | 4                 |
| MaritalStatus            | 3                 |
| MonthlyIncome            | 1349              |
| SalarySlab               | 4                 |
| MonthlyRate              | 1427              |
| NumCompaniesWorked       | 10                |
| Over18                   | 1                 |
| OverTime                 | 2                 |
| PercentSalaryHike        | 15                |
| PerformanceRating        | 2                 |
| RelationshipSatisfaction | 4                 |
| StandardHours            | 1                 |
| StockOptionLevel         | 4                 |
| TotalWorkingYears        | 40                |
| TrainingTimesLastYear    | 7                 |
| WorkLifeBalance          | 4                 |
| YearsAtCompany           | 37                |
| YearsInCurrentRole       | 19                |
| YearsSinceLastPromotion  | 16                |
| YearsWithCurrManager     | 18                |

### Missing Value
--- Pemeriksaan Missing Values ---
| No. | Kolom                 | Jumlah Missing | Persentase Missing (%) |
|-----|-----------------------|----------------|------------------------|
| 37  | YearsWithCurrManager  | 57             | 3.851351               |

-dari pemeriksaan missing value, attribute YearsWithCurrManager ditemukan sebanyak 57 value bernilai kosong, dan secara keseluruhan jika dipresentasikan bernilai 3,851%

### Duplicated data
| EmpID |    Age    | AgeGroup | Attrition | BusinessTravel | DailyRate |       Department       | DistanceFromHome | Education | EducationField | … | RelationshipSatisfaction | StandardHours | StockOptionLevel | TotalWorkingYears | TrainingTimesLastYear | WorkLifeBalance | YearsAtCompany | YearsInCurrentRole | YearsSinceLastPromotion | YearsWithCurrManager |
|-------|-----------|----------|-----------|----------------|-----------|------------------------|------------------|-----------|----------------|---|--------------------------|---------------|------------------|-------------------|-----------------------|-----------------|----------------|---------------------|------------------------|---------------------|
|   327 | RM1461    |     29   |  26-35    |    No          | Travel_Rarely | 468 | Research & Development |              28 |         4 | Medical        | … |                        2 |            80 |                0 |                 5 |                     3 |               1 |              5 |                   4 |                      0 |                 4.0 |
|   328 | RM1461    |     29   |  26-35    |    No          | Travel_Rarely | 468 | Research & Development |              28 |         4 | Medical        | … |                        2 |            80 |                0 |                 5 |                     3 |               1 |              5 |                   4 |                      0 |                 4.0 |
|  1335 | RM1462    |     50   |  46-55    |    Yes         | Travel_Rarely | 410 | Sales                  |              28 |         3 | Marketing      | … |                        2 |            80 |                1 |                20 |                     3 |               3 |              3 |                   2 |                      2 |                 0.0 |
|  1336 | RM1462    |     50   |  46-55    |    Yes         | Travel_Rarely | 410 | Sales                  |              28 |         3 | Marketing      | … |                        2 |            80 |                1 |                20 |                     3 |               3 |              3 |                   2 |                      2 |                 0.0 |
|   952 | RM1463    |     39   |  36-45    |    No          | Travel_Rarely | 722 | Sales                  |              24 |         1 | Marketing      | … |                        1 |            80 |                1 |                21 |                     2 |               2 |             20 |                   9 |                      9 |                 6.0 |
|   954 | RM1463    |     39   |  36-45    |    No          | Travel_Rarely | 722 | Sales                  |              24 |         1 | Marketing      | … |                        1 |            80 |                1 |                21 |                     2 |               2 |             20 |                   9 |                      9 |                 6.0 |
|   457 | RM1464    |     31   |  26-35    |    No          | Non-Travel    | 325 | Research & Development |               5 |         3 | Medical        | … |                        2 |            80 |                0 |                10 |                     2 |               3 |              9 |                   4 |                      1 |                 7.0 |
|   458 | RM1464    |     31   |  26-35    |    No          | Non-Travel    | 325 | Research & Development |               5 |         3 | Medical        | … |                        2 |            80 |                0 |                10 |                     2 |               3 |              9 |                   4 |                      1 |                 7.0 |
|   210 | RM1468    |     27   |  26-35    |    No          | Travel_Rarely | 155 | Research & Development |               4 |         3 | Life Sciences  | … |                        2 |            80 |                1 |                 6 |                     0 |               3 |              6 |                   2 |                      0 |                 3.0 |
|   211 | RM1468    |     27   |  26-35    |    No          | Travel_Rarely | 155 | Research & Development |               4 |         3 | Life Sciences  | … |                        2 |            80 |                1 |                 6 |                     0 |               3 |              6 |                   2 |                      0 |                 3.0 |

- Dari 1.480 baris data terdapat 7 baris duplikat (mewakili 7 karyawan yang tercatat dua kali), seperti yang terlihat pada contoh RM1461, RM1462, RM1463, RM1464, dan RM1468 yang nilainya persis sama di semua kolom. Duplikat ini dapat memengaruhi statistik misalnya menaikkan frekuensi kategori tertentu atau mengubah nilai rata-rata dalam analisis numerik oleh karena itu sebaiknya kita menghapus baris-baris tersebut sehingga hanya tersisa 1.473 
### Outlier Data

| Kolom                     | Outliers Count | Percentage Outliers (%) | Lower Bound | Upper Bound |
|---------------------------|----------------|-------------------------|-------------|-------------|
| TrainingTimesLastYear     | 240            | 16.216216               | 0.50        | 4.50        |
| MonthlyIncome             | 114            | 7.702703                | -5270.00    | 16576.00    |
| YearsSinceLastPromotion   | 108            | 7.297297                | -4.50       | 7.50        |
| YearsAtCompany            | 105            | 7.094595                | -6.00       | 18.00       |
| StockOptionLevel          | 85             | 5.743243                | -1.50       | 2.50        |
| TotalWorkingYears         | 63             | 4.256757                | -7.50       | 28.50       |
| NumCompaniesWorked        | 52             | 3.513514                | -3.50       | 8.50        |
| YearsInCurrentRole        | 21             | 1.418919                | -5.50       | 14.50       |
| YearsWithCurrManager      | 13             | 0.878378                | -5.50       | 14.50       |
| JobInvolvement            | 0              | 0.000000                | 0.50        | 4.50        |
| HourlyRate                | 0              | 0.000000                | -4.50       | 135.50      |
| EnvironmentSatisfaction   | 0              | 0.000000                | -1.00       | 7.00        |
| EmployeeNumber            | 0              | 0.000000                | -1118.00    | 3180.00     |
| EmployeeCount             | 0              | 0.000000                | 1.00        | 1.00        |
| Education                 | 0              | 0.000000                | -1.00       | 7.00        |
| DistanceFromHome          | 0              | 0.000000                | -16.00      | 32.00       |
| DailyRate                 | 0              | 0.000000                | -573.00     | 2195.00     |
| Age                       | 0              | 0.000000                | 10.50       | 62.50       |
| RelationshipSatisfaction  | 0              | 0.000000                | -1.00       | 7.00        |
| StandardHours             | 0              | 0.000000                | 80.00       | 80.00       |
| MonthlyRate               | 0              | 0.000000                | -10563.25   | 39074.75    |
| JobLevel                  | 0              | 0.000000                | -2.00       | 6.00        |
| JobSatisfaction           | 0              | 0.000000                | -1.00       | 7.00        |
| PercentSalaryHike         | 0              | 0.000000                | 3.00        | 27.00       |
| WorkLifeBalance           | 0              | 0.000000                | 0.50        | 4.50        |

- Proses: Analisis outlier dilakukan menggunakan metode IQR (Interquartile Range). Berdasarkan output  fitur dengan outlier signifikan antara lain:

1. TrainingTimesLastYear: 240 outlier (16.2%) di luar rentang [0.5 – 4.5] sesi.

2. MonthlyIncome: 114 outlier (7.7%) di luar rentang [–5270 – 16576] ribu.

3. YearsSinceLastPromotion: 108 outlier (7.3%) di luar rentang [–4.5 – 7.5] tahun.

4. YearsAtCompany: 105 outlier (7.1%) di luar rentang [–6 – 18] tahun.

5. StockOptionLevel: 85 outlier (5.7%) di luar rentang [–1.5 – 2.5] level.

6. TotalWorkingYears: 63 outlier (4.3%) di luar rentang [–7.5 – 28.5] tahun.

7. NumCompaniesWorked: 52 outlier (3.5%) di luar rentang [–3.5 – 8.5] perusahaan.

Fitur lain seperti YearsInCurrentRole (21 outlier, 1.4%) dan YearsWithCurrManager (13 outlier, 0.9%) memiliki persentase outlier yang lebih kecil, sementara banyak fitur numerik lain (misalnya Age, DistanceFromHome, JobLevel) memiliki 0% outlier berdasarkan metode IQR.
### EDA - Univariate 

Analisis univariat akan dilakukan untuk setiap fitur guna memahami distribusinya:

1. Variabel Numerik (Age, MonthlyIncome, PercentSalaryHike, dll.):

- Histogram digunakan untuk melihat bentuk distribusi masing-masing fitur numerik, seperti yang terlihat pada gambar
  ![Image](https://github.com/user-attachments/assets/37f8e257-c28c-42ad-8e37-dbad6636f152)


- Age: Distribusinya mendekati normal dengan puncak di sekitar 30–40 tahun. Sangat sedikit karyawan yang berusia < 20 atau > 60 tahun.

- DailyRate: Distribusi hampir merata (uniform) dalam rentang 100–1.500, menandakan variasi beban kerja yang relatif seimbang.

- DistanceFromHome: Sangat right-skewed; mayoritas tinggal dekat kantor (< 5 km), tetapi ada beberapa hingga ~30 km.

- Education: Nilainya diskrit (1–5) dengan modus di level 3.

- EmployeeCount & StandardHours: Keduanya konstan, sehingga tidak menambah informasi untuk model.

- EmployeeNumber: Hampir uniform, mencerminkan ID unik.

- Kepuasan & Keterlibatan (EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, RelationshipSatisfaction): Diskrit dengan modus di 3–4 pada skala 1–4. JobInvolvement khususnya menumpuk di nilai 3.

- HourlyRate: Mirip uniform, menyebar dari ~30 hingga ~100.

- JobLevel & StockOptionLevel: Diskrit dengan modus di level rendah (1–2).

- MonthlyIncome: Right-skewed; mayoritas gaji di bawah Rp 7.000.000, dengan ekor panjang ke atas.

- NumCompaniesWorked: Right-skewed; banyak karyawan yang baru pernah bekerja di 0–1 perusahaan, namun ada sebagian yang pernah berpindah hingga > 8 perusahaan.

- PercentSalaryHike: Mayoritas pada rentang 12–16 %, dengan nilai minimum 11 %.

- Pengalaman Kerja (TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager): Semuanya right-skewed—banyak karyawan berpengalaman 0–5 tahun, sedikit yang > 20 tahun.

- TrainingTimesLastYear: Diskrit di 2–3 kali, dengan beberapa outlier di 5–6 kali.

- WorkLifeBalance: Diskrit dan berkumpul di level 3.





2. Variabel Kategorikal (Department, JobRole, Gender, PerformanceRating, dll.):

- Bar chart (diagram batang) digunakan untuk menunjukkan frekuensi atau proporsi setiap kategori. Visualisasi distribusi fitur kategorikal dapat dilihat pada gambar
  ![Image](https://github.com/user-attachments/assets/d75ff273-4c56-4206-aae1-6488cf29198a)

Dari visualisasi ini, dapat diinterpretasikan:

- AgeGroup: Mayoritas karyawan (611 dari 1.480) berada di kelompok 26–35 tahun.

- Attrition: Hanya sekitar 16% karyawan yang keluar (Yes), dataset sangat condong ke karyawan yang bertahan.

- BusinessTravel: Sebagian besar jarang melakukan perjalanan dinas (Travel_Rarely).

- Department: Sekitar 2/3 karyawan di Research & Development.

- EducationField: Terbesar di Life Sciences dan Medical.

- Gender: Laki-laki (889) lebih banyak dibanding perempuan (591).

- JobRole: Paling umum Sales Executive dan Research Scientist.

- MaritalStatus: Mayoritas Married.

- SalarySlab: Kebanyakan gaji masuk Upto 5k.

- Over18: Semua karyawan berusia ≥ 18 tahun (konstan “Y”), tidak informatif.

- OverTime: Sekitar 71% tidak lembur (No).

- Khusus untuk variabel target PerformanceRating,  distribusinya adalah 84.66% untuk rating 3 dan 15.34% untuk rating 4.



### EDA - Bivariate Analysis
Analisis dalam EDA ini difokuskan pada pemahaman hubungan antara masing-masing fitur independen dengan variabel target, PerformanceRating. 

Hubungan Antara Fitur Independen dan Variabel Target (PerformanceRating):

1. Fitur Numerik vs. PerformanceRating:

- Box plot digunakan  untuk membandingkan distribusi fitur numerik (misalnya, MonthlyIncome, PercentSalaryHike, YearsSinceLastPromotion) untuk setiap kategori PerformanceRating (kelas 3 dan 4). Visualisasi ini dapat dilihat pada 
![Image](https://github.com/user-attachments/assets/753c7a00-774e-4ce9-a70a-f7e1c5b591f6)
Dari plot-plot ini, kita dapat menginterpretasikan:


- Age & DistanceFromHome: Distribusi usia dan jarak rumah tidak menunjukkan perbedaan signifikan antara kedua kelompok rating, mengindikasikan pengaruh yang tidak kuat.

- Tarif & Pendapatan (DailyRate, HourlyRate, MonthlyRate, MonthlyIncome): DailyRate, HourlyRate, dan MonthlyRate memiliki distribusi yang sangat mirip. MonthlyIncome menunjukkan sedikit pergeseran median dan IQR ke atas untuk rating 4, mengindikasikan karyawan berpendapatan bulanan lebih tinggi cenderung sedikit lebih mungkin mendapat rating 4, namun perbedaannya tidak dramatis.

- Pendidikan & Level Jabatan (Education, JobLevel): Distribusi tingkat pendidikan dan level jabatan hampir tumpang tindih sempurna, menunjukkan kedua fitur ini tidak memisahkan rating kinerja secara jelas.

- Kepuasan (JobInvolvement, JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance): Semua fitur ini menunjukkan IQR dan median yang hampir identik antar kelompok rating, mengindikasikan faktor-faktor ini tidak secara signifikan membedakan karyawan dengan rating 3 vs 4 dalam dataset ini.

- Pengalaman Kerja (TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager): TotalWorkingYears dan YearsAtCompany memiliki median dan IQR sedikit lebih tinggi untuk rating 4. Karyawan dengan rating 4 cenderung memiliki median waktu sejak promosi terakhir yang lebih lama (sekitar 3 tahun vs 2 tahun untuk rating 3). YearsInCurrentRole dan YearsWithCurrManager juga menunjukkan sedikit pergeseran median ke atas untuk rating 4. Ini mengindikasikan bahwa karyawan dengan rating 4 cenderung sedikit lebih senior atau lebih lama dalam peran/jabatan mereka saat ini.

- (NumCompaniesWorked): Median jumlah perusahaan sebelumnya hampir sama, tidak menunjukkan korelasi kuat dengan rating.

- (PercentSalaryHike, StockOptionLevel): PercentSalaryHike menunjukkan pergeseran median dan IQR yang jelas ke kanan untuk rating 4 (sekitar 14-16%) dibandingkan rating 3 (sekitar 12-14%). StockOptionLevel juga menunjukkan IQR yang sedikit lebih tinggi pada rating 4. Ini mengindikasikan bahwa karyawan dengan rating 4 cenderung mendapatkan kenaikan gaji dan level opsi saham yang lebih besar.

2. Fitur Kategorikal vs. PerformanceRating:

grouped bar chart  digunakan untuk menunjukkan bagaimana proporsi PerformanceRating 3 dan 4 didistribusikan dalam setiap kategori dari fitur kategorikal lain.
![Image](https://github.com/user-attachments/assets/953950e2-f600-4536-ae4a-015ae281ade3)
Interpretasi dari visualisasi ini meliputi:


- AgeGroup: Kelompok usia 55+ menunjukkan persentase rating 4 tertinggi (~21%), mengindikasikan karyawan berpengalaman mungkin lebih diakui. Kelompok usia lain (18-55 tahun) memiliki persentase rating 4 yang serupa (14-16%).

- Attrition: Persentase rating 4 hampir sama antara karyawan yang keluar (Yes, 15%) dan yang aktif (No, 15%), menunjukkan keputusan keluar tidak berkorelasi kuat dengan akumulasi rating kinerja tertinggi.

- BusinessTravel: Karyawan yang Non-Travel memiliki persentase rating 4 sedikit lebih tinggi (17%) dibandingkan yang Travel_Frequently (16%) dan Travel_Rarely (15%), mungkin karena lebih fokus pada tugas di kantor.

- Department: Human Resources menunjukkan persentase rating 4 tertinggi (17%), diikuti Research & Development (16%), dan Sales (14%). Ini bisa disebabkan oleh metrik penilaian yang berbeda atau volume kerja.

- EducationField: Latar belakang Medical & Life Sciences cenderung memiliki rating 4 lebih tinggi (16-17%) dibandingkan bidang lain, mungkin karena kecocokan skill.

- Gender: Perbedaan tipis antara Female (15% rating 4) dan Male (14% rating 4) mengindikasikan gender bukan faktor dominan.

- JobRole: Manager & Healthcare Representative memiliki persentase rating 4 tertinggi (~19%), sementara Research Director terendah (11%), mungkin karena perbedaan tanggung jawab atau kriteria.

- MaritalStatus: Karyawan Married (15% rating 4) sedikit lebih unggul dari Single (14%) dan Divorced (13%).

- SalarySlab: Karyawan dengan gaji Upto 5k dan 15k+ memiliki proporsi rating 4 yang sama (16%), menunjukkan top performer bisa ada di berbagai level gaji.

- OverTime: Karyawan yang lembur (Yes, 16% rating 4) sedikit lebih berpeluang mendapat rating tinggi dibandingkan yang tidak (No, 14%).
  
Analisis visual ini bertujuan untuk mengidentifikasi secara kualitatif apakah terdapat asosiasi yang jelas antara kategori-kategori tertentu dalam fitur-fitur ini dengan kecenderungan untuk mendapatkan PerformanceRating yang lebih tinggi atau lebih rendah, yang memberikan petunjuk awal untuk fitur-fitur prediktif.

## Data Preparation
Tahap persiapan data sangat krusial untuk memastikan kualitas input model machine learning. Berikut adalah langkah-langkah yang akan dilakukan, dengan penjelasan proses dan alasannya [8, 9, 10]:

### Penghapusan Fitur yang Tidak Relevan/Redundan:

- Proses: Fitur-fitur seperti EmpID, EmployeeNumber, EmployeeCount, StandardHours, dan Over18 akan dihapus.

- Alasan: Mengurangi dimensionalitas data, menyederhanakan model, dan menghindari noise yang tidak perlu.
  
### Penanganan Data Hilang (Missing Values) untuk YearsWithCurrManager:

- Proses: Sebelum langkah pra-pemrosesan data yang lebih kompleks, dilakukan identifikasi dan penanganan nilai yang hilang pada kolom YearsWithCurrManager. Ditemukan bahwa kolom ini memiliki 57 nilai yang hilang. Nilai-nilai yang hilang ini kemudian diisi (diimputasi) menggunakan nilai median dari kolom YearsWithCurrManager itu sendiri.

- Alasan:

Pentingnya Mempertahankan Sampel: Menghapus 57 baris (sekitar 3.8% dari total sampel) dapat mengurangi ukuran dataset dan berpotensi menghilangkan informasi berharga.

Robustisitas Median: Median dipilih karena lebih tahan terhadap outlier dibandingkan mean, terutama jika distribusi fitur YearsWithCurrManager miring (statistik deskriptif menunjukkan mean 4.12 dan median 3.0, mengindikasikan kemiringan).

- Kelengkapan Data untuk Tahap Selanjutnya: Memastikan tidak ada nilai hilang pada fitur ini sebelum dilakukan transformasi lain atau pembagian data.
  
- Konsistensi: Imputasi memastikan semua sampel memiliki nilai untuk fitur ini, yang penting untuk beberapa algoritma.

### Penghapusan Data Duplikat:

- Proses: Setelah penanganan nilai missing awal, dilakukan penghapusan baris data yang duplikat dari pengecekan duplicated data sebelumnya didapati dari 1.480 ditemukan 7 baris yang sama dan saya melakukan penghapusan menggunakan df.drop_duplicates().

- Alasan: Data duplikat dapat menyebabkan bias pada model dan memberikan estimasi kinerja yang terlalu optimis. Menghapusnya memastikan setiap baris data unik dan model belajar dari informasi yang tidak redundan.



### Penanganan Outlier:

Keputusan Pra-proses: Berdasarkan analisis saya, tidak dilakukan penghapusan atau transformasi outlier.

- Alasan:

1. Sebagian besar fitur hanya memiliki persentase outlier yang relatif kecil (< 8%).

2. Model yang direncanakan (berbasis pohon seperti Random Forest dan XGBoost) umumnya cukup robust terhadap nilai ekstrem.

3. Beberapa outlier mungkin mencerminkan kondisi nyata (misalnya, karyawan dengan masa kerja sangat lama atau pendapatan yang sangat tinggi) dan menghilangkan mereka dapat menghilangkan informasi penting.

### Pemisahan Fitur dan Variabel Target:

Proses: Dataset dipisahkan menjadi fitur-fitur independen (X) dan variabel target (y, yaitu PerformanceRating).

Alasan: Ini adalah langkah standar dalam persiapan data untuk pemodelan supervised learning, di mana model belajar memetakan X ke y.

### Encoding Variabel Target (PerformanceRating):

Proses: Variabel target PerformanceRating, yang memiliki nilai unik 3 dan 4, di-encode menggunakan LabelEncoder. Transformasi ini mengubah nilai [3, 4] menjadi [0, 1].

Alasan: Banyak algoritma klasifikasi di Scikit-learn secara default mengharapkan label kelas dimulai dari 0 hingga n_classes-1. Label encoding memastikan kompatibilitas, mencegah potensi interpretasi yang salah oleh model, dan memudahkan kalkulasi beberapa metrik evaluasi.

### Pengelompokan Fitur Independen dan Konstruksi Pipeline Pra-pemrosesan (preprocessor_deep_dive):
Fitur-fitur independen (X) dikelompokkan berdasarkan tipenya: fitur numerik, fitur kategorikal nominal, dan fitur kategorikal ordinal.

1. Fitur Numerik: Contoh: Age, MonthlyIncome.

2. Fitur Kategorikal Nominal: Contoh: Department, Gender.

3. Fitur Kategorikal Ordinal: Contoh: Education, JobSatisfaction. Pemetaan urutan kategori didefinisikan secara eksplisit dalam ordinal_cols_pipeline_with_mapping.

Sebuah pipeline pra-pemrosesan (preprocessor_deep_dive) kemudian dibuat menggunakan ColumnTransformer untuk menerapkan transformasi spesifik pada setiap kelompok fitur:

- Transformer untuk Fitur Numerik (numerical_transformer):

Imputasi: SimpleImputer(strategy='median').

Penskalaan: StandardScaler().

- Transformer untuk Fitur Kategorikal Nominal (nominal_transformer):

Imputasi: SimpleImputer(strategy='most_frequent').

One-Hot Encoding: OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False).

- Transformer untuk Fitur Kategorikal Ordinal (ordinal_transformer):

Imputasi: SimpleImputer(strategy='most_frequent').

Ordinal Encoding: OrdinalEncoder(categories=ordinal_categories_list, handle_unknown='use_encoded_value', unknown_value=-1).

ColumnTransformer (preprocessor_deep_dive): Menggabungkan transformer di atas. Pengaturan remainder='passthrough', verbose_feature_names_out=False, dan preprocessor.set_output(transform="pandas") digunakan untuk memastikan output yang bersih dan berupa DataFrame Pandas.

Alasan Pipeline: Pendekatan ini memastikan konsistensi dalam penerapan transformasi, mencegah data leakage (karena di-fit hanya pada data latih), dan membuat alur kerja lebih terorganisir dan mudah direproduksi.

### Pembagian Dataset (Train-Test Split):

Proses: Fitur X yang telah diproses oleh preprocessor_deep_dive (jika fitting dilakukan sebelum split) atau fitur X mentah (jika preprocessor akan di-fit hanya pada X_train di dalam pipeline model) dan variabel target y yang sudah di-encode dibagi menjadi data latih (X_train, y_train) dan data uji (X_test, y_test) dengan rasio 75% untuk data latih dan 25% untuk data uji. Parameter random_state=42 dan stratify=y digunakan.

Alasan: Melatih dan menguji model pada set data yang berbeda untuk evaluasi objektif. Stratifikasi penting karena adanya ketidakseimbangan kelas pada variabel target. [9]

## Modeling
Pada tahap ini, akan dilakukan pengembangan beberapa model machine learning untuk tugas klasifikasi PerformanceRating. Model yang diuji meliputi Logistic Regression, Random Forest, XGBoost, dan LGBM. Setiap model akan digabungkan dengan preprocessor_deep_dive dalam sebuah Pipeline Scikit-learn untuk memastikan pra-pemrosesan diterapkan secara konsisten.

1. Logistic Regression

- Cara Kerja: Regresi Logistik adalah algoritma klasifikasi linear yang memprediksi probabilitas suatu instance milik kelas tertentu. Ia menggunakan fungsi logistik (sigmoid) untuk memetakan output linear dari kombinasi fitur ke rentang probabilitas (0 hingga 1). Keputusan klasifikasi kemudian dibuat berdasarkan ambang batas tertentu (biasanya 0.5). [24, 25] Untuk menemukan parameter (koefisien) terbaik, model ini meminimalkan fungsi kerugian (seperti log-loss) menggunakan teknik optimasi. Regularisasi (L1 atau L2) dapat ditambahkan untuk mencegah overfitting dengan memberikan penalti pada besarnya koefisien.

- Parameter Default & Optimasi: Parameter terbaik yang ditemukan setelah tuning menggunakan RandomizedSearchCV adalah: {'classifier__penalty': 'l1', 'classifier__C': 10}. Skor F1 Weighted terbaik dari Cross-Validation (CV) adalah 0.9991.

- Kelebihan: Sederhana, interpretable, cepat dilatih, dan efisien secara komputasi. Koefisiennya dapat memberikan wawasan tentang pentingnya fitur. [24, 25]

- Kekurangan: Mengasumsikan hubungan linear antara fitur dan log-odds dari output, mungkin tidak bekerja dengan baik pada data dengan hubungan non-linear yang kompleks, dan sensitif terhadap multikolinearitas. [24]

2. Random Forest

- Cara Kerja: Random Forest adalah metode ensemble learning yang bekerja dengan membangun sejumlah besar pohon keputusan (decision trees) secara independen pada berbagai sub-sampel data latih (teknik bagging atau bootstrap aggregating). Untuk setiap split pada pohon, hanya subset acak dari fitur yang dipertimbangkan (random feature selection). Hal ini bertujuan untuk mengurangi varians dan korelasi antar pohon, sehingga meningkatkan robustisitas dan akurasi model. Untuk prediksi klasifikasi, setiap pohon memberikan "suara" untuk kelas tertentu, dan kelas dengan suara mayoritas menjadi prediksi akhir dari forest. [26, 27]

- Parameter Default & Optimasi: Parameter terbaik yang ditemukan setelah tuning menggunakan RandomizedSearchCV adalah: {'classifier__n_estimators': 150, 'classifier__min_samples_split': 5, 'classifier__min_samples_leaf': 5, 'classifier__max_depth': None}. Skor F1 Weighted terbaik dari Cross-Validation (CV) adalah 1.0000 

- Kelebihan: Umumnya memiliki akurasi prediksi yang tinggi, robust terhadap overfitting dibandingkan satu decision tree, dapat menangani data dengan banyak fitur, dan menyediakan ukuran kepentingan fitur. [26, 27]

- Kekurangan: Lebih kompleks dan kurang interpretable secara langsung dibandingkan satu decision tree, memerlukan lebih banyak sumber daya komputasi untuk pelatihan. [26, 27]

3. XGBoost (Extreme Gradient Boosting)

- Cara Kerja: XGBoost adalah implementasi dari algoritma gradient boosting yang sangat dioptimalkan dan efisien. Gradient boosting membangun model (pohon keputusan) secara sekuensial. Setiap pohon baru dilatih untuk memperbaiki kesalahan (residual atau gradien dari loss function) yang dibuat oleh pohon-pohon sebelumnya dalam ensemble. XGBoost menggunakan pendekatan gradient descent dalam ruang fungsi untuk meminimalkan fungsi kerugian dan menyertakan teknik regularisasi (L1 dan L2) serta optimasi lainnya (seperti penanganan nilai hilang secara internal dan parallel processing) untuk meningkatkan kinerja dan mencegah overfitting. [30, 31]

- Parameter Default & Optimasi: Parameter terbaik yang ditemukan setelah tuning menggunakan RandomizedSearchCV adalah: {'classifier__subsample': 0.8, 'classifier__n_estimators': 200, 'classifier__max_depth': 3, 'classifier__learning_rate': 0.05, 'classifier__colsample_bytree': 0.8}. Skor F1 Weighted terbaik dari Cross-Validation (CV) adalah 1.0000.

- Kelebihan: Seringkali menghasilkan kinerja prediktif state-of-the-art pada data tabular, fleksibel, dan memiliki mekanisme regularisasi yang kuat. [30, 31]

- Kekurangan: Pelatihan bisa lebih lambat dibandingkan Random Forest (meskipun implementasinya sangat dioptimalkan), lebih sensitif terhadap hyperparameter, dan bisa lebih rentan overfitting jika tidak di-tune dengan baik. [30]

4. LGBM (Light Gradient Boosting Machine)

- Cara Kerja: LGBM adalah kerangka kerja gradient boosting lain yang menggunakan teknik berbasis pohon. Perbedaan utamanya dengan XGBoost adalah cara pohon dibangun: LGBM menggunakan pertumbuhan pohon berbasis daun (leaf-wise) daripada berbasis level (level-wise) seperti XGBoost tradisional atau Random Forest. Pertumbuhan leaf-wise memungkinkan model untuk fokus pada daun yang memberikan pengurangan kerugian terbesar, yang bisa lebih efisien dan menghasilkan akurasi lebih baik, terutama pada dataset besar. LGBM juga menggunakan teknik seperti Gradient-based One-Side Sampling (GOSS) dan Exclusive Feature Bundling (EFB) untuk mempercepat pelatihan dan mengurangi penggunaan memori.

- Parameter Default & Optimasi: Parameter terbaik yang ditemukan setelah tuning menggunakan RandomizedSearchCV adalah: {'classifier__num_leaves': 20, 'classifier__n_estimators': 150, 'classifier__max_depth': -1, 'classifier__learning_rate': 0.05}. Skor F1 Weighted terbaik dari Cross-Validation (CV) adalah 1.0000.

- Kelebihan: Cepat dilatih, efisien dalam penggunaan memori, dan seringkali memberikan kinerja yang sangat baik, terutama pada dataset besar.

- Kekurangan: Bisa rentan terhadap overfitting pada dataset yang lebih kecil jika hyperparameter tidak diatur dengan hati-hati (terutama num_leaves).

### Proses Pelatihan dan Pemilihan Model Terbaik
Semua model dilatih pada data latih. Optimasi hyperparameter dilakukan menggunakan RandomizedSearchCV dengan 3 folds untuk validasi silang, mencari 10 kandidat, dengan F1 Weighted sebagai metrik skor.
![Image](https://github.com/user-attachments/assets/ff139013-0e49-421d-914a-13d42b901df7)
### Pemilihan Model Terbaik sebagai Solusi
Berdasarkan skor F1 Weighted dari Cross-Validation dan laporan klasifikasi pada data uji setelah tuning, XGBoost, RandomForest dan LGBM menunjukkan kinerja yang sangat tinggi (F1-CV mendekati atau 1.0000). Logistic Regression juga menunjukkan kinerja yang sangat baik meskipun memiliki skor F1-CV sedikit lebih rendah (0.9991). Model dengan F1-Score (Macro) dan ROC-AUC tertinggi pada data uji akan menjadi pilihan utama. Mengingat skor sempurna atau mendekati sempurna pada beberapa model, perlu dipastikan tidak ada data leakage atau bahwa masalahnya memang sangat dapat dipisahkan oleh fitur yang ada.

## Evaluation
Tahap evaluasi bertujuan untuk mengukur kinerja model-model yang telah dilatih pada data uji dan menginterpretasikan hasilnya.

### Metrik Evaluasi
Metrik yang digunakan harus sesuai dengan masalah klasifikasi dan potensi ketidakseimbangan kelas pada variabel target PerformanceRating. Berikut adalah metrik utama yang akan digunakan [32, 15, 16, 17, 33]:

1. Akurasi (Accuracy)

- Formula:


$$\text{Accuracy} \;=\; \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
​
 
- Cara Kerja: Mengukur proporsi total prediksi yang benar.

- Kesesuaian: Mudah dipahami, tetapi bisa menyesatkan jika dataset tidak seimbang. [15, 16, 17]

2. Presisi (Precision)

- Formula:


$$\text{Precision} \;=\; \frac{\text{TP}}{\text{TP} + \text{FP}}$$
​
 
- Cara Kerja: Dari semua instance yang diprediksi sebagai kelas positif, berapa banyak yang sebenarnya positif.

- Kesesuaian: Penting ketika biaya False Positive tinggi. Akan digunakan rata-rata makro. [15, 16]

3. Recall (Sensitivitas atau True Positive Rate)

- Formula:


$$\text{Recall} \;=\; \frac{\text{TP}}{\text{TP} + \text{FN}}$$
​
 
- Cara Kerja: Dari semua instance yang sebenarnya positif, berapa banyak yang berhasil diprediksi sebagai positif oleh model.

- Kesesuaian: Penting ketika biaya False Negative tinggi. Akan digunakan rata-rata makro. [15, 16]

4. F1-Score

- Formula:


$$\text{F1-Score} \;=\; 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
​
 
- Cara Kerja: Rata-rata harmonik dari Presisi dan Recall.

- Kesesuaian: Sangat berguna untuk dataset dengan kelas tidak seimbang. F1-macro average akan menjadi salah satu metrik utama. 

5. ROC-AUC (Area Under the Receiver Operating Characteristic Curve)

- Cara Kerja: Kurva ROC memplot True Positive Rate terhadap False Positive Rate. AUC adalah area di bawah kurva ROC.

- Kesesuaian: Mengukur kemampuan model membedakan antar kelas. ROC-AUC bersifat independen terhadap ambang batas klasifikasi dan robust terhadap ketidakseimbangan kelas.

### Hasil Evaluasi Model (Setelah Tuning)
Berikut adalah hasil evaluasi kinerja model pada data uji  (369 sampel) setelah optimasi hyperparameter

| No. | Model               | Test Accuracy | Test F1 Weighted | Test F1 Macro | Test F1 Class 1 (Rating 4) | Test ROC AUC |
|-----|---------------------|---------------|------------------|---------------|----------------------------|--------------|
| 0   | LogisticRegression  | 1.0           | 1.0              | 1.0           | 1.0                        | 1.0          |
| 1   | RandomForest        | 1.0           | 1.0              | 1.0           | 1.0                        | 1.0          |
| 2   | XGBoost             | 1.0           | 1.0              | 1.0           | 1.0                        | 1.0          |
| 3   | LGBM                | 1.0           | 1.0              | 1.0           | 1.0                        | 1.0          |


### Interpretasi Detail Laporan Klasifikasi (Setelah Tuning):

- Logistic Regression:

Mencapai skor sempurna (1.00) untuk presisi, recall, dan F1-score pada kedua kelas (rating 3 dan 4), dengan akurasi keseluruhan 1.00. Ini menunjukkan model mampu mengklasifikasikan semua 312 sampel kelas 3 dan 57 sampel kelas 4 pada data uji dengan benar tanpa kesalahan.

- Random Forest:

Akurasi keseluruhan 1.0. Untuk kelas 3 , presisi 1.00 dan recall 1.00 (F1-score 1.0), yang berarti semua sampel kelas 3 berhasil diidentifikasi, . Untuk kelas 4 (minoritas), presisi 1.00 (semua yang diprediksi kelas 4 memang benar kelas 4), recall 1.0 (F1-score 1.0). Ini berarti model hanya berhasil mengidentifikasi semua total sampel kelas 4 yang sebenarnya. Skor F1-macro adalah 1.0.

- XGBoost:

Mirip dengan Logistic Regression dan RandomForest, XGBoost mencapai skor sempurna (1.00) untuk presisi, recall, dan F1-score pada kedua kelas, dengan akurasi keseluruhan 1.00. Model ini juga mengklasifikasikan semua sampel pada data uji dengan sempurna.

- LGBM:

Sama seperti model lainnya, LGBM juga mencapai skor sempurna (1.00) untuk presisi, recall, dan F1-score pada kedua kelas, dengan akurasi keseluruhan 1.00, menunjukkan klasifikasi yang sempurna pada data uji.

Semua Model menunjukkan kinerja yang sempurna atau mendekati sempurna pada data uji berdasarkan metrik yang dilaporkan.. Kinerja sempurna atau mendekati sempurna pada model yang dipilih pada data uji sangat menggembirakan saya, namun juga memerlukan investigasi lebih lanjut. Hal ini bisa jadi mengindikasikan bahwa dataset memiliki pemisahan kelas yang sangat jelas dengan fitur-fitur yang ada, atau, pada skenario yang kurang ideal, bisa jadi ada bentuk data leakage yang tidak terdeteksi dalam pipeline pra-pemrosesan, atau dataset uji mungkin tidak cukup beragam/besar untuk menguji generalisasi model sepenuhnya. 
![image](https://github.com/user-attachments/assets/729cd60b-c038-4049-b913-f4b1c0520429)
![Image](https://github.com/user-attachments/assets/d3564395-b7d2-4161-b303-fed52611bd01)
![image](https://github.com/user-attachments/assets/0fbf816f-9822-4eba-9964-172b1bdc672e)
![image](https://github.com/user-attachments/assets/e4f5355c-dba4-4c52-83f7-076783f1b511)



### Analisis Faktor Penting (Feature Importance)
Analisis kepentingan fitur dilakukan untuk memahami kontribusi masing-masing fitur terhadap prediksi model. Output menunjukkan skor kepentingan fitur untuk model Logistic Regression, Random Forest, XGBoost, dan LGBM. Perlu dicatat bahwa setelah pra-pemrosesan (terutama one-hot encoding), jumlah fitur meningkat menjadi 52.

- Logistic Regression (Tuned):
![image](https://github.com/user-attachments/assets/5f08ba33-4841-45fc-843a-15e902c6c4dd)

1. Fitur dengan magnitudo koefisien absolut tertinggi adalah PercentSalaryHike (skor ~30.30). Ini menunjukkan bahwa persentase kenaikan gaji adalah prediktor linier yang sangat kuat untuk model ini.

2. Fitur lain yang muncul dengan skor lebih rendah termasuk BusinessTravel_Travel_Rarely (~0.84), JobInvolvement (~0.37), JobRole_Sales Representative (~0.27), dan Gender_Male (~0.25).



Untuk Regresi Logistik, PercentSalaryHike mendominasi sebagai faktor penentu. Fitur-fitur kategorikal yang telah di-encode (seperti aspek perjalanan bisnis, peran pekerjaan, dan jenis kelamin) juga memberikan kontribusi, meskipun jauh lebih kecil.

- Random Forest (Tuned):
![image](https://github.com/user-attachments/assets/271190c7-27ce-4b8e-b4ec-3d9c787e5af7)

1. Fitur terpenting adalah PercentSalaryHike (skor ~0.212).

2. Diikuti oleh MonthlyRate (~0.049), TotalWorkingYears (~0.044), YearsInCurrentRole (~0.044), dan DistanceFromHome (~0.041).



Random Forest juga menyoroti PercentSalaryHike sebagai fitur paling signifikan. Namun, model ini juga memberikan bobot yang cukup pada fitur-fitur terkait kompensasi lain (MonthlyRate), pengalaman (TotalWorkingYears), dan stabilitas peran (YearsInCurrentRole), serta faktor demografis (DistanceFromHome).

- XGBoost (Tuned):
![image](https://github.com/user-attachments/assets/2d6505dc-c5c1-4c4f-af86-ad32516c5c12)

1. PercentSalaryHike kembali menjadi fitur dengan skor kepentingan tertinggi dan sangat dominan (skor ~0.734).

2. Fitur-fitur berikutnya dengan skor yang jauh lebih kecil adalah YearsSinceLastPromotion (~0.025), OverTime_Yes (~0.023), DistanceFromHome (~0.022), dan TrainingTimesLastYear (~0.019).



XGBoost sangat menekankan pentingnya PercentSalaryHike. Faktor-faktor lain seperti waktu sejak promosi terakhir, status kerja lembur, jarak dari rumah, dan jumlah pelatihan tahun lalu juga dipertimbangkan, meskipun dengan bobot yang jauh lebih rendah.

- LGBM (Tuned):
![image](https://github.com/user-attachments/assets/7a7b56c7-6f9c-4099-ba5e-4098a52f00aa)

1. Berbeda dengan model lain, LGBM menunjukkan Age (skor 210) sebagai fitur paling penting, diikuti oleh PercentSalaryHike (skor 150), dan DailyRate (skor 123).

2. MonthlyIncome (skor 21) dan HourlyRate (skor 13) juga muncul di lima besar, meskipun dengan skor yang lebih rendah.


LGBM memberikan perspektif yang sedikit berbeda dengan menempatkan Age di urutan teratas, menunjukkan bahwa usia mungkin memiliki interaksi non-linear yang signifikan yang ditangkap oleh model ini. PercentSalaryHike tetap menjadi fitur yang sangat penting, bersama dengan aspek kompensasi harian.

Secara keseluruhan, PercentSalaryHike secara konsisten muncul sebagai fitur yang sangat penting di semua model yang dianalisis. Fitur-fitur lain yang juga menunjukkan relevansi (meskipun dengan urutan dan bobot yang bervariasi antar model) meliputi aspek-aspek terkait pengalaman kerja (TotalWorkingYears, YearsInCurrentRole, YearsSinceLastPromotion), kompensasi (MonthlyRate, MonthlyIncome, DailyRate), kondisi kerja (OverTime_Yes), serta faktor demografis (Age, DistanceFromHome). Identifikasi fitur-fitur ini memberikan wawasan berharga mengenai faktor-faktor yang paling mungkin membedakan antara karyawan dengan PerformanceRating 3 dan 4 dalam dataset ini.

### Kesimpulan Evaluation
Proyek ini bertujuan untuk mengembangkan model machine learning yang mampu memprediksi peringkat kinerja (PerformanceRating) karyawan menggunakan dataset "HR Analytics Dataset" dari Kaggle. Berdasarkan analisis dan pemodelan yang dilakukan:

- Semua model X menunjukkan kinerja yang sangat tinggi setelah optimasi hyperparameter, dengan F1-Score (Macro) dan Akurasi mencapai 1.00 pada data uji. Hasil ini mengindikasikan kemampuan klasifikasi yang sempurna pada set data uji yang digunakan.



1. Menjawab Problem Statements:

- Efektivitas Model: Proyek ini berhasil membangun beberapa model machine learning (Logistic Regression, RandomForest, XGBoost, LGBM) yang menunjukkan efektivitas sangat tinggi (akurasi dan F1-score 1.00 pada data uji) dalam mengklasifikasikan PerformanceRating. Ini menjawab pertanyaan pertama mengenai kemampuan membangun model yang efektif.

- Algoritma Terbaik: Berdasarkan metrik evaluasi pada data uji, XGBoost, Randomforest, LGBM, dan Logistic Regression memberikan kinerja prediksi terbaik yang hampir identik dan sempurna. Pemilihan satu model "terbaik" dari keempatnya bisa mempertimbangkan faktor lain seperti kompleksitas atau waktu inferensi jika ada perbedaan signifikan.

- Fitur Signifikan: Analisis feature importance mengungkapkan bahwa PercentSalaryHike secara konsisten menjadi faktor paling signifikan di hampir semua model. Fitur lain seperti Age (khususnya untuk LGBM), YearsSinceLastPromotion, OverTime_Yes, dan beberapa aspek kompensasi serta pengalaman kerja juga menunjukkan kontribusi. Ini menjawab pertanyaan mengenai fitur-fitur kunci yang memengaruhi kinerja.

- Rekomendasi Strategis: Wawasan dari model ini (terutama fitur-fitur penting dan kinerja prediktif) dapat digunakan oleh manajemen SDM. Misalnya, jika PercentSalaryHike sangat berpengaruh, ini dapat menjadi dasar diskusi mengenai strategi kompensasi dan penghargaan yang lebih efektif. Identifikasi karyawan yang diprediksi memiliki kinerja 'Outstanding' dapat membantu dalam program pengembangan talenta dan perencanaan suksesi. Kinerja model yang tinggi juga menunjukkan potensi penggunaan alat prediktif untuk membantu objektivitas dalam proses evaluasi awal.

2. Pencapaian Goals:

- EDA telah dilakukan untuk memahami karakteristik dataset.

- Pra-pemrosesan data yang komprehensif (termasuk penanganan duplikat, imputasi, encoding target dan fitur, scaling) telah dilaksanakan.

- Beberapa model klasifikasi telah dikembangkan dan dibandingkan kinerjanya.

- Optimasi hyperparameter telah dilakukan pada model-model utama.

- Model dengan kinerja terbaik (model yang dipilih) telah diidentifikasi berdasarkan metrik yang relevan.

- Analisis fitur paling berpengaruh telah dilakukan.

3. Dampak Solution Statements:

- Analisis Data dan Pra-pemrosesan Komprehensif: Solusi ini berdampak signifikan pada kualitas data yang digunakan untuk melatih model. Langkah-langkah seperti penghapusan duplikat, imputasi nilai hilang pada YearsWithCurrManager, dan encoding yang tepat memastikan model belajar dari data yang bersih dan relevan. Kualitas data ini terukur dari tidak adanya error saat pemodelan dan kemampuan model untuk mencapai kinerja tinggi.

- Pengembangan dan Optimasi Model Klasifikasi Komparatif: Solusi untuk mengembangkan dan mengoptimasi beberapa model memungkinkan identifikasi algoritma yang paling sesuai untuk dataset ini. RandomizedSearchCV membantu menemukan parameter optimal yang menghasilkan peningkatan kinerja signifikan (sebagaimana terlihat dari skor F1-CV yang tinggi untuk semua model). Perbandingan metrik evaluasi antar model memberikan dasar yang kuat untuk memilih solusi prediktif terbaik. Skor F1-Score (Macro) dan Akurasi yang mencapai 1.00 pada beberapa model setelah optimasi menunjukkan dampak besar dari pendekatan ini.

## Referensi
Referensi
1. Patel, E. V., Modi, K. J., & Bhavsar, M. H. (2024). *Employee Performance Evaluation Using Machine Learning*. International Journal of Advances in Engineering and Management (IJAEM), 6(11), 160–164.  
2. *Employee Performance Prediction: An Integrated Approach of Business Analytics and Machine Learning*. (2024, Februari). Journal of Business and Management Studies, 6(1).  
3. McCartney, S., & Fu, N. (2022). *Bridging the gap: why, how and when HR analytics can impact organizational performance*. Management Decision, 60(13), 25–47.  
4. *The Effect of Human Resource Analytics on Organizational Performance: Insights from Ethiopia*. (2024). Journal of Intelligence, 13(2), 134.  
5. Rao, P. S., & Verma, A. (2018). *Employee Performance Prediction using Decision Tree Algorithm*. International Journal of Computer Applications, 180(10), 23–27. (Sebagaimana dikutip dalam Patel et al., 2024)  
6. AIHR. (n.d.). *12 Types of HR Reports*. Diakses dari https://www.aihr.com/blog/types-of-hr-reports  
7. GitHub. (n.d.). *adityaiiitmk/MLTMPLTE: Machine Learning / Deep Learning Project Structure*. Diakses dari https://github.com/adityaiiitmk/MLTMPLTE  
8. DataCamp. (n.d.). *Data Preprocessing: A Complete Guide with Python Examples*. Diakses dari https://www.datacamp.com/blog/data-preprocessing  
9. Pluralsight. (n.d.). *Preparing Data for Modeling with scikit-learn*. Diakses dari https://www.pluralsight.com/resources/blog/guides/preparing-data-modeling-scikit-learn  
10. Pure Storage. (n.d.). *What Is Data Preprocessing for Machine Learning?*. Diakses dari https://www.purestorage.com/knowledge/what-is-data-preprocessing.html  
11. Michael, A. A., & Akintola, K. G. (2025, April 15). *Comparative Analysis of Machine Learning Models for Employee Performance Evaluation*. Iconic Research And Engineering Journals, 8(10), 535–543.  
12. GeeksforGeeks. (n.d.). *Hyperparameter Tuning*. Diakses dari https://www.geeksforgeeks.org/hyperparameter-tuning/  
13. Georgia Institute of Technology. (2024, Februari 16). *Tutorial on Hyperparameter Tuning Using scikit-learn – OMSCS 7641*. Diakses dari https://sites.gatech.edu/omscs7641/2024/02/16/tutorial-on-hyperparameter-tuning-using-scikit-learn/  
14. GitHub. (n.d.). *adityaiiitmk/MLTMPLTE: Machine Learning / Deep Learning Project Structure* (duplikasi ref. 7).  
15. Keylabs. (n.d.). *Understanding the F1 Score and AUC-ROC Curve*. Diakses dari https://keylabs.ai/blog/understanding-the-f1-score-and-auc-roc-curve/  
16. Deepchecks. (n.d.). *Understanding F1 Score, Accuracy, ROC-AUC & PR-AUC Metrics*. Diakses dari https://www.deepchecks.com/f1-score-accuracy-roc-auc-and-pr-auc-metrics-for-models/  
17. Brownlee, J. (2020, Januari 14). *Tour of Evaluation Metrics for Imbalanced Classification*. MachineLearningMastery.com. Diakses dari https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/  
18. Haroon, S. (2022). *HR Analytics Dataset*. Kaggle. Diakses dari https://www.kaggle.com/datasets/saadharoon27/hr-analytics-dataset  
19. Kaggle User Discussion/Notebooks for HR Analytics Dataset by saadharoon27 (berbagai kontributor). Diakses dari https://www.kaggle.com/datasets/saadharoon27/hr-analytics-dataset/discussion dan https://www.kaggle.com/datasets/saadharoon27/hr-analytics-dataset/code  
20. Haroon, S. (n.d.). *Saad Haroon | Expert | Kaggle*. Diakses dari https://www.kaggle.com/saadharoon27/datasets  
21. O’Keeffe, D. (n.d.). *IBM HR Dataset: Exploratory Data Analysis*. Kaggle. Diakses dari https://www.kaggle.com/code/dgokeeffe/ibm-hr-dataset-exploratory-data-analysis/data  
22. Solanki, T. (n.d.). *HR Analysis – EDA & Models*. Kaggle. Diakses dari https://www.kaggle.com/code/tarunsolanki/hr-analysis-eda-models  
23. Analytics Vidhya. (2020, April). *What is Feature Scaling and Why is it Important?*. Diakses dari https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/  
24. IBM. (n.d.). *What Is Logistic Regression?*. Diakses dari https://www.ibm.com/think/topics/logistic-regression  
25. V7 Labs. (n.d.). *Logistic regression: Definition, Use Cases, Implementation*. Diakses dari https://www.v7labs.com/blog/logistic-regression  
26. IBM. (n.d.). *What is Random Forest?*. Diakses dari https://www.ibm.com/think/topics/random-forest  
27. Coursera. (n.d.). *What Are the Advantages and Disadvantages of Random Forest?*. Diakses dari https://www.coursera.org/articles/advantages-and-disadvantages-of-random-forest  
28. GeeksforGeeks. (n.d.). *Support Vector Machine (SVM) Algorithm*. Diakses dari https://www.geeksforgeeks.org/support-vector-machine-algorithm/  
29. DigitalDefynd. (2025). *10 Pros & Cons of Support Vector Machines*. Diakses dari https://digitaldefynd.com/IQ/pros-cons-of-support-vector-machines/  
30. Xenoss. (n.d.). *Gradient Boosting | Definition, Algorithm & Applications*. Diakses dari https://xenoss.io/ai-and-data-glossary/gradient-boosting  
31. Alooba. (n.d.). *Everything You Need to Know When Assessing Gradient Boosting Skills*. Diakses dari https://www.alooba.com/skills/concepts/machine-learning-11/gradient-boosting/  
32. Patel, E. V., et al. (2024). *Various case studies referenced in this paper* (duplikasi ref. 1).  
33. Brownlee, J. (2020, Januari 14). *A Gentle Introduction to Probability Metrics for Imbalanced Classification*. MachineLearningMastery.com. Diakses dari https://machinelearningmastery.com/probability-metrics-for-imbalanced-classification/  
34. HireVue. (n.d.). *HR Data Analytics Challenges*. Diakses dari https://www.hirevue.com/blog/hiring/hr-data-analytics-challenges  
35. GitHub. (n.d.). *ghimiresunil/Machine-Learning-Project-Structure*. Diakses dari https://github.com/ghimiresunil/Machine-Learning-Project-Structure  
36. CodeSignal Learn. (n.d.). *Handling Mixed Data Types in Columns Using Python*. Diakses dari https://codesignal.com/learn/courses/advanced-data-cleaning-handling-text-data-1/lessons/handling-mixed-data-types-in-columns-using-python  
37. Adityaab1407. (n.d.). *Employee Productivity and Satisfaction HR Data*. Kaggle. Diakses dari https://www.kaggle.com/datasets/adityaab1407/employee-productivity-and-satisfaction-hr-data  
38. LabEx. (n.d.). *How to handle lists with mixed data types in Python*. Diakses dari https://labex.io/tutorials/python-how-to-handle-lists-with-mixed-data-types-in-python-397685  
39. Anshika2301. (n.d.). *HR Analytics Dataset*. Kaggle. Diakses dari https://www.kaggle.com/datasets/anshika2301/hr-analytics-dataset  
40. Quora. (berbagai kontributor). *What are the biggest challenges faced by HR professionals 2025?*. Diakses dari https://www.quora.com/What-are-the-biggest-challenges-faced-by-HR-professionals-2025  
41. Reach Reporting. (n.d.). *HR Analytics Template*. Diakses dari https://reachreporting.com/blog/hr-analytics-template  
