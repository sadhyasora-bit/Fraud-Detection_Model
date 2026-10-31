# Raw Dataset

Folder ini digunakan untuk menyimpan **dataset mentah (raw dataset)**
yang menjadi sumber data penelitian deteksi fraud transaksi kartu
kredit.

## Sumber Dataset

Dataset diperoleh dari Kaggle:

-   **Nama:** Credit Card Fraud Detection
-   **Penyedia:** Machine Learning Group -- Université Libre de
    Bruxelles (ULB)
-   **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
-   **File utama:** `creditcard.csv`

Dataset berisi transaksi kartu kredit dengan label: - `Class = 0`:
Non-Fraud - `Class = 1`: Fraud

## Mengapa File Dataset Tidak Disimpan di GitHub?

File dataset mentah berukuran relatif besar sehingga tidak disertakan
langsung dalam repository GitHub. Untuk menjaga repository tetap ringan
dan menghindari batas ukuran file GitHub, file dataset perlu diunduh
secara mandiri dari sumber aslinya.

File dataset tidak diubah dan tidak boleh digantikan dengan data hasil
preprocessing pada folder ini.

## Cara Mendapatkan Dataset

1.  Buka halaman [Credit Card Fraud Detection di
    Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2.  Login ke akun Kaggle.
3.  Unduh dataset melalui tombol **Download**.
4.  Ekstrak file ZIP jika hasil unduhan masih berbentuk ZIP.
5.  Letakkan file `creditcard.csv` ke dalam folder `dataset/raw/`.

Pastikan struktur folder menjadi seperti berikut:

``` text
Fraud-Detection_Model/
└── dataset/
    └── raw/
        ├── README.md
        └── creditcard.csv
```

## Catatan Penggunaan

-   Notebook EDA dan preprocessing menggunakan file
    `dataset/raw/creditcard.csv` sebagai data masukan awal.
-   Jangan mengunggah dataset hasil preprocessing atau hasil balancing
    ke folder `raw`.
-   Hasil pembersihan, pembagian data, preprocessing, dan penanganan
    class imbalance disimpan pada folder terpisah sesuai tahapan
    eksperimen.
-   Jika nama atau lokasi file berbeda, sesuaikan path pada notebook
    yang digunakan.

## Reproducibility

Agar eksperimen dapat direplikasi, gunakan dataset dari sumber Kaggle
yang sama dan dokumentasikan setiap perubahan atau tahapan pengolahan
data pada notebook terkait.
