# 📚 Klasifikasi Buah dengan Naive Bayes

## Import Library yang dibutuhkan
```python
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

```
### ✅Fungsi
Import semua library yang dibutuhkan:

- pandas dan numpy → manipulasi data.

- LabelEncoder → konversi label kategori ke angka.

- train_test_split → membagi data latih dan uji.

- StandardScaler → standarisasi fitur numerik.

- GaussianNB → membuat model Naive Bayes.

- confusion_matrix, classification_report, accuracy_score → evaluasi model.
  
# 📥 Membaca Dataset
```python
dataset = pd.read_excel('fruit.xlsx')
dataset.head()

```
### ✅Fungsi
- Membaca file Excel fruit.xlsx dan menampilkan 5 baris pertama untuk memastikan data berhasil dibaca.

# 📊Info Dataset
```python
dataset.info()
```
### ✅Fungsi
Menampilkan :
- Jumlah total baris (entries).

- Nama kolom.

- Jumlah data non-null di tiap kolom.

- Tipe data setiap kolom (misalnya int64, float64, object/string, dsb).


# 🔧 Preprocessing Data
```python
en = LabelEncoder()

dataset['name'] = en.fit_transform(dataset['name'])
dataset.head()
```
### ✅Fungsi
Kolom name berisi jenis buah dikonversi ke angka agar bisa diproses oleh model Machine Learning.

Contoh :

apple → 0

banana → 1

orange → 2

# Pisahkan fitur dan label
```python
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```
### ✅Fungsi
- x berisi semua fitur (diameter, berat, warna).

- y berisi label buah.


# 📊 Split Data menjadi Latih dan Uji
```python
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=123)

print("x_train = ", len(x_train))
print("x_test = ", len(x_test))
print("y_train = ", len(y_train))
print("y_test = ", len(y_test))
```
### ✅ Fungsi
Digunakan untuk membagi dataset menjadi data training dan data testing.

Parameter :

- x : Fitur (input) — biasanya DataFrame tanpa label target.

- y : Label (output/target).

- test_size = 0.2 : 20% data digunakan untuk testing, 80% untuk training.

- random_state = 123 : Agar hasil split selalu sama (reproducible).

Hasil Split:

- x_train : Fitur untuk training.

- x_test : Fitur untuk testing.

- y_train : Label untuk training.

- y_test : Label untuk testing.

# ⚖️ Standarisasi Fitur
```python
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```
### ✅ Fungsi
- Standarisasi fitur agar memiliki skala yang sama, mencegah bias terhadap fitur dengan nilai besar.

# 🧠 Membuat Model Naive Bayes
```python
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```
✅ Fungsi
- Inisialisasi model Gaussian Naive Bayes.

- Latih model menggunakan data latih.
  
# 🔮 Prediksi Data Uji
```python
classifier = GaussianNB()
classifier.fit(x_train, y_train)
```
### ✅ Fungsi
- Prediksi jenis buah dari data uji.

- Hitung matriks kebingungan (benar/salah prediksi).

- Lihat precision, recall, f1-score tiap kelas.

- Hitung akurasi keseluruhan.


# 📈 Evaluasi Model
```python
ydata = pd.DataFrame()
ydata['y_test'] = pd.DataFrame(y_test)
ydata['y_pred'] = pd.DataFrame(y_pred)
ydata
```
### ✅ Fungsi

- Prediksi jenis buah dari data uji.

- Hitung matriks kebingungan (benar/salah prediksi).

- Lihat precision, recall, f1-score tiap kelas.

- Hitung akurasi keseluruhan.
  
# 💾 Simpan Hasil ke Excel
```python
ydata.to_excel('dataactualpred.xlsx', index=False)
```
### ✅ Fungsi:
- Membuat file Excel hasil_prediksi.xlsx berisi perbandingan label asli vs hasil prediksi.


