# 🧠 Brain Tumor Classification – Hybrid Deep CNN + XGBoost (Flask GUI)

Aplikasi **Brain Tumor Classification** ini merupakan sistem berbasis **Flask Web GUI** yang dirancang untuk melakukan **deteksi dan klasifikasi tumor otak** secara otomatis menggunakan citra MRI.  
Tujuan utama aplikasi ini adalah memberikan alat bantu diagnosis yang mudah digunakan dan dapat diakses tanpa keahlian teknis, dengan hasil prediksi yang cepat, akurat, dan informatif.

---

## 🖥️ Fitur Utama GUI

- 🌐 **Antarmuka berbasis web (Flask)** — Dapat dijalankan secara lokal di browser tanpa konfigurasi tambahan.  
- 📤 **Upload Citra MRI** — Pengguna dapat mengunggah gambar MRI otak dengan format `.jpg`, `.jpeg`, atau `.png`.  
- ⚙️ **Prediksi Otomatis** — Aplikasi memproses gambar, mengekstraksi fitur dengan dua CNN, dan menampilkan hasil klasifikasi.  
- 📊 **Visualisasi Hasil** — Menampilkan label tumor (Glioma, Meningioma, Pituitary, No Tumor) beserta probabilitas (%) setiap kelas.  
- 🔁 **Sistem Real-time Response** — Hasil muncul secara instan setelah proses inferensi selesai.

---

## 🧠 Arsitektur Model

- **EfficientNetV2-M** – Ekstraktor fitur utama dengan efisiensi tinggi dan performa kuat pada citra medis.  
- **InceptionV3** – Melengkapi ekstraksi fitur dengan memperhatikan representasi spasial kompleks.  
- **XGBoost Classifier** – Digunakan sebagai lapisan klasifikasi akhir untuk meningkatkan akurasi dan stabilitas hasil.  

Gabungan model ini menghasilkan pendekatan **Hybrid Dual Deep CNN + XGBoost**, yang terbukti meningkatkan sensitivitas terhadap tipe tumor minor dan memberikan generalisasi yang lebih baik pada citra MRI.

---
