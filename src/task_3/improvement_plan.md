# Rencana perbaikan Task 3

Hasil audit `src/task_3/training/` dan level 2–3. Semua langkah dikerjakan **hanya di validasi**. Test dibuka satu kali di akhir, setelah konfigurasi dibekukan. Tidak ada jaminan hasil akan naik. Tujuannya membuat evaluasi jujur, dan hasil mendekati acak tetap kesimpulan yang sah.

## Ringkasan temuan

- Validasi MCC 0.09–0.17, test MCC −0.04 sampai −0.10. Tidak ada model Level 2 atau 3 yang mengalahkan baseline di test.
- Tidak ditemukan kebocoran informasi masa depan. Gap berasal dari seleksi checkpoint pada validasi kecil dan noisy (optimisme validasi).
- Test sudah diakses berulang kali (korpus asli, Level 1, 2, 3, magnitude, weekly), jadi bukan holdout bersih.

## Langkah 0: ukur optimisme dengan permutation test

- **Apa:** jalankan prosedur yang sama (3 fold, 3 seed, seleksi epoch MCC) pada label training yang diacak per blok (sekitar 20 hari), 50–100 kali.
- **Untuk apa:** menghasilkan distribusi MCC validasi yang muncul hanya karena seleksi pada data tanpa sinyal. Kalau MCC acak sering 0.10–0.15, skor validasi 0.12–0.17 tidak berarti.
- **Di mana:** skrip baru di `src/task_3/sanity_check/`, memanggil `fit_fusion` di `training/controlled_fusion.py`. Pola fold ada di `sanity_check/r2_inner_validation.py`.
- **Biaya:** sekitar 100 kali satu run R2 (kecil, Word2Vec 300 dimensi).

## 1. Ganti seleksi epoch dari MCC ke log loss

**Masalah:** kunci `(mcc, macro_f1, -log_loss)` di `controlled_fusion.py` (sekitar baris 243) memilih puncak noise. MCC pada 158 baris melompat-lompat.

**Perubahan:**
- Kunci menjadi `-val_log_loss` saja (kontinu, tidak bergantung threshold).
- Haluskan: rata-rata bergerak 5 epoch dari `val_log_loss`, lalu ambil minimum. Tambah `min_delta` sekitar 1e-3.
- `max_epochs` 200 → sekitar 60, `patience` 25 → sekitar 10. History R2 menunjukkan epoch terbaik biasanya di bawah 60.
- Refit tetap memakai median epoch per seed. Catat bahwa data refit (930) lebih besar dari fold (453–772). Epoch tidak diskalakan karena tidak ada dasarnya.

**Verifikasi:** bandingkan kurva `val_log_loss` terhadap epoch dengan log loss prior (0.676 untuk target harian). Kalau tidak pernah di bawah prior, model tidak belajar apa pun.

## 2. Kecilkan input teks dan perkuat regularisasi

**Catatan:** padding nol pada slot 768 tidak menambah kapasitas efektif (input nol tidak punya gradien). Yang besar adalah input nyata: R2 punya 14 + 300 + 2 = 316 fitur untuk 930 sampel. FinBERT dan RoBERTa punya 784. Rasio fitur terhadap sampel terlalu tinggi.

**Pilihan, urut dari yang paling murah:**
- **A. Regularisasi:** `alpha` 0.001 → {0.01, 0.1, 1}, dropout 0.2–0.5, hidden layer `[16]` atau satu layer.
- **B. Reduksi dimensi:** PCA 16–32 komponen, di-fit hanya pada rata-rata artikel training fold, lalu diterapkan ke validasi (harus di dalam fold).
- **C. Sentimen skalar FinBERT:** `training/finbert.py` sudah memuat `AutoModelForSequenceClassification` tapi hanya memakai `base_model`. Ambil logit kepala klasifikasi, softmax 3 kelas, agregasi per hari (mean, min, max). Hasilnya sekitar 9 fitur menggantikan 768.

**Pembanding wajib:** regresi logistik L2 (scikit-learn) pada fitur yang sama. Kalau MLP tidak mengalahkannya, kompleksitas MLP tidak perlu dipertahankan.

**Pemilihan:** grid kecil, kriteria `val_log_loss` rata-rata 3 fold. Jangan dituning lebih jauh karena fold hanya 158 baris.

## 3. Periksa dan perbaiki skala fitur teks

**Hipotesis (belum terbukti):** `training/fusion.py` memakai `StandardScaler(with_mean=False)` untuk teks, sehingga offset rata-rata embedding tetap ada dan variasi antar hari kecil dibanding offset itu.

**Diagnostik dulu (tanpa melatih model):**
- Rata-rata kosinus antar vektor teks harian pada training. Di atas 0.95 berarti vektor harian hampir identik.
- Proporsi varians komponen utama pertama, dan rasio varians antar hari vs di dalam satu hari.
- Kalau variasi antar hari sangat kecil, itu penjelasan mengapa teks tidak memberi sinyal, dan layak ditulis di laporan.

**Perbaikan kalau diagnostik mendukung:**
- `with_mean=True` (dihitung pada baris yang punya berita). Baris tanpa berita harus tetap nol karena `BlockScaler.transform` menolak nilai non-nol di sana. Hampir semua hari punya berita (99.9%), jadi dampaknya kecil.
- Fitur perubahan: vektor hari ini dikurangi rata-rata bergerak 20 hari.

## 4. Hitung metadata dari data

- Ganti `fit_rows: 930` dan `test_rows: 235` di `training/testing.py` dan `training/representation_testing.py` dengan `len(...)` dari data yang dipakai.
- Turunkan `test_used_for_fitting_or_stopping` dari catatan fit (setiap fit di test punya `evaluation is None` dan `fixed_epochs` terisi), bukan konstanta `False`.
- Gabungkan dua versi `test_context` dan `save_test_results` (Level 1 dan Level 2/3) menjadi satu helper.

## 5. Protokol pelaporan

- Tentukan kriteria sukses **sebelum** melihat hasil. Contoh: log loss validasi lebih baik dari prior di minimal 2 dari 3 fold, dan CI block bootstrap untuk selisihnya tidak mencakup nol.
- Hanya satu konfigurasi akhir yang dibuka di test, dibekukan dengan hash seperti sekarang.
- Catat jumlah evaluasi test sebelumnya (sekitar 15 perbandingan). Sebutkan koreksi pengujian berganda (misalnya α/15) atau laporkan sebagai eksploratif.
- Laporkan baseline di tabel yang sama: prior, last direction, dan model price-only dengan prosedur identik, agar kontribusi teks bisa diisolasi.

## Masalah kecil lain (dari audit)

- `MarketAlignedMLP._init_coef` (`fusion.py`) hanya aktif untuk `fan_in == 14 + 768 + 2`. Level 3 memakai jalur torch di `controlled_fusion.py` dengan inisialisasi terpisah, jadi perbandingan antar level tidak sepenuhnya setara.
- Tidak ada class weighting dan threshold tetap 0.5, padahal model condong memprediksi satu kelas.
- Tiga seed tanpa ensemble. Metrik dirata-rata per seed.

## Urutan kerja

1. Langkah 0 dan diagnostik poin 3 (murah, menentukan apakah ada yang perlu diperbaiki).
2. Poin 1, lalu poin 2A, diuji pada R2 saja di tiga fold validasi.
3. Kalau ada perbaikan nyata di validasi, lanjut poin 2B/2C dan 3.
4. Poin 4 dan 5 dikerjakan paralel, tidak bergantung hasil.
