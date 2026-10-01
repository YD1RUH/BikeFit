# BikeFit

Aplikasi desktop untuk menganalisis posisi pesepeda (*bike fitting*) dari **video atau webcam**, menggunakan estimasi pose berbasis kecerdasan buatan. Aplikasi menampilkan **parameter yang perlu diperbaiki, rekomendasi penyesuaian, dan skor 0-100** dari tiga sudut pandang: samping, depan, dan belakang.

<img src="https://github.com/YD1RUH/BikeFit/blob/main/sample.gif?raw=true" alt="Example using Video" width="100%"/>

Dibangun dengan **Python 3**, **DearPyGui**, **OpenCV**, dan **MediaPipe Pose Landmarker**.

> **Penting:** aplikasi ini adalah alat bantu estimasi berbasis kamera 2D, bukan alat medis dan bukan pengganti bike fitter atau fisioterapis profesional. Lihat bagian [Dasar Ilmiah dan Keterbatasan](#dasar-ilmiah-dan-keterbatasan).

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Fitur](#fitur)
- [Cara Kerja](#cara-kerja)
  - [Alur Pemrosesan](#alur-pemrosesan)
  - [Parameter yang Dinilai](#parameter-yang-dinilai)
  - [Sistem Skor](#sistem-skor)
  - [Loop Video Otomatis](#loop-video-otomatis)
- [Dasar Ilmiah dan Keterbatasan](#dasar-ilmiah-dan-keterbatasan)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Kustomisasi](#kustomisasi)
- [Struktur Proyek](#struktur-proyek)
- [Pemecahan Masalah](#pemecahan-masalah)
- [Rencana Pengembangan](#rencana-pengembangan)
- [Disclaimer](#disclaimer)
- [Acuan](#acuan)
- [Lisensi](#lisensi)

---

## Latar Belakang

Posisi tubuh di atas sepeda memengaruhi kenyamanan, efisiensi mengayuh, dan risiko cedera akibat penggunaan berlebih (*overuse*), terutama pada lutut. Penyesuaian posisi (*bike fitting*) umumnya dilakukan oleh fitter profesional dengan goniometer atau sistem *motion capture*, yang biayanya tidak murah dan tidak selalu mudah diakses.

Proyek ini mencoba menjembatani celah tersebut. Dengan satu kamera (webcam atau video ponsel), aplikasi mengukur sudut-sudut sendi utama saat pesepeda mengayuh, membandingkannya dengan rentang acuan, lalu memberi umpan balik yang mudah dipahami: bagian mana yang menyimpang, apa yang disarankan diubah, dan seberapa baik posisi secara keseluruhan.

Tujuan proyek:

- Memberi **gambaran awal** posisi di sepeda secara cepat dan murah.
- Membantu pesepeda **melacak perubahan** setelah menggeser sadel, stem, atau cleat.
- Menjadi dasar eksperimen untuk pengembangan *bike fitting* berbasis *computer vision*.

Proyek ini **tidak** dimaksudkan untuk mendiagnosis cedera atau menggantikan penilaian profesional.

---

## Fitur

- Input dari **webcam** (pilih indeks kamera) atau **file video** (`.mp4`, `.avi`, `.mov`, `.mkv`).
- Tiga mode sudut pandang: **Samping**, **Depan**, dan **Belakang**.
- Tiga tipe sepeda dengan rentang acuan berbeda: **Road**, **MTB**, **City/Hybrid**.
- Overlay kerangka dan nilai sudut langsung di atas video.
- Tabel hasil per parameter: nilai terukur, rentang ideal, skor, dan status (**OK / Perhatikan / Perbaiki**).
- Daftar **titik yang perlu diperbaiki** beserta **rekomendasi** konkret, diurutkan dari skor terendah.
- **Skor total 0-100**.
- **Loop video otomatis** sampai tidak ada parameter yang tertahan di status "Perhatikan" (dengan batas putaran).
- Ekspor laporan ke file teks.

---

## Cara Kerja

### Alur Pemrosesan

```
Video / Webcam
      │
      ▼
OpenCV: baca frame ──► MediaPipe Pose Landmarker (33 titik tubuh)
                                  │
                                  ▼
                 Ambil titik bahu, siku, pergelangan tangan,
                 pinggul, lutut, pergelangan kaki
                                  │
                                  ▼
              Hitung sudut per frame (geometri 2D)
                                  │
                                  ▼
      Simpan ke jendela 150 frame terakhir ─► rangkum (persentil / median / rentang)
                                  │
                                  ▼
          Bandingkan dengan rentang ideal ─► skor, status, rekomendasi
                                  │
                                  ▼
                        Tampil di GUI DearPyGui
```

### Parameter yang Dinilai

**Mode Samping** (sudut dihitung pada bidang sagital; pilih sisi tubuh yang menghadap kamera):

| Parameter | Definisi | Ideal (Road) | Dasar |
|---|---|---|---|
| Sudut lutut (BDC) | Sudut pinggul-lutut-pergelangan kaki; persentil ke-95 (lutut paling lurus) | 140-150° | Berbasis riset, disesuaikan untuk pengukuran dinamis [1][2] |
| Sudut pinggul (tutup) | Sudut bahu-pinggul-lutut; persentil ke-5 (paling tertutup) | 45-65° | Heuristik praktik bike fitting |
| Sudut punggung | Sudut garis pinggul-bahu terhadap horizontal; median | 35-50° | Heuristik; riset menunjukkan trade-off [5] |
| Sudut siku | Sudut bahu-siku-pergelangan tangan; median | 150-170° | Sebagian selaras dengan data fleksi siku rata-rata [6] |
| Sudut bahu | Sudut pinggul-bahu-siku; median | 75-95° | Heuristik |

**Mode Depan dan Belakang** (bidang frontal):

| Parameter | Definisi | Ideal | Mode |
|---|---|---|---|
| Lutut kiri / kanan (*tracking*) | Simpangan lutut dari garis pinggul-pergelangan kaki (+ = ke dalam/valgus, - = ke luar/varus) | -6° s/d +6° | Depan, Belakang |
| Goyangan panggul | Rentang (persentil 95 - 5) kemiringan garis pinggul | ≤ 5° | Depan, Belakang |
| Goyangan torso | Rentang kemiringan lateral garis pinggul ke bahu | ≤ 6° | Depan, Belakang |
| Kemiringan bahu | Kemiringan garis bahu terhadap horizontal (nilai mutlak median) | ≤ 4° | Depan, Belakang |
| Lebar setang / bahu | Jarak pergelangan tangan dibagi jarak bahu; median | 0,90-1,20 | Depan |

Rentang untuk MTB dan City/Hybrid ada di dalam kode (`SIDE_METRICS`).

### Sistem Skor

Skor tiap parameter bernilai 100 jika berada dalam rentang ideal, lalu turun secara linear sampai 0 pada penyimpangan sebesar toleransi:

```
penyimpangan = max(ideal_min - nilai, nilai - ideal_max, 0)
skor         = max(0, 100 × (1 - penyimpangan / toleransi))
```

Skor total adalah rata-rata berbobot dari skor tiap parameter.

| Mode | Bobot parameter |
|---|---|
| Samping | Lutut 30%, pinggul 20%, punggung 20%, siku 15%, bahu 15% |
| Depan | Lutut kiri 20%, lutut kanan 20%, panggul 20%, torso 15%, bahu 10%, setang 15% |
| Belakang | Dinormalisasi dari bobot mode Depan tanpa parameter setang |

Toleransi: 15° (mode Samping), 10° (jalur lutut, goyangan panggul dan torso), 8° (kemiringan bahu), 0,25 (rasio setang).

Status per parameter dan status keseluruhan:

| Skor | Status |
|---|---|
| ≥ 85 | **OK** |
| 60 - < 85 | **Perhatikan** |
| < 60 | **Perbaiki** |

Bobot, toleransi, dan batas status adalah rancangan proyek ini dan belum divalidasi secara klinis.

### Loop Video Otomatis

Untuk file video, aplikasi dapat memutar ulang video secara otomatis:

1. Setiap 10 frame, hasil dievaluasi.
2. Hasil dianggap **pasti** jika 3 evaluasi berturut-turut tidak memiliki parameter berstatus "Perhatikan" (dan kesimpulan total juga bukan "Perhatikan").
3. Jika video selesai dan hasil belum pasti, video diulang dari awal, sampai batas **Maks putaran**.
4. Jika nilai semua parameter **konsisten antar putaran** (selisih sekitar 1°) selama 2 putaran, loop berhenti lebih awal. Artinya parameter yang masih "Perhatikan" memang temuan nyata, bukan *noise* data, dan perlu disesuaikan sesuai rekomendasi.

Pengulangan video hanya membantu jika masalahnya data yang kurang atau *noise*. Jika nilai sebenarnya memang di luar rentang ideal, statusnya tidak akan berubah.

---

## Dasar Ilmiah dan Keterbatasan

Tingkat dukungan bukti **berbeda antar parameter**:

- **Sudut lutut: dasar paling kuat.** Acuan klasiknya adalah Holmes dkk. (1994), yang mengusulkan fleksi lutut 25-35° (setara 145-155° ekstensi) saat pedal di titik terendah (BDC), diukur secara statis dengan goniometer [1]. Aplikasi ini mengukur lutut secara **dinamis** saat mengayuh, dan literatur melaporkan bahwa nilai dinamis berbeda dari nilai statis sekitar 8° pada lutut [3][4]. Karena itu rentang Road diset ke 140-150° (fleksi 30-40°). Rentang MTB dan City/Hybrid yang lebih longgar adalah penyesuaian proyek ini, bukan hasil studi.
- **Bukti pencegahan cedera dari tinggi sadel masih terbatas.** Tinjauan sistematis menyimpulkan metode pengukuran dinamis lebih didukung daripada rumus tunggal berbasis inseam, tetapi hubungan perubahan kecil tinggi sadel dengan pencegahan cedera belum kuat [2].
- **Pinggul, punggung, siku, bahu: bukti lemah.** Sendi-sendi ini belum diteliti secara memadai, sehingga data posisi optimal tubuh bagian atas dan panggul masih sedikit [6]. Rentang yang dipakai adalah aturan praktis dari praktik bike fitting. Untuk sudut punggung, riset lebih menunjukkan *trade-off* (punggung horizontal penuh tidak menguntungkan, dan kontribusi sendi berubah saat torso direndahkan) daripada satu angka ideal [5].
- **Mode depan dan belakang: heuristik.** Ambang jalur lutut, goyangan panggul dan torso, kemiringan bahu, serta rasio setang berdasarkan prinsip umum bike fitting, bukan nilai dari studi tervalidasi.

Keterbatasan pengukuran:

- Titik MediaPipe bukan titik anatomi goniometer (kondilus lateral, trokanter mayor, malleolus lateral). Akurasi terhadap goniometer atau *motion capture* 3D **belum diuji** di proyek ini.
- Kamera 2D rentan terhadap distorsi perspektif dan sudut kamera yang tidak tegak lurus.
- Aplikasi tidak melacak sudut engkol, sehingga "BDC" didekati dengan persentil ke-95 sudut lutut selama mengayuh.
- Rentang sudut statis dan dinamis tidak bisa dipertukarkan begitu saja [3][4].

Hasil sebaiknya dibaca sebagai **tren dan arah penyesuaian**, bukan angka absolut.

---

## Instalasi

### Prasyarat

- Python **3.9 atau lebih baru**
- Webcam (opsional, bila ingin analisis langsung)
- Koneksi internet pada saat pertama kali dijalankan (mengunduh model pose ±9 MB)

### Langkah

1. Salin (*clone*) repositori dan masuk ke foldernya:

   ```bash
   git clone https://github.com/YD1RUH/BikeFit.git
   cd BikeFit
   ```

2. (Disarankan) Buat *virtual environment*:

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate
   ```

3. Install dependensi:

   ```bash
   pip install --upgrade dearpygui opencv-python mediapipe numpy
   ```

   Atau menggunakan `requirements.txt`, dengan  menjalankan `pip install -r requirements.txt`:

4. Jalankan aplikasi:

   ```bash
   python bikefit.py
   ```

   Pada Windows, jika perintah `python` tidak dikenali, gunakan `py bikefit.py`.

Saat pertama dijalankan, file model `pose_landmarker_full.task` diunduh otomatis ke folder yang sama dengan `bikefit.py`. Jika unduhan gagal, unduh manual dari:

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

lalu letakkan di folder yang sama dengan `bikefit.py`.

---

## Penggunaan

1. Pilih **Sudut pandang** (Samping / Depan / Belakang) dan **Tipe sepeda**.
2. Pada mode Samping, pilih **sisi tubuh yang menghadap kamera** (Kanan/Kiri).
3. Klik **Webcam** (atur indeks kamera bila perlu) atau **Buka Video**.
4. Minta pesepeda mengayuh dengan stabil minimal 5-10 detik.
5. Baca skor, tabel parameter, dan rekomendasi di panel kanan.
6. (Opsional) Klik **Simpan Laporan (.txt)**.

Kontrol lain: **Pause/Lanjut**, **Stop**, **Reset Data**, serta opsi **Loop video** dan **Maks putaran**.

### Panduan penempatan kamera

| Sudut pandang | Posisi kamera |
|---|---|
| Samping | Tegak lurus dari samping, setinggi pinggul. Seluruh tubuh dan sepeda terlihat. Sepeda sebaiknya di *trainer* atau kayuhan stabil. |
| Depan | Di depan sepeda, setinggi pinggul, lurus (tidak miring). Tangan, bahu, pinggul, lutut, dan kaki terlihat. |
| Belakang | Di belakang sepeda, setinggi pinggul, lurus. Bahu, pinggul, lutut, dan kaki terlihat. |

Tips umum: jarak kamera sekitar 2-3 m, pencahayaan cukup, latar tidak ramai, pakaian tidak terlalu longgar, dan kamera tidak miring (kemiringan kamera akan terbaca sebagai bahu atau panggul tidak rata). Pada mode depan dan belakang, "kiri" dan "kanan" berarti sisi tubuh pesepeda, bukan sisi layar.

### Prinsip penyesuaian

Ubah **satu variabel per kali**, dengan langkah kecil (misalnya 2-5 mm untuk tinggi sadel), lalu ukur ulang. Hentikan dan konsultasikan ke profesional bila ada nyeri yang menetap.

---

## Kustomisasi

Semua rentang ideal, bobot, toleransi, dan teks rekomendasi ada di bagian atas `bikefit.py`:

- `SIDE_METRICS`: parameter mode samping (memakai fungsi bantu `M(...)`).
- `FRONT_COMMON` dan `BAR_METRIC`: parameter mode depan dan belakang.
- `METRICS_BY_VIEW`: pemetaan parameter ke sudut pandang.
- `VIEW_TIPS`: teks panduan kamera.

Contoh mengubah rentang lutut Road ke acuan statis Holmes (145-155°):

```python
M("knee", "Sudut lutut (BDC)", 0.30,
  {"Road": (145, 155), "MTB": (138, 150), "City/Hybrid": (135, 150)},
  ...
```

Ambang status (85 dan 60) ada di fungsi `status_of`, dan syarat konvergensi loop di `check_converged`.

---

## Struktur Proyek

```
.
├── bikefit.py                   # Aplikasi utama (GUI, pose, skor, rekomendasi)
├── pose_landmarker_full.task    # Model MediaPipe (diunduh otomatis)
├── laporan_bikefit_<mode>.txt   # Laporan hasil ekspor (dibuat saat disimpan)
├── requirements.txt             # (opsional) daftar dependensi
└── README.md
```

---

## Pemecahan Masalah

| Masalah | Penyebab dan solusi |
|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | MediaPipe versi baru menghapus API lama `mp.solutions`. Versi kode ini sudah memakai API Tasks (`PoseLandmarker`). Pastikan memakai `bikefit.py` terbaru dan `pip install --upgrade mediapipe`. |
| Gagal mengunduh model | Periksa koneksi internet, atau unduh manual dari URL pada bagian [Instalasi](#instalasi) dan taruh di folder yang sama dengan `bikefit.py`. |
| "Pose tidak terdeteksi" | Pastikan seluruh tubuh terlihat, cahaya cukup, dan kontras pakaian terhadap latar baik. |
| "Titik tubuh kurang jelas / sisi salah?" | Pada mode Samping, pilih sisi (Kanan/Kiri) yang benar-benar menghadap kamera. |
| Webcam tidak terbuka | Ubah indeks kamera (0, 1, 2, ...), tutup aplikasi lain yang memakai kamera, dan periksa izin kamera di sistem operasi. |
| Status terus "Perhatikan" | Bisa berarti nilai memang di luar rentang ideal (temuan nyata). Ikuti rekomendasi, ubah satu variabel, lalu rekam ulang. |
| Angka sudut tidak stabil | Gunakan kamera diam (tripod), kayuhan stabil, dan durasi rekaman lebih panjang. |

---

## Rencana Pengembangan

- Pilihan acuan statis (Holmes) versus dinamis pada sudut lutut.
- Label "berbasis riset" dan "heuristik" langsung di tabel hasil.
- Kolom referensi pada laporan `.txt`.
- Fitur kalibrasi dengan goniometer untuk mengukur selisih terhadap hasil kamera.
- Deteksi otomatis siklus kayuhan (pengganti pendekatan persentil).
- Penambahan sudut pergelangan kaki.
- Penggabungan hasil tiga sudut pandang menjadi satu laporan dan satu skor.
- Ekspor laporan ke PDF.

Kontribusi berupa *issue* dan *pull request* sangat diterima, terutama data validasi terhadap goniometer atau *motion capture*.

---

## Disclaimer

Aplikasi ini disediakan "apa adanya" untuk tujuan edukasi dan informasi. Hasilnya adalah estimasi dari kamera 2D, bukan diagnosis atau saran medis. Untuk fitting yang akurat atau jika Anda mengalami nyeri, konsultasikan dengan bike fitter bersertifikat, dokter olahraga, atau fisioterapis.

---

## Acuan

Sumber yang dirujuk dalam dokumen ini. Tidak semua detail sitasi (misalnya penulis) tersedia lengkap pada sumber yang dibaca, dan pembaca dianjurkan memverifikasi ke DOI atau halaman aslinya.

### Literatur bike fitting

1. Holmes JC, Pruitt AL, Whalen NJ. Lower extremity overuse in bicycling. *Clinics in Sports Medicine*. 1994;13(1):187-203. doi:[10.1016/S0278-5919(20)30363-X](https://doi.org/10.1016/S0278-5919(20)30363-X)
2. Bini R, Priego-Quesada J. Methods to determine saddle height in cycling and implications of changes in saddle height in performance and injury risk: a systematic review. *Journal of Sports Sciences*. 2022;40(4):386-400. doi:[10.1080/02640414.2021.1994727](https://doi.org/10.1080/02640414.2021.1994727). PMID: [34706617](https://pubmed.ncbi.nlm.nih.gov/34706617/)
3. Details our eyes cannot see: challenges for the analysis of body position during bicycle fitting. *Sports Biomechanics*. doi:[10.1080/14763141.2021.1987509](https://doi.org/10.1080/14763141.2021.1987509) (memuat ringkasan perbedaan sudut statis vs dinamis dari Bini & Hume, 2016).
4. Anthropometrics, flexibility and training history as determinants for bicycle configuration. PMC9219349. https://pmc.ncbi.nlm.nih.gov/articles/PMC9219349
5. Cycling position optimisation: a systematic review of the impact of positional changes on biomechanical and physiological factors in cycling. *Journal of Sports Sciences*. doi:[10.1080/02640414.2024.2394752](https://doi.org/10.1080/02640414.2024.2394752)
6. Biomechanical and postural evaluation of optimal bike fit for non-traumatic injury prevention among cyclists: a narrative review. https://www.researchgate.net/publication/381495046

Rujukan pelengkap yang dikutip oleh sumber-sumber di atas dan layak dibaca langsung:

- Bini RR, Hume PA, Croft JL. Effects of bicycle saddle height on knee injury risk and cycling performance. *Sports Medicine*. 2011;41(6):463-476.
- Millour G, Duc S, Puel F, Bertucci W. Comparison of static and dynamic methods based on knee kinematics to determine optimal saddle height in cycling. *Acta of Bioengineering and Biomechanics*. 2019;21:93-99.
- Bini R, Hume P. A comparison of static and dynamic measures of lower limb joint angles in cycling: application to bicycle fitting. 2016.

### Perangkat lunak

1. Bazarevsky V, Grishchenko I, Raveendran K, Zhu T, Zhang F, Grundmann M. BlazePose: on-device real-time body pose tracking. arXiv:[2006.10204](https://arxiv.org/abs/2006.10204). 2020.
2. Google. MediaPipe Pose Landmarker. https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
3. Bradski G. The OpenCV Library. *Dr. Dobb's Journal of Software Tools*. 2000. https://opencv.org
4. Dear PyGui. https://github.com/hoffstadt/DearPyGui

---

## Lisensi
GPL-3.0 license

