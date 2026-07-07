# Prompt Training 5 Arsitektur — Plant Disease Patch Classification

Saya sedang membangun sistem klasifikasi kesehatan daun tanaman (healthy/unhealthy) berbasis patch-splitting, mereplikasi dan memperluas metodologi dari paper Bouacida et al. 2025 "Innovative deep learning approach for cross-crop plant disease detection". Saya ingin melatih dan membandingkan 5 arsitektur berbeda dengan preprocessing yang identik agar perbandingan hasil valid (apple-to-apple).

Device training: RTX 5050 8GB VRAM, RAM 16GB. Framework: PyTorch. Semua file ditaruh di folder `train_model_script/`.

Dataset asumsi terstruktur: `dataset_root/{healthy,unhealthy}/*.jpg` (kalau struktur asli berbeda, sesuaikan otomatis dan dokumentasikan asumsi di komentar kode).

## BAGIAN 1 — MODUL SHARED: `train_model_script/preprocessing.py`

Buat SATU modul preprocessing yang akan diimport oleh SEMUA 5 script training, supaya logic splitting dan filtering identik di semua eksperimen. Isinya:

1. Fungsi `split_into_patches(image, patch_size=32)`:
   - Resize gambar ke kelipatan 32 terdekat jika ukuran asli bukan kelipatan 32
   - Split gambar jadi grid non-overlapping patch 32×32 (untuk gambar 256×256 menghasilkan grid 8×8 = 64 patch)
   - Return list/array semua patch beserta koordinat grid-nya (i,j)

2. Fungsi `calculate_black_pixel_percentage(patch)`:
   - Hitung persentase pixel dengan RGB=(0,0,0) terhadap total pixel di patch

3. Fungsi `filter_patches(patches, black_pixel_percentages, threshold=50)`:
   - Buang patch dengan black pixel percentage = 100% (leaf piece absence)
   - Pertahankan hanya patch dengan black pixel percentage <= threshold (default 50, sesuai paper), threshold harus configurable
   - Return list patch yang lolos filter beserta koordinatnya

4. Class `PatchDataset(torch.utils.data.Dataset)`:
   - Constructor menerima path dataset root, list transform (augmentation/resize), dan black pixel threshold
   - Saat inisialisasi: iterasi semua gambar di folder healthy/unhealthy, split tiap gambar jadi patch, filter patch, assign label sesuai folder asal (healthy=0, unhealthy=1)
   - Simpan hasil patch dalam bentuk yang efisien memory (jangan load semua patch mentah ke RAM sekaligus kalau dataset besar — pertimbangkan lazy loading atau simpan index+koordinat, crop on-the-fly saat `__getitem__`)
   - `__getitem__` return (patch_tensor, label)
   - Tambahkan opsi caching ke disk (misal .npy atau .pt) supaya tidak perlu re-splitting tiap kali script dijalankan ulang

5. Fungsi `calculate_prevalence_rate(healthy_count, unhealthy_count)`:
   - Implementasi Eq. 10 dari paper: P = (unhealthy_count * 100) / (healthy_count + unhealthy_count)
   - Dipakai saat inference production (bukan saat training), untuk agregasi hasil semua patch dari satu gambar utuh

6. Fungsi `get_dataloaders(dataset_root, batch_size, val_split=0.2, transform_train=None, transform_val=None, num_workers=4)`:
   - Split dataset 80/20 train/val (stratified by label kalau memungkinkan)
   - Return DataLoader train dan val

Semua 5 script training di bawah HARUS `from preprocessing import *` atau import spesifik dari modul ini — jangan reimplementasi logic splitting/filtering secara terpisah di masing-masing script.

## BAGIAN 2 — KONVENSI UMUM (wajib konsisten di semua 5 script)

Supaya hasil 5 model bisa dibandingkan langsung dalam satu tabel, terapkan format berikut di SEMUA script:

- Struktur folder output per model: `train_model_script/outputs/{model_name}/` berisi:
  - `checkpoints/best_model.pt` (bobot terbaik berdasarkan val_loss terendah)
  - `logs/training_log.csv` dengan kolom: `epoch,train_loss,train_acc,val_loss,val_acc,lr`
  - `logs/learning_curve.png` (plot loss dan accuracy train vs val dalam 1 figure, 2 subplot)
  - `eval/metrics.json` berisi: `{"accuracy":..., "precision":..., "recall":..., "f1_score":..., "confusion_matrix":[[...]]}`
  - `eval/confusion_matrix.png`
  - `exported/model.onnx`
  - `exported/model_int8.onnx` (dynamic quantized, kecuali script 1 yang cukup ONNX fp32 saja karena jadi baseline replikasi paper)
- Semua script pakai CLI argument (argparse) dengan nama argumen SAMA di semua script: `--dataset_root`, `--epochs`, `--batch_size`, `--lr`, `--output_dir`, `--patch_black_threshold` (default 50), `--seed` (default 42 untuk reproducibility)
- Semua script set random seed (torch, numpy, random) di awal untuk hasil reproducible
- Semua script pakai mixed precision training (`torch.cuda.amp`) untuk efisiensi VRAM
- Semua script print ringkasan akhir ke terminal: total training time, best val_acc, final test metrics, ukuran file model (MB) untuk ONNX fp32 dan int8
- Semua script cetak jumlah parameter model (trainable vs total) sebelum training mulai

## BAGIAN 3 — SCRIPT 1: `train_small_inception.py` (baseline replikasi paper, training from scratch)

Implementasi arsitektur "small Inception" dari paper dari nol (tanpa pretrained weights):

- **Conv Module** = Conv2D(kernel k×k) → BatchNorm → ReLU
- **Inception Module** = input dipecah ke 2 branch paralel: Conv Module 1×1 dan Conv Module 3×3, hasil di-concat di channel axis
- **Downsample Module** = input dipecah ke 2 branch paralel: Conv Module 3×3 stride 2, dan MaxPool 3×3 stride 2, hasil di-concat

Susunan layer lengkap:
```
Input(32×32×3)
→ Conv 3×3 (96ch)
→ Inception (64ch)
→ Inception (80ch)
→ Downsample (→15×15, 160ch)
→ Inception ×4 berturut-turut (160, 160, 160, 144ch)
→ Downsample (→7×7, 240ch)
→ Inception ×2 berturut-turut (336, 336ch)
→ Global Average Pooling
→ Fully Connected (2 output)
```

Channel di atas adalah rekonstruksi dari figure paper — kalau ada shape mismatch saat implementasi, sesuaikan otomatis tapi PERTAHANKAN urutan modul (Conv → Inception → Inception → Downsample → Inception×4 → Downsample → Inception×2 → GAP → FC).

Training config:
- Training FROM SCRATCH (tanpa pretrained weights, karena arsitektur custom)
- Hyperparameter ikuti Table 3 paper: batch_size=32, learning_rate=0.001, optimizer Adam (default params), epochs=300
- Early stopping: monitor val_loss, patience 20-30 (karena device terbatas, biarkan early stopping yang menentukan durasi aktual)
- Tidak perlu resize/upsample patch — input tetap native 32×32

## BAGIAN 4 — SCRIPT 2: `train_mobilenetv3_small.py` (pretrained, edge-optimized)

- Load `torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')`
- Upsample tiap patch dari 32×32 ke 96×96 (bilinear interpolation) sebelum masuk backbone — backbone pretrained ImageNet butuh resolusi lebih besar dari 32×32
- Replace classifier head jadi `Linear(in_features, 2)`
- Strategi fine-tuning bertahap:
  - Fase 1 (5-10 epoch pertama): freeze seluruh backbone, hanya training classifier head
  - Fase 2 (sisa epoch): unfreeze semua layer, training dengan learning_rate lebih kecil (1e-4), scheduler CosineAnnealingLR atau ReduceLROnPlateau (pilih salah satu, dokumentasikan alasan di komentar)
- Total epoch 40-60, early stopping monitor val_loss patience 10
- Batch size mulai dari 64, turunkan otomatis (dengan retry logic atau minimal beri warning) kalau terjadi CUDA OOM

## BAGIAN 5 — SCRIPT 3: `train_efficientnet_b0.py` (pretrained, akurasi lebih tinggi)

- Load EfficientNet-B0 pretrained ImageNet (`torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')` atau `timm.create_model('efficientnet_b0', pretrained=True)`, pilih salah satu library dan konsisten)
- Upsample tiap patch dari 32×32 ke 128×128 (bilinear interpolation) — resolusi lebih besar dari MobileNet karena backbone lebih dalam
- Replace classifier head jadi `Linear(in_features, 2)`
- Strategi fine-tuning bertahap sama seperti script 2: freeze backbone 5-10 epoch pertama, lalu unfreeze semua dengan learning_rate kecil (1e-4 atau lebih kecil), scheduler CosineAnnealingLR
- Total epoch 40-60, early stopping monitor val_loss patience 10
- Batch size mulai dari 32 (lebih berat dari MobileNet, waspada VRAM 8GB), mixed precision WAJIB (bukan opsional) di script ini

## BAGIAN 6 — SCRIPT 4: `train_shufflenetv2.py` (pretrained, tercepat inference)

- Load `torchvision.models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')`
- Upsample tiap patch dari 32×32 ke 96×96 (bilinear interpolation)
- Replace fully connected layer terakhir jadi `Linear(in_features, 2)`
- Strategi fine-tuning: freeze backbone 5-10 epoch pertama, unfreeze semua dengan learning_rate 1e-4, scheduler ReduceLROnPlateau
- Total epoch 30-50 (biasanya konvergen lebih cepat karena model ringan), early stopping monitor val_loss patience 10
- Batch size bisa lebih besar (128) karena model sangat ringan
- TAMBAHAN KHUSUS script ini: setelah training selesai, jalankan benchmark inference speed — ukur waktu rata-rata (dalam ms) untuk memproses 1 batch berisi 64 patch (simulasi 1 gambar daun utuh diproses penuh), lakukan minimal 50 kali run dan ambil rata-rata + std deviation, simpan hasil ke `eval/inference_benchmark.json`

## BAGIAN 7 — SCRIPT 5: `train_squeezenet.py` (pretrained, footprint paling kecil)

- Load `torchvision.models.squeezenet1_1(weights='IMAGENET1K_V1')`
- Upsample tiap patch dari 32×32 ke 96×96 (bilinear interpolation)
- PENTING: classifier SqueezeNet berbentuk `Conv2d(512, 1000, kernel_size=1)` bukan `Linear` — ganti jadi `Conv2d(512, 2, kernel_size=1)`, lalu pastikan forward pass tetap benar (AdaptiveAvgPool2d diikuti flatten agar output akhir berbentuk (batch_size, 2))
- Strategi fine-tuning: freeze backbone 5-10 epoch pertama, unfreeze semua dengan learning_rate 1e-4, scheduler ReduceLROnPlateau
- Total epoch 30-50, early stopping monitor val_loss patience 10
- Batch size bisa besar (128) karena model sangat kecil (~5MB)
- TAMBAHAN KHUSUS script ini: cetak dan simpan perbandingan ukuran file model akhir (dalam MB) untuk versi ONNX fp32 vs int8 ke `eval/model_size_comparison.json`, karena tujuan utama arsitektur ini adalah footprint minimal untuk deployment.

## CATATAN PENUTUP UNTUK AGENT

Setelah kelima script selesai dibuat, buatkan juga satu script tambahan `train_model_script/compare_results.py` yang membaca `eval/metrics.json` dan `eval/model_size_comparison.json`/`eval/inference_benchmark.json` (kalau ada) dari kelima folder output model, lalu compile jadi satu tabel perbandingan (accuracy, precision, recall, F1, ukuran model ONNX fp32/int8, waktu training total, dan inference speed kalau tersedia) dalam format CSV dan tabel markdown, supaya siap langsung dipakai di bagian hasil laporan/skripsi.