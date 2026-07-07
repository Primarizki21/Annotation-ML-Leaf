# Prompt Perbaikan — Bottleneck PatchDataset (I/O-bound, GPU underutilized)

Saya menemukan bottleneck performa di training pipeline saya. GPU utilization naik-turun tajam (tidak pernah stabil di utilization tinggi), dan mengubah batch size (32 vs 512) hampir tidak mengubah waktu training per epoch — ini indikasi kuat bottleneck ada di data loading (CPU/I/O-bound), bukan di GPU compute. Root cause yang sudah teridentifikasi: `PatchDataset.__getitem__` memanggil `self.df.iloc[idx]` di setiap pemanggilan, dan `pandas.DataFrame.iloc` per-row access punya overhead signifikan saat dipanggil jutaan kali (dataset saya berisi ratusan ribu hingga jutaan patch).

PENTING: Perbaikan ini HARUS murni optimasi I/O/CPU — jangan mengubah urutan data, label mapping, transform/augmentasi, atau split train/val/test yang sudah ada. Tujuannya percepat loading tanpa mengubah hasil akhir training sedikit pun (data yang di-load ke model harus identik dengan sebelumnya, cuma caranya diakses yang dioptimasi).

## Konteks file yang perlu diperbaiki

Class `PatchDataset` saat ini:

```python
class PatchDataset(Dataset):
    def __init__(self, csv_path, patches_root, split, transform=None):
        self.patches_root = Path(patches_root)
        df = pd.read_csv(csv_path)
        if "split" not in df.columns:
            raise ValueError(f"{csv_path} has no 'split' column")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows with split='{split}' in {csv_path}")
        sample = self.patches_root / self.df.iloc[0]["patch_path"]
        if not sample.exists():
            raise FileNotFoundError(f"Sample patch not found: {sample}")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.patches_root / row["patch_path"]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])
```

Dan `get_dataloaders` yang memanggilnya:

```python
def get_dataloaders(csv_path, patches_root, batch_size, up_sample_size=128,
                     num_workers=4, seed=42, pin_memory=True):
    train_ds = PatchDataset(csv_path, patches_root, split="train",
                             transform=get_transforms(up_sample_size, train=True))
    val_ds = PatchDataset(csv_path, patches_root, split="val",
                           transform=get_transforms(up_sample_size, train=False))
    test_ds = PatchDataset(csv_path, patches_root, split="test",
                            transform=get_transforms(up_sample_size, train=False))
    loader_kw = dict(num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               drop_last=True, generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader
```

## Perbaikan yang diminta

### 1. Hilangkan overhead pandas per-item di `PatchDataset`

Ganti akses `self.df.iloc[idx]` dengan precomputed numpy array yang disiapkan sekali di `__init__`, bukan di-generate ulang tiap `__getitem__` dipanggil:

- Di `__init__`, setelah filter dataframe berdasarkan split, ekstrak kolom `patch_path` (gabungkan dengan `patches_root` jadi full path string) dan kolom `label` menjadi numpy array (`self.paths` dan `self.labels`), simpan sebagai atribut instance
- Di `__getitem__`, akses langsung `self.paths[idx]` dan `self.labels[idx]` — jangan sentuh `self.df` sama sekali lagi di method ini
- Pastikan validasi file exists (sample check) tetap dipertahankan seperti kode asli, tapi dijalankan sebelum konversi ke numpy array
- Pastikan urutan data dan isi tiap baris (path + label) identik 100% dengan versi sebelumnya — cukup ubah cara akses, bukan isi datanya

### 2. Tambahkan `prefetch_factor` di DataLoader

Update `loader_kw` di `get_dataloaders` untuk menambahkan `prefetch_factor=4` ketika `num_workers > 0` (biarkan `None`/default kalau `num_workers=0`, karena `prefetch_factor` tidak valid untuk num_workers=0). Ini memberi worker buffer lebih banyak batch di depan, mengurangi kemungkinan GPU menunggu.

### 3. Buat utility benchmark terpisah untuk verifikasi perbaikan

Buat script kecil `benchmark_dataloader.py` yang:
- Menerima argumen path CSV, patches_root, batch_size, num_workers (via argparse)
- Menjalankan iterasi DataLoader selama minimal 100 batch TANPA forward/backward pass ke model (murni ukur kecepatan data loading saja)
- Mencetak throughput dalam samples/detik
- Bisa dijalankan sebelum dan sesudah perbaikan untuk membandingkan angka secara langsung (before vs after)

### 4. (Opsional, kalau setelah fix #1 masih terasa lambat) Tambahkan mode caching in-memory untuk path dan label

Jika dataset (jumlah baris CSV) tidak terlalu besar untuk muat di RAM 16GB sebagai daftar path saja (bukan gambar aktual), pastikan struktur numpy array dari langkah #1 sudah cukup — TIDAK PERLU load semua gambar ke RAM sekaligus (itu akan boros memori dan berpotensi crash), cukup path dan label saja yang di-precompute.

## Yang TIDAK boleh diubah

- Jangan ubah logic `get_transforms()` (augmentasi/resize harus tetap sama persis)
- Jangan ubah cara split train/val/test ditentukan (tetap ikuti kolom `split` di CSV apa adanya)
- Jangan tambahkan caching gambar terdekode ke disk/memory dalam bentuk apapun kecuali diminta eksplisit — cukup optimasi akses metadata (path+label), karena mengubah cara gambar dibaca berisiko mengubah hasil training secara halus
- Jangan ubah signature fungsi publik (`get_dataloaders`, constructor `PatchDataset`) supaya kode lain yang memanggilnya tidak perlu diubah

## Output yang diharapkan

1. Kode `PatchDataset` yang sudah diperbaiki (drop-in replacement, backward compatible)
2. Kode `get_dataloaders` dengan tambahan `prefetch_factor`
3. Script `benchmark_dataloader.py` yang siap dijalankan untuk verifikasi speed-up
4. Ringkasan singkat di akhir (komentar/docstring) yang menjelaskan apa yang diubah dan kenapa, supaya bisa saya cantumkan sebagai catatan teknis di laporan jika diperlukan