# %%
import pandas as pd
import glob
import os

# Tentukan direktori tempat CSV masing-masing anggota tersimpan
# Sesuaikan path ini dengan struktur folder kamu
CSV_DIR = "./annotations/*.csv"
OUTPUT_FILE = "consensus_review_master.csv"

# %%
def generate_consensus_spreadsheet():
    # 1. Baca semua file CSV hasil anotasi
    csv_files = glob.glob(CSV_DIR)
    if not csv_files:
        print("File CSV tidak ditemukan!")
        return
    
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        # Ambil hanya kolom yang relevan
        temp_df = temp_df[['patch_path', 'class_name', 'label', 'annotator', 'is_skipped']]
        df_list.append(temp_df)
    
    # Gabungkan semua data
    df_all = pd.concat(df_list, ignore_index=True)
    
    # Abaikan yang di-skip saat menghitung konsensus (opsional, tergantung kebijakan)
    df_valid = df_all[df_all['is_skipped'] == False]

    # 2. Pivot data: baris = patch, kolom = annotator
    df_pivot = df_valid.pivot_table(
        index=['patch_path', 'class_name'], 
        columns='annotator', 
        values='label', 
        aggfunc='first'
    ).reset_index()

    # 3. Hitung persetujuan dan konversi ke label numerik
    def calculate_consensus(row):
        # Ambil hanya jawaban dari annotator (mulai dari kolom ke-3)
        answers = row[2:].dropna().tolist()
        
        if not answers:
            return pd.Series([None, 0, True])
        
        # Hitung mayoritas
        majority_text = max(set(answers), key=answers.count)
        agreement_ratio = answers.count(majority_text) / len(answers)
        
        # Flag konflik jika tidak 100% sepakat
        needs_discussion = agreement_ratio < 1.0
        
        # Konversi ke numerik untuk standar training model (0 = healthy, 1 = unhealthy)
        numeric_label = 0 if majority_text == "healthy" else 1
        
        return pd.Series([numeric_label, f"{answers.count(majority_text)}/{len(answers)}", needs_discussion])

    # Terapkan fungsi ke dataframe
    df_pivot[['suggested_numeric_label', 'agreement_level', 'needs_discussion']] = df_pivot.apply(calculate_consensus, axis=1)

    # Urutkan berdasarkan yang paling butuh diskusi (konflik di atas)
    df_pivot = df_pivot.sort_values(by=['needs_discussion', 'patch_path'], ascending=[False, True])

    # 4. Simpan hasilnya
    df_pivot.to_csv(OUTPUT_FILE, index=False)
    print(f"File berhasil digenerate: {OUTPUT_FILE}")
    print(f"Total patch yang butuh diskusi: {df_pivot['needs_discussion'].sum()}")

# %%
if __name__ == "__main__":
    generate_consensus_spreadsheet()